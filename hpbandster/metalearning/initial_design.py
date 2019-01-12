import numpy as np
import time
from hpbandster.metalearning.util import make_config_compatible
from hpbandster.optimizers.config_generators.bohb import BOHB as BohbConfigGenerator
from hpbandster.core.result import logged_results_to_HBS_result
from ConfigSpace import Configuration
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import multiprocessing as mp

class InitialDesign():
    def __init__(self, configs, origins):
        self.pointer = 0
        self.configs = configs
        self.config_space = None
        self.origins = origins
    
    def set_config_space(self, config_space):
        self.config_space = config_space
    
    def __len__(self):
        return len(self.configs)
    
    def __iter__(self):
        self.pointer = 0
        return self
    
    def __next__(self):
        if self.pointer >= len(self.configs):
            raise StopIteration
        if self.config_space:
            result = make_config_compatible(self.configs[self.pointer].get_dictionary(), self.config_space)
        else:
            result = self.configs[self.pointer]
        origin = self.origins[self.pointer]
        self.pointer += 1
        return result, origin


class InitialDesignLearner():
    def add_result(self, result, config_space):
        raise NotImplementedError
    
    def learn(self, num_configs=None):
        raise NotImplementedError

def rank(x):
    return np.argsort(np.argsort(x))

class Hydra(InitialDesignLearner):
    def __init__(self, cost_estimation_model=None, cost_calculation=rank, num_processes=0, 
            distributed=False, run_id=0, working_dir=".", master=True, host='localhost'):
        self.results = list()
        self.config_spaces = list()
        self.exact_cost_models = list()
        self.origins = list()

        self.incumbents = None
        self.budgets = None
        self.incumbent_origins = None

        self.cost_estimation_model = cost_estimation_model or RandomForestRegressor(n_estimators=100)
        self.cost_calculation = cost_calculation
        self.num_repeat_imputation = 10
        self.num_processes = num_processes
        self.distributed = distributed
        self.run_id = run_id
        self.working_dir = working_dir
        self.master = master
        self.host = host

    def add_result(self, result, config_space, origin, exact_cost_model=None):
        self.results.append(result)
        self.config_spaces.append(config_space)
        self.exact_cost_models.append(exact_cost_model)
        self.origins.append(origin)

    def learn(self, num_configs=None):
        cost_matrix = self._get_cost_matrix()
        print(cost_matrix)
        if self.distributed and not self.master:
            return None

        initial_design = []
        num_configs = num_configs or len(self.results)
        for _ in range(num_configs):
            if len(initial_design) == len(self.incumbents):
                break
            new_incumbent = self._greedy_step(initial_design, cost_matrix)
            initial_design.append(new_incumbent)
            print("Initial Design:", initial_design, "Cost:", self._cost(initial_design, cost_matrix))
        initial_design_configs = [self.incumbents[i] for i in initial_design]
        initial_design_origins = [self.incumbent_origins[i] for i in initial_design]
        return InitialDesign(initial_design_configs, initial_design_origins)

    def _greedy_step(self, initial_design, cost_matrix):
        available_incumbents = set(range(len(self.incumbents))) - set(initial_design)
        return min(available_incumbents, key=lambda inc: self._cost(initial_design + [inc], cost_matrix))

    def _cost(self, initial_design, cost_matrix):
        return np.mean(np.min(cost_matrix[np.array(initial_design), :], axis=0))

    def _get_cost_matrix(self):
        self._get_incumbents()
        if self.num_processes == 0 and not self.distributed:
            print("A")
            ranks = map(self._get_cost_matrix_column, enumerate(zip(self.results, self.config_spaces, self.exact_cost_models)))
        elif not self.distributed:
            print("B")
            with mp.Pool(self.num_processes) as p:
                ranks = p.map(self._get_cost_matrix_column, enumerate(zip(self.results, self.config_spaces, self.exact_cost_models)))
        else:
            print("C")
            with DistributedMap(self.run_id, self.working_dir, self.master, self.host) as p:
                ranks = p.map(self._get_cost_matrix_column, enumerate(zip(self.results, self.config_spaces, self.exact_cost_models)))
            if not self.master:
                return None
        ranks = np.hstack(ranks) # incumbents x estimated performances on datasets
        return ranks

    def _get_incumbents(self):
        self.incumbents = list()
        self.incumbent_origins = list()
        self.budgets = list()
        for i, (result, config_space) in enumerate(zip(self.results, self.config_spaces)):
            try:
                if isinstance(result, str):
                    result = logged_results_to_HBS_result(result)
                id2config = result.get_id2config_mapping()
                trajectory = result.get_incumbent_trajectory(bigger_is_better=False, non_decreasing_budget=False)
            except Exception as e:
                print(e)
                print("No incumbent found!")
                continue
            print("Incumbent loss:",  trajectory["losses"][-1], " budget:", trajectory["budgets"][-1])
            incumbent = id2config[trajectory["config_ids"][-1]]["config"]
            self.incumbents.append(Configuration(config_space, incumbent))
            self.incumbent_origins.append(self.origins[i])
            self.budgets.append(trajectory["budgets"][-1])

    def _get_cost_matrix_column(self, args):
        column_idx, result, config_space, exact_cost_model = args[0], args[1][0], args[1][1], args[1][2]
        print(column_idx)
        if isinstance(result, str):
            result = logged_results_to_HBS_result(result)

        # no model given to compute cost exactly. Evaluate by learning the cost for each run using cost_estimation_model.
        if exact_cost_model is None:
            try:
                model, imputer = self._train_cost_estimation_model(result, config_space)
                X = np.vstack([make_config_compatible(i, config_space).get_array() for i in self.incumbents])
                predicted_losses = model.predict(imputer(X))
            except:
                print("No model trained")
                predicted_losses = np.array([0] * len(self.incumbents))
        
        # compute the cost exactly using given exact cost model
        else:
            predicted_losses = np.zeros(len(self.incumbents))
            with exact_cost_model as m:
                for i, (incumbent, budget) in enumerate(zip(self.incumbents, self.budgets)):
                    try:
                        predicted_losses[i] = m.evaluate(make_config_compatible(incumbent, config_space), budget)
                    except Exception as e:
                        print(e)
                        predicted_losses[i] = float("inf")
        return self.cost_calculation(predicted_losses).reshape((-1, 1))

    def _train_cost_estimation_model(self, result, config_space):
        config_generator = BohbConfigGenerator(config_space)
        X, y = list(), list()
        id2config = result.get_id2config_mapping()
        for config_id, config in id2config.items():
            config = Configuration(config_space, config["config"])
            runs = result.get_runs_by_id(config_id)
            runs = [r for r in runs if r.loss is not None]
            if len(runs) == 0:
                continue
            best_run = min(runs, key=lambda run: run.loss)
            X.append(config.get_array())
            y.append(best_run.loss)
        X_list = X
        y_list = y

        X = np.vstack(X_list * self.num_repeat_imputation)
        y = np.array(y_list * self.num_repeat_imputation)
        X = config_generator.impute_conditional_data(X)
        model = clone(self.cost_estimation_model)
        model.fit(X, y)
        print(model.score(config_generator.impute_conditional_data(np.vstack(X_list)), np.array(y_list)))

        return model, config_generator.impute_conditional_data

import Pyro4
import os
import pickle
import traceback

@Pyro4.expose
class DistributedMap():
    def __init__(self, run_id, working_dir, master, host):
        self.run_id = run_id
        self.working_dir = working_dir
        self.master = master
        self.host = host

        self.daemon = None
        self.uri = None
        self.results = list()
        self.iterator_done = False

        self.iterator = None
        self.func = None
    
    def __enter__(self):
        if self.master:
            self.old_server_type = Pyro4.config.SERVERTYPE
            Pyro4.config.SERVERTYPE = "multiplex"
            self.daemon = Pyro4.Daemon(host=self.host)
            self.uri = self.daemon.register(self)
        return self
    
    def __exit__(self, error_type, error_value, error_traceback):
        if self.master:
            Pyro4.config.SERVERTYPE = self.old_server_type
        return error_type is not None


    def map(self, func, iterator):
        self.iterator = enumerate(iterator)
        self.func = func

        print("distributed map called")

        if self.master:
            return self.start_master()
        return self.start_worker()

    
    def start_worker(self):
        print("start worker")
        while True:
            try:
                with open(os.path.join(self.working_dir, "distributed_map_uri_%s.txt" % self.run_id), "r") as f:
                    self.uri = f.readline().strip()
                    print("loaded uri", self.uri)
                    if self.uri == "shutdown":
                        return None
                    master = Pyro4.Proxy(self.uri)
                    i = master.get_item()
                    while i is not None:
                        item = self.get_item_by_idx(i)
                        print("working on", i)
                        result = self.func(item)
                        master.register_result(i, result.tolist())
                        i = master.get_item()
            except Exception as e:
                print(e)
                traceback.print_exc()
                print("".join(Pyro4.util.getPyroTraceback()))
                print("Sleeping")
                time.sleep(30)

    
    def start_master(self):
        print("start_master")
        with open(os.path.join(self.working_dir, "distributed_map_uri_%s.txt" % self.run_id), "w") as f:
            print(self.uri, file=f)

        self.daemon.requestLoop()
        print("distributed map finished")
        return self.results
        
    # WORKER
    def get_item_by_idx(self, idx):
        for i, item in self.iterator:
            if i == idx:
                return item
    
    # MASTER
    def get_item(self):
        try:
            i, item = next(self.iterator)
            self.results.append("NO_RESULT")
            print("assigned", i)
            return i
        except StopIteration:
            print("all jobs assigned")
            self.iterator_done = True
            with open(os.path.join(self.working_dir, "distributed_map_uri_%s.txt" % self.run_id), "w") as f:
                print("shutdown", file=f)
            if self.iterator_done and not any(isinstance(r, str) and r == "NO_RESULT" for r in self.results):
                self.daemon.shutdown()
            return None
    
    # MASTER
    def register_result(self, i, result):
        print("Registered result", i, result)
        self.results[i] = np.array(result)
        print("All jobs assigned:", self.iterator_done)
        print("Waiting for results:", any(isinstance(r, str) and r == "NO_RESULT" for r in self.results))

        if self.iterator_done and not any(isinstance(r, str) and r == "NO_RESULT" for r in self.results):
            self.daemon.shutdown()