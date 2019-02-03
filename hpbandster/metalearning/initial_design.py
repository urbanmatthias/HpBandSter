import numpy as np
import time
import os
from hpbandster.metalearning.util import make_config_compatible
from hpbandster.optimizers.config_generators.bohb import BOHB as BohbConfigGenerator
from hpbandster.core.result import logged_results_to_HBS_result
from ConfigSpace import Configuration
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import multiprocessing as mp
from NamedAtomicLock import NamedAtomicLock
import traceback

class InitialDesign():
    def __init__(self, configs, origins, num_configs_per_sh_iter, budgets):
        self.pointer = 0
        self.configs = configs
        self.config_space = None
        self.origins = origins
        self.num_configs_per_sh_iter = num_configs_per_sh_iter
        self.budgets = budgets
    
    def set_config_space(self, config_space):
        self.config_space = config_space
    
    def get_num_configs(self):
        return self.num_configs_per_sh_iter
    
    def get_total_budget(self):
        return sum(map(lambda x: x[0] * x[1], zip(self.budgets, self.num_configs_per_sh_iter)))

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


def rank(x):
    return np.argsort(np.argsort(x))

def normalized_distance_to_min(x):
    maximum = np.max(x[np.isfinite(x)])
    minimum = np.min(x[np.isfinite(x)])
    result = (x - minimum) / (maximum - minimum)
    result[~np.isfinite(result)] = 2
    return result


class Hydra():
    def __init__(self, normalize_loss=normalized_distance_to_min, bigger_is_better=True):
        self.incumbents = None
        self.origins = None

        self.loss_matrices = dict()

        self.normalize_loss = normalize_loss
        self.bigger_is_better = bigger_is_better

    def set_incumbent_losses(self, losses, incumbent_dict):
        self.origins = sorted(list(incumbent_dict.keys()))
        self.incumbents = [incumbent_dict[origin] for origin in self.origins]
        budgets = set(map(lambda x:x["budget"], losses))
        self.loss_matrices = {b: np.zeros((len(self.origins), len(self.origins))) for b in budgets}

        origin_to_id = {origin: i for i, origin in enumerate(self.origins)}

        for l in losses:
            incumbent_id = origin_to_id[l["incumbent_origin"]]
            dataset_id = origin_to_id[l["dataset_origin"]]
            self.loss_matrices[l["budget"]][incumbent_id, dataset_id] = l["loss"]
    
    def learn(self, convergence_threshold, max_total_budget, max_initial_design_size=float("inf")):
        largest_budget = max(self.loss_matrices.keys())
        max_num_max_budget = int(max_total_budget // largest_budget)
        max_num_sh_iter = len(self.loss_matrices)
        initial_designs = list()
        costs = list()

        print("Try single SH-iter")
        r = self._learn(convergence_threshold, max_total_budget, num_max_budget=max_num_max_budget, num_sh_iter=1, max_size=max_initial_design_size)
        if r is not None:
            initial_designs.append(r[0])
            costs.append(r[1])

        for num_sh_iter in range(2, max_num_sh_iter + 1):
            for num_max_budget in range(1, max_num_max_budget + 1):
                print("Try %s SH iterations with %s configurations evaluated at max budget" % (num_sh_iter, num_max_budget))
                r = self._learn(convergence_threshold, max_total_budget, num_max_budget=num_max_budget, num_sh_iter=num_sh_iter, max_size=max_initial_design_size)
                if r is not None:
                    initial_designs.append(r[0])
                    costs.append(r[1])
        try:
            idx = np.argmin([x.get_total_budget() for x in initial_designs])  # return initial design with lowest total budget
            # idx = np.argmin(costs)  # return initial design with lowest cost
            return initial_designs[idx], costs[idx]
        except:
            return None, None

    def _learn(self, convergence_threshold, max_total_budget, num_max_budget, num_sh_iter, max_size):
        initial_design = []
        cost = float("inf")
        for _ in range(len(self.incumbents)):
            new_incumbent, cost = self._greedy_step(initial_design, num_max_budget, num_sh_iter)
            initial_design.append(new_incumbent)
            print("Initial Design:", list(map(lambda x: self.origins[x], initial_design)), "Cost:", cost, end="\r")
            if cost <= convergence_threshold:
                print("\nCost lower than given threshold: Coverged.")
                break
            if self.get_total_budget(len(initial_design), num_max_budget, num_sh_iter) > max_total_budget:
                print("\nTotal budget of initial design larger than given threshold: Failed.")
                return None
            if len(initial_design) > max_size:
                print("\nInitial design is larger than given maximum size. Failed")
                return None
        if cost > convergence_threshold:
            print("\nCould not find initial design with cost lower than given threshold")
            return None

        initial_design_configs = [self.incumbents[i] for i in initial_design]
        initial_design_origins = [self.origins[i] for i in initial_design]

        num_configs_per_sh_iter = self.get_num_configs_per_sh_iter(len(initial_design), num_max_budget, num_sh_iter)
        budgets = self.get_budgets(num_sh_iter)
        print("Found initial design with cost %s and total budget %s" % (cost, self.get_total_budget(len(initial_design), num_max_budget, num_sh_iter)))
        return InitialDesign(initial_design_configs, initial_design_origins, num_configs_per_sh_iter, budgets), cost
    
    def get_budgets(self, num_sh_iter):
        return sorted(list(self.loss_matrices.keys()))[-num_sh_iter:]
    
    def get_total_budget(self, num_min_budget, num_max_budget, num_sh_iter):
        ns = self.get_num_configs_per_sh_iter(num_min_budget, num_max_budget, num_sh_iter)
        budgets = self.get_budgets(num_sh_iter)
        return sum(map(lambda x: x[0] * x[1], zip(budgets, ns)))
    
    def get_num_configs_per_sh_iter(self, num_min_budget, num_max_budget, num_sh_iter):
        if num_sh_iter <= 1:
            return [num_min_budget]
        num_max_budget = min(num_min_budget, num_max_budget)
        s = num_sh_iter - 1
        eta = (num_min_budget / num_max_budget) ** (1 / s)
        n0 = num_min_budget
        ns = [max(int(n0 * (eta ** (-i))), 1) for i in range(s + 1)]
        return ns

    def _greedy_step(self, initial_design, num_max_budget, num_sh_iter):
        # check number of configurations per sh iteration to estimate cost
        num_configs_per_sh_iter = self.get_num_configs_per_sh_iter(len(initial_design) + 1, num_max_budget, num_sh_iter)

        # return best from available incumbents
        available_incumbents = set(range(len(self.incumbents))) - set(initial_design)

        def cost_of_incumbent(inc):
            return self._cost(initial_design + [inc], num_configs_per_sh_iter)

        inc = min(available_incumbents, key=cost_of_incumbent)
        return inc, cost_of_incumbent(inc)

    def _cost(self, initial_design, num_configs_per_sh_iter):
        dataset_costs = list()
        budgets = self.get_budgets(len(num_configs_per_sh_iter))
        assert len(budgets) == len(num_configs_per_sh_iter)

        # iterate over all datasets
        for d in range(len(self.origins)):
            current_cost = float("inf")
            current_configs = np.array(initial_design)
    
            # simulate SH
            for i, b in enumerate(budgets):

                # cost in current SH iteration
                losses = self.loss_matrices[b][:, d]
                normalized_losses = self.normalize_loss(losses)

                cost = np.min(normalized_losses[current_configs])
                if not self.bigger_is_better or b == max(budgets):
                    current_cost = min(cost, current_cost)
                
                # update configs for next SH iteration
                if b != max(budgets):
                    ranks = np.argsort(np.argsort(losses[current_configs]))
                    current_configs = current_configs[ranks < num_configs_per_sh_iter[i + 1]]
            
            assert np.isfinite(current_cost)
            dataset_costs.append(current_cost)

        return np.mean(dataset_costs)


class LossMatrixComputation():
    def __init__(self, bigger_is_better=True, lock_dir=None):
        self.results = list()
        self.config_spaces = list()
        self.origins = list()
        self.exact_cost_models = list()
        self.bigger_is_better = bigger_is_better
        self.budgets = None
        self.lock_dir = lock_dir

    def add_result(self, result, config_space, origin, exact_cost_model):
        try:
            if isinstance(result, str):
                r = logged_results_to_HBS_result(result)
            r.get_incumbent_trajectory(bigger_is_better=False, non_decreasing_budget=False)
        except:
            print("Did not add empty result")
            return False

        all_runs = r.get_all_runs(only_largest_budget=False)
        budgets = sorted(set(map(lambda r: r.budget, all_runs)))
        if self.budgets is None:
            self.budgets = budgets
        else:
            assert self.budgets == budgets, "Budgets of all results need to be equivalent"

        self.results.append(result)
        self.config_spaces.append(config_space)
        self.origins.append(origin)
        self.exact_cost_models.append(exact_cost_model)
        return True
    
    def get_num_entries(self):
        assert self.budgets is not None, "Add at least one result first"
        return len(self.results) * len(self.results)
    
    def write_loss(self, directory, entry, num_files=1):
        try:
            os.mkdir(directory)
        except:
            pass

        print("start computing loss matrix_entry", time.time())
        loss_dict, incumbent_id, dataset_id = self.compute_cost_matrix_entry(entry)
        incumbent_origin = self.origins[incumbent_id]
        dataset_origin = self.origins[dataset_id]
        path = os.path.join(directory, "loss_matrix_%s.txt" % (entry % num_files))
        print("done computing loss_matrix_entry", time.time())

        lock_name = ("lock:" + os.path.abspath(path)).replace(os.sep, '')
        lock = NamedAtomicLock(lock_name, lockDir=self.lock_dir)
        try:
            print("acquire named lock:", lock_name, time.time())
            lock.acquire()
            print("success", time.time())
            with open(path, "a") as f:
                for budget, loss in loss_dict.items():
                    print("\t".join(map(str, [entry, loss, incumbent_origin, dataset_origin, budget])), file=f)
        finally:
            print("release lock", time.time())
            lock.release()
    
    def read_loss(self, directory):
        assert os.path.exists(directory)
        files = next(os.walk(directory))[2]
        paths = [os.path.join(directory, f) for f in files if f.startswith("loss_matrix")]
        losses = list()
        for path in paths:
            with open(path, "r") as f:
                for line in f:
                    _, loss, incumbent_origin, dataset_origin, budget = line.split("\t")
                    entry = {
                        "loss": float(loss),
                        "incumbent_origin": incumbent_origin,
                        "dataset_origin": dataset_origin,
                        "budget": float(budget)
                    }
                    losses.append(entry)
        incumbents = {self.origins[i]: self._get_incumbent(i) for i in range(len(self.origins))}
        return losses, incumbents

    def compute_cost_matrix_entry(self, entry):
        assert entry > 0, "entries start with 1"
        entry -= 1
        num_matrix_entries = len(self.results) * len(self.results)
        assert entry < num_matrix_entries, "Given entry is too large"
        incumbent_id = entry % len(self.results)
        dataset_id = entry // len(self.results)

        incumbent = self._get_incumbent(incumbent_id)
        exact_cost_model = self.exact_cost_models[dataset_id]
        config_space = self.config_spaces[dataset_id]

        print("Start compting loss matrix entry", time.time())
        loss_dict = dict()
        for budget in self.budgets:
            print("Start compting loss matrix entry for budget", budget, time.time())
            with exact_cost_model as m:
                try:
                    loss = m.evaluate(make_config_compatible(incumbent, config_space), budget)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    loss = float("inf")
                loss_dict[budget] = loss
            print("Done compting loss matrix entry for budget", time.time())
        return loss_dict, incumbent_id, dataset_id

    def _get_incumbent(self, i):
        result = self.results[i]
        config_space = self.config_spaces[i]
        
        if isinstance(result, str):
            result = logged_results_to_HBS_result(result)
        id2config = result.get_id2config_mapping()
        trajectory = result.get_incumbent_trajectory(bigger_is_better=self.bigger_is_better,
            non_decreasing_budget=self.bigger_is_better)

        incumbent = id2config[trajectory["config_ids"][-1]]["config"]
        return Configuration(config_space, incumbent)
