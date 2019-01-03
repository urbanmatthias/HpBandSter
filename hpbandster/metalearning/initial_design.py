import numpy as np
from hpbandster.metalearning.util import make_config_compatible
from hpbandster.optimizers.config_generators.bohb import BOHB as BohbConfigGenerator
from hpbandster.core.result import logged_results_to_HBS_result
from ConfigSpace import Configuration
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from multiprocessing import Pool

class InitialDesign():
    def __init__(self, configs):
        self.pointer = 0
        self.configs = configs
        self.config_space = None
    
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
        self.pointer += 1
        return result


class InitialDesignLearner():
    def add_result(self, result, config_space):
        raise NotImplementedError
    
    def learn(self, num_configs=None):
        raise NotImplementedError


class Hydra(InitialDesignLearner):
    def __init__(self, cost_estimation_model=None, cost_calculation=np.argsort, num_processes=0):
        self.results = list()
        self.config_spaces = list()
        self.exact_cost_models = list()

        self.incumbents = None
        self.budgets = None

        self.cost_estimation_model = cost_estimation_model or RandomForestRegressor(n_estimators=100)
        self.cost_calculation = cost_calculation
        self.num_repeat_imputation = 10
        self.num_processes = num_processes

    def add_result(self, result, config_space, exact_cost_model=None):
        self.results.append(result)
        self.config_spaces.append(config_space)
        self.exact_cost_models.append(exact_cost_model)

    def learn(self, num_configs=None):
        cost_matrix = self._get_cost_matrix()
        print(cost_matrix)

        initial_design = []
        num_configs = num_configs or len(self.results)
        for _ in range(num_configs):
            new_incumbent = self._greedy_step(initial_design, cost_matrix)
            initial_design.append(new_incumbent)
            print("Initial Design:", initial_design, "Cost:", self._cost(initial_design, cost_matrix))
        initial_design_configs = [self.incumbents[i] for i in initial_design]
        return InitialDesign(initial_design_configs)

    def _greedy_step(self, initial_design, cost_matrix):
        available_incumbents = set(range(len(self.incumbents))) - set(initial_design)
        return min(available_incumbents, key=lambda inc: self._cost(initial_design + [inc], cost_matrix))

    def _cost(self, initial_design, cost_matrix):
        return np.mean(np.min(cost_matrix[np.array(initial_design), :], axis=0))

    def _get_cost_matrix(self):
        self._get_incumbents()
        if self.num_processes == 0:
            ranks = map(self._get_cost_matrix_column, enumerate(zip(self.results, self.config_spaces, self.exact_cost_models)))
        else:
            with Pool(self.num_processes) as p:
                ranks = p.map(self._get_cost_matrix_column, enumerate(zip(self.results, self.config_spaces, self.exact_cost_models)))
        ranks = np.hstack(ranks)  # incumbents x estimated performances on datasets
        return ranks

    def _get_incumbents(self):
        self.incumbents = list()
        self.budgets = list()
        for result, config_space in zip(self.results, self.config_spaces):
            if isinstance(result, str):
                result = logged_results_to_HBS_result(result)
            id2config = result.get_id2config_mapping()
            try:
                trajectory = result.get_incumbent_trajectory(bigger_is_better=False, non_decreasing_budget=False)
            except:
                print("No incumbent found!")
                continue
            print("Incumbent loss:",  trajectory["losses"][-1])
            incumbent = id2config[trajectory["config_ids"][-1]]["config"]
            self.incumbents.append(Configuration(config_space, incumbent))
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
            except:
                print("No model trained")
                raise
            X = np.vstack([make_config_compatible(i, config_space).get_array() for i in self.incumbents])
            predicted_losses = model.predict(imputer(X))
        
        # compute the cost exactly using given exact cost model
        else:
            predicted_losses = np.zeros(len(self.incumbents))
            with exact_cost_model as m:
                for i, (incumbent, budget) in enumerate(zip(self.incumbents, self.budgets)):
                    predicted_losses[i] = m.evaluate(make_config_compatible(incumbent, config_space), budget)
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