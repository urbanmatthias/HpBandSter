import numpy as np
from hpbandster.metalearning.util import make_config_compatible, fix_boolean_config
from hpbandster.optimizers.config_generators.bohb import BOHB as BohbConfigGenerator
from hpbandster.core.result import logged_results_to_HBS_result
from ConfigSpace import Configuration
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import clone

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
    def __init__(self, model=None, cost_calculation=None):
        self.results = list()
        self.config_spaces = list()
        self.incumbents = None
        self.models = None
        self.model = model or RandomForestRegressor(n_estimators=100)
        self.cost_calculation = cost_calculation or np.argsort

    def add_result(self, result, config_space):
        self.results.append(result)
        self.config_spaces.append(config_space)

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
        ranks = list()
        for i, (result, config_space) in enumerate(zip(self.results, self.config_spaces)):
            print(i)
            if isinstance(result, str):
                result = logged_results_to_HBS_result(result)
            try:
                model, imputer = self._train_single_model(result, config_space)
            except:
                print("No model trained")
                raise
            X = np.vstack([make_config_compatible(i, config_space).get_array() for i in self.incumbents])
            predicted_losses = model.predict(imputer(X))
            print(predicted_losses)
            ranks.append(self.cost_calculation(predicted_losses).reshape((-1, 1)))
        ranks = np.hstack(ranks)  # incumbents x estimated performances
        return ranks

    def _get_incumbents(self):
        self.incumbents = list()
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
            self.incumbents.append(Configuration(config_space, fix_boolean_config(incumbent)))

    def _train_single_model(self, result, config_space):
        config_generator = BohbConfigGenerator(config_space)
        X, y = list(), list()
        id2config = result.get_id2config_mapping()
        for config_id, config in id2config.items():
            config = Configuration(config_space, fix_boolean_config(config["config"]))
            runs = result.get_runs_by_id(config_id)
            runs = [r for r in runs if r.loss is not None]
            if len(runs) == 0:
                continue
            best_run = min(runs, key=lambda run: run.loss)
            X.append(config.get_array())
            y.append(best_run.loss)
        X_list = X
        y_list = y

        X = np.vstack(X_list * 10)
        y = np.array(y_list * 10)
        X = config_generator.impute_conditional_data(X)
        model = clone(self.model)
        model.fit(X, y)
        print(model.score(config_generator.impute_conditional_data(np.vstack(X_list)), np.array(y_list)))

        return model, config_generator.impute_conditional_data