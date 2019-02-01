import sys
import numpy as np
from hpbandster.metalearning.util import make_vector_compatible, make_bw_compatible
from collections import namedtuple
from hpbandster.core.result import logged_results_to_HBS_result
from hpbandster.optimizers.config_generators.bohb import BOHB as BohbConfigGenerator
from hpbandster.core.dispatcher import Job

class WarmstartedModel():
    def __init__(self, good_kdes, bad_kdes, kde_config_spaces, origins):
        assert len(good_kdes) == len(bad_kdes) == len(kde_config_spaces)
        self._good_kdes = good_kdes
        self._bad_kdes = bad_kdes
        self._kde_config_spaces = kde_config_spaces
        self._origins = origins
        self._current_config_space = None
        self._current_config_space_imputer = None
        self._current_good_kdes = list()
        self._current_bad_kdes = list()
        self._current_origins = list()
        self._weights = np.array([1] * len(good_kdes))
        self._weight_history = dict()
        self._weight_history["SUM_OF_WEIGHTS"] = list()
        self.normalize_pdf = False
    
    def get_good_kdes(self):
        return self._good_kdes + self._current_good_kdes

    def get_bad_kdes(self):
        return self._bad_kdes + self._current_bad_kdes
    
    def get_kde_configspaces(self):
        return self._kde_config_spaces + [self._current_config_space] * len(self._current_bad_kdes)
    
    def get_origins(self):
        return self._origins + self._current_origins
    
    def is_current_kde(self, i):
        return i >= len(self._good_kdes)
    
    def update_weights(self, weights):
        assert np.all(weights >= 0)

        self._weight_history["SUM_OF_WEIGHTS"].append(np.sum(weights))
        for i, origin in enumerate(self.get_origins()):
            if origin not in self._weight_history:
                self._weight_history[origin] = [0] * (len(self._weight_history["SUM_OF_WEIGHTS"]) - 1)
            self._weight_history[origin].append(weights[i])
        self._weights = weights
    
    def print_weight_history(self, file=sys.stdout):
        history_of_sum = np.array(self._weight_history["SUM_OF_WEIGHTS"])
        print("History of sum of weights:", ", ".join(map(str,self._weight_history["SUM_OF_WEIGHTS"])), file=file)
        print("Normalized weight history:", file=file)
        for origin, history in self._weight_history.items():
            if origin == "SUM_OF_WEIGHTS":
                continue
            history = np.array(history) / history_of_sum
            print(origin, ":", ", ".join(map(str, history)), file=file)
    
    def set_current_config_space(self, current_config_space, config_generator):
        self._current_config_space = current_config_space
        self._current_config_space_imputer = config_generator.impute_conditional_data
    
    def set_current_kdes(self, kdes):
        self._current_good_kdes = [kde["good"] for budget, kde in sorted(kdes.items())]
        self._current_bad_kdes = [kde["bad"] for budget, kde in sorted(kdes.items())]
        self._current_origins = ["current:" + str(b) for b in sorted(kdes.keys())]

    def pdf(self, kdes, vector):
        pdf_values = np.zeros(len(kdes))
        # iterate over all kdes
        for i, (kde, kde_config_space) in enumerate(zip(kdes, self.get_kde_configspaces())):
            
            # only query those with positive weight
            if self._weights[i] <= 0:
                continue

            # query KDE
            imputer = BohbConfigGenerator(kde_config_space).impute_conditional_data
            pdf_value = kde.pdf(make_vector_compatible(vector, self._current_config_space, kde_config_space, imputer))
            if np.isfinite(pdf_value):
                pdf_values[i] = max(0, pdf_value)
        
        # weighting of pdf values
        if self.normalize_pdf:
            return np.sum(pdf_values * self._weights) / np.sum(self._weights)
        return np.sum(pdf_values * self._weights)

    def __getitem__(self, good_or_bad):
        good = good_or_bad == "good"
        kdes = self.get_good_kdes() if good else self.get_bad_kdes()
        i = np.random.choice(len(kdes), p=self._weights/np.sum(self._weights))
        metalearning_kde = namedtuple("metalearning_kde", ["bw", "data", "pdf"])
        return metalearning_kde(
            bw=make_bw_compatible(kdes[i].bw, self.get_kde_configspaces()[i], self._current_config_space),
            data=make_vector_compatible(kdes[i].data, self.get_kde_configspaces()[i],
                self._current_config_space, self._current_config_space_imputer),
            pdf=lambda vector: self.pdf(kdes, vector)
        )
    
    def clean(self):
        self._current_config_space = None
        self._current_config_space_imputer = None
        self._current_good_kdes = list()
        self._current_bad_kdes = list()
        self._current_origins = list()
        self._weights = np.array([1] * len(self._good_kdes))
        self._weight_history = dict()
        self._weight_history["SUM_OF_WEIGHTS"] = list()

class WarmstartedModelBuilder():
    def __init__(self):
        self.results = list()
        self.config_spaces = list()
        self.kde_config_spaces = list()
        self.origins = list()
        self.origins_with_budget = list()
    
    def add_result(self, result, config_space, origin):
        try:
            if isinstance(result, str):
                result = logged_results_to_HBS_result(result)
            result.get_incumbent_trajectory(bigger_is_better=False, non_decreasing_budget=False)
        except:
            print("Did not add empty result")
            return False

        self.results.append(result)
        self.config_spaces.append(config_space)
        self.origins.append(origin)
        return True
    
    def build(self):
        good_kdes = list()
        bad_kdes = list()
        for i, (result, config_space, origin) in enumerate(zip(self.results, self.config_spaces, self.origins)):
            print(i)
            if isinstance(result, str):
                try:
                    result = logged_results_to_HBS_result(result)
                except:
                    continue
            good, bad, budgets = self.train_kde(result, config_space)
            good_kdes.extend(good)
            bad_kdes.extend(bad)
            self.kde_config_spaces.extend([config_space] * len(good))
            self.origins_with_budget.extend([origin + ":" + str(b) for b in budgets])
        return WarmstartedModel(good_kdes, bad_kdes, self.kde_config_spaces, self.origins_with_budget)
    
    def train_kde(self, result, config_space):
        cg = BohbConfigGenerator(config_space)

        results_for_budget = dict()
        build_model_jobs = dict()
        id2conf = result.get_id2config_mapping()
        for id in id2conf:
            for r in result.get_runs_by_id(id):
                j = Job(id, config=id2conf[id]['config'], budget=r.budget)
                if r.loss is None:
                    r.loss = float('inf')
                if r.info is None:
                    r.info = dict()
                j.result = {'loss': r.loss, 'info': r.info}
                j.error_logs = r.error_logs

                if r.budget not in results_for_budget:
                    results_for_budget[r.budget] = list()
                results_for_budget[r.budget].append(j)

                if r.loss is not None and r.budget not in build_model_jobs:
                    build_model_jobs[r.budget] = j
                    continue
                cg.new_result(j, update_model=False)
        for j in build_model_jobs.values():
            cg.new_result(j, update_model=True)
        
        good_kdes = [m["good"] for b, m in sorted(cg.kde_models.items())]
        bad_kdes = [m["bad"] for b, m in sorted(cg.kde_models.items())]
        budgets = sorted(cg.kde_models.keys())
        return good_kdes, bad_kdes, budgets
