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
        self._current_good_kdes = dict()
        self._current_bad_kdes = dict()
        self._weights = None
        self._weight_history = {"ADDITIONAL_INFO": list()}
        self.sample_budget = None
        self.choose_sample_budget_strategy = "max_available"  # alternative: current
        self.choose_similarity_budget_strategy = "current"  # alternative: max_with_model
        self.weight_type = "likelihood"  # alternatives: max_likelihood
    
    def get_max_budget(self):
        return max(self._good_kdes[0].keys())

    def get_min_budget(self):
        return min(self._good_kdes[0].keys())

    def get_good_kdes(self, budget):
        result =  list(map(lambda x: x[budget], self._good_kdes))
        if budget in self._current_good_kdes:
            result.append(self._current_good_kdes[budget])
        elif self._current_good_kdes:
            result.append(max(self._current_good_kdes.items())[1])
        return result

    def get_bad_kdes(self, budget):
        result =  list(map(lambda x: x[budget], self._bad_kdes))
        if budget in self._current_bad_kdes:
            result.append(self._current_bad_kdes[budget])
        elif self._current_bad_kdes:
            result.append(max(self._current_bad_kdes.items())[1])
        return result
    
    def get_kde_configspaces(self):
        return self._kde_config_spaces + \
            ([self._current_config_space] if self._current_good_kdes else [])
    
    def get_origins(self):
        return self._origins + (["current"] if self._current_good_kdes else [])
    
    def is_current_kde(self, i):
        return i >= len(self._good_kdes)
    
    def update_weights(self, weights, **additional_info):
        # prepare
        weights = np.maximum(np.copy(weights), 0)
        num_kdes =  len(self.get_good_kdes(self.sample_budget))

        # all get equal weight
        if np.sum(weights) == 0:
            weights = np.ones(num_kdes)
        
        # set the weights
        self._weights = weights / np.sum(weights)

        # update weight history
        self._weight_history["ADDITIONAL_INFO"].append(additional_info)
        for i, origin in enumerate(self.get_origins()):
            if origin not in self._weight_history:
                self._weight_history[origin] = list()
            self._weight_history[origin].append(self._weights[i])
    
    def print_weight_history(self, file=sys.stdout):
        for origin, history in self._weight_history.items():
            if origin in ["current", "ADDITIONAL_INFO"]:
                continue
            len_history = len(history)
            print(origin, "\t", ", ".join(map(str, history)), file=file)

        history = np.array(([0] * (len_history - len(self._weight_history["current"]))) + self._weight_history["current"])
        print("current", "\t", ", ".join(map(str, history)), file=file)

        additional_info_keys = set([k for x in self._weight_history["ADDITIONAL_INFO"] for k in x.keys()])
        additional_info_histories = {k: list() for k in additional_info_keys}
        for info in self._weight_history["ADDITIONAL_INFO"]:
            for key in additional_info_keys:
                additional_info_histories[key].append(info[key] if key in info else float("nan"))
        
        for origin, history in additional_info_histories.items():
            print(origin, "\t", ", ".join(map(str, history)), file=file)


    
    def set_current_config_space(self, current_config_space, config_generator):
        self._current_config_space = current_config_space
        self._current_config_space_imputer = config_generator.impute_conditional_data
    
    def set_current_kdes(self, kdes):
        self._current_good_kdes = {budget: kde["good"] for budget, kde in kdes.items()}
        self._current_bad_kdes = {budget: kde["bad"] for budget, kde in kdes.items()}

    def pdf(self, kdes, vector):
        pdf_values = np.zeros(len(kdes))
        # iterate over all kdes
        for i, (kde, kde_config_space) in enumerate(zip(kdes, self.get_kde_configspaces())):
            
            # only query those with positive weight
            if np.isclose(self._weights[i], 0):
                continue

            # query KDE
            imputer = BohbConfigGenerator(kde_config_space).impute_conditional_data
            pdf_value = kde.pdf(make_vector_compatible(vector, self._current_config_space, kde_config_space, imputer))
            if np.isfinite(pdf_value):
                pdf_values[i] = max(0, pdf_value)
        
        # weighting of pdf values
        return np.sum(pdf_values * self._weights)

    def __getitem__(self, good_or_bad):
        good = good_or_bad == "good"
        kdes = self.get_good_kdes(self.sample_budget) if good else self.get_bad_kdes(self.sample_budget)
        i = np.random.choice(len(kdes), p=self._weights)
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
        self._current_good_kdes = dict()
        self._current_bad_kdes = dict()
        self._weights = None
        self._weight_history = {"ADDITIONAL_INFO": list()}

class WarmstartedModelBuilder():
    def __init__(self):
        self.results = list()
        self.config_spaces = list()
        self.kde_config_spaces = list()
        self.origins = list()
    
    def add_result(self, result, config_space, origin):
        try:
            if isinstance(result, str):
                result = logged_results_to_HBS_result(result)
            result.get_incumbent_trajectory()
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
            good_kdes.append(dict(zip(budgets, good)))
            bad_kdes.append(dict(zip(budgets, bad)))
            self.kde_config_spaces.append(config_space)
        return WarmstartedModel(good_kdes, bad_kdes, self.kde_config_spaces, self.origins)
    
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
