import numpy as np
from hpbandster.metalearning.util import make_vector_compatible, fix_boolean_config
from collections import namedtuple
from hpbandster.core.result import logged_results_to_HBS_result
from hpbandster.optimizers.config_generators.bohb import BOHB as BohbConfigGenerator
from hpbandster.core.dispatcher import Job

class WarmstartedModel():
    def __init__(self, good_kdes, bad_kdes, kde_config_spaces):
        assert len(good_kdes) == len(bad_kdes) == len(kde_config_spaces)
        self.good_kdes = good_kdes
        self.bad_kdes = bad_kdes
        self.kde_config_spaces = kde_config_spaces
        self.current_config_space = None
        self.current_good_kdes = list()
        self.current_bad_kdes = list()
        self.weights = np.array([1] * len(good_kdes))
    
    def get_good_kdes(self):
        return self.good_kdes + self.current_good_kdes

    def get_bad_kdes(self):
        return self.bad_kdes + self.current_bad_kdes
    
    def get_kde_configspaces(self):
        return self.kde_config_spaces + [self.current_config_space] * len(self.current_bad_kdes)
    
    def update_weights(self, weights):
        self.weights = weights
    
    def set_current_config_space(self, current_config_space):
        self.current_config_space = current_config_space
    
    def set_current_kdes(self, kdes):
        self.current_good_kdes.extend([kde["good"] for budget, kde in sorted(kdes.items())])
        self.current_bad_kdes.extend([kde["bad"] for budget, kde in sorted(kdes.items())])
    
    def pdf(self, kdes, vector):
        pdf_values = np.zeros(len(kdes))
        for i, (kde, kde_config_space) in enumerate(zip(kdes, self.get_kde_configspaces())):
            pdf_value = kde.pdf(make_vector_compatible(vector, self.current_config_space, kde_config_space))
            if np.isfinite(pdf_value):
                pdf_values[i] = pdf_value
        return np.sum(pdf_values * self.weights) / np.sum(self.weights)
    
    def __getitem__(self, good_or_bad):
        good = good_or_bad == "good"
        kdes = self.get_good_kdes() if good else self.get_bad_kdes()
        i = np.random.choice(len(kdes), p=self.weights/np.sum(self.weights))
        metalearning_kde = namedtuple("metalearning_kde", ["bw", "data", "pdf"])
        return metalearning_kde(
            bw=kdes[i].bw,
            data=kdes[i].data,
            pdf=lambda vector: self.pdf(kdes, vector)
        )

class WarmstartedModelBuilder():
    def __init__(self):
        self.results = list()
        self.config_spaces = list()
        self.kde_config_spaces = list()
    
    def add_result(self, result, config_space):
        self.results.append(result)
        self.config_spaces.append(config_space)
    
    def build(self):
        good_kdes = list()
        bad_kdes = list()
        for i, (result, config_space) in enumerate(zip(self.results, self.config_spaces)):
            print(i)
            if isinstance(result, str):
                result = logged_results_to_HBS_result(result)
            good, bad = self.train_kde(result, config_space)
            good_kdes.extend(good)
            bad_kdes.extend(bad)
            self.kde_config_spaces.extend([config_space] * len(good))
        return WarmstartedModel(good_kdes, bad_kdes, self.kde_config_spaces)
    
    def train_kde(self, result, config_space):
        cg = BohbConfigGenerator(config_space)

        results_for_budget = dict()
        build_model_jobs = dict()
        id2conf = result.get_id2config_mapping()
        for id in id2conf:
            for r in result.get_runs_by_id(id):
                j = Job(id, config=fix_boolean_config(id2conf[id]['config']), budget=r.budget)
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
        
        good_kdes = [m["good"] for m in cg.kde_models.values()]
        bad_kdes = [m["bad"] for m in cg.kde_models.values()]
        return good_kdes, bad_kdes
