import numpy as np
from hpbandster.metalearning.util import make_vector_compatible
from collections import namedtuple
from hpbandster.core.result import logged_results_to_HBS_result
from hpbandster.optimizers.config_generators.bohb import BOHB as BohbConfigGenerator
from hpbandster.core.dispatcher import Job

class WarmstartedModel():
    def __init__(self, good_kdes, bad_kdes, config_spaces):
        assert len(good_kdes) == len(bad_kdes)
        self.good_kdes = good_kdes
        self.bad_kdes = bad_kdes
        self.config_spaces = config_spaces
        self.config_space = None
        self.weights = np.array([1] * len(good_kdes))
        self.last_configs = None
        self.las_performances = None
    
    def update_weights(self, configs, performances):
        self.last_configs = configs
        self.las_performances = performances
        pass
    
    def add_kde(self, good_kde, bad_kde, config_space):
        self.good_kdes.append(good_kde)
        self.bad_kdes.append(bad_kde)
        self.config_spaces.append(config_space)
        self.update_weights(self.last_configs, self.las_performances)
    
    def set_config_space(self, config_space):
        self.config_space = config_space
    
    def pdf(self, kdes, vector):
        pdf_values = np.zeros(len(kdes))
        for i, (kde, config_space) in enumerate(zip(kdes, self.config_spaces)):
            pdf_values[i] = kde.pdf(make_vector_compatible(vector, self.config_space, config_space))
        return np.sum(pdf_values * self.weights) / np.sum(self.weights)
    
    def __getitem__(self, good_or_bad):
        good = good_or_bad == "good"
        kdes = self.good_kdes if good else self.bad_kdes
        i = np.random.choice(len(kdes), p=self.weights/np.sum(self.weights))
        metalearning_kde = namedtuple("metalearning kde", ["bw", "data", "pdf"])
        return metalearning_kde(
            bw=kdes[i].bw,
            data=kdes[i].data,
            pdf=lambda vector: self.pdf(kdes, vector)
        )

class WarmstartedModelBuilder():
    def __init__(self):
        self.results = list()
        self.config_spaces = list()
    
    def add_result(self, result, config_space):
        self.results.append(result)
        self.config_spaces.append(config_space)
    
    def build(self):
        good_kdes = list()
        bad_kdes = list()
        for result, config_space in self.config_spaces:
            if isinstance(result, str):
                result = logged_results_to_HBS_result(result)
            good, bad = self.train_kde(result, config_space)
            good_kdes.extend(good)
            bad_kdes.extend(bad)
        return WarmstartedModel(good_kdes, bad_kdes, self.config_spaces)
    
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
        
        good_kdes = [m["good"] for m in cg.kde_models.values()]
        bad_kdes = [m["bad"] for m in cg.kde_models.values()]
        return good_kdes, bad_kdes
