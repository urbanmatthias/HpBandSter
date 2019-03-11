import numpy as np
from hpbandster.metalearning.model_warmstarting import WarmstartedModelBuilder
from hpbandster.metalearning.config_generator import MetaLearningBOHBConfigGenerator
from hpbandster.core.result import logged_results_to_HBS_result
from hpbandster.core.dispatcher import Job
import ConfigSpace

class Evaluator():

    def __init__(self, bigger_budget_is_better, cg_kwargs=None):
        self.results = list()
        self.config_spaces = list()
        self.origins = list()
        self.bigger_budget_is_better = bigger_budget_is_better
        self.cg_kwargs = cg_kwargs
    
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
    
    def evaluate_cv(self, num_splits=5):
        raise NotImplementedError()
    
    def evaluate(self, valid_split=0.2):
        valid_indices = np.random.choice(len(self.results), size=int(len(self.results) * valid_split))
        builder = WarmstartedModelBuilder()
        eval_dataset = list()
        for i in range(len(self.results)):
            if i in valid_indices:
                eval_dataset.append((self.results[i], self.config_spaces[i], self.origins[i]))
            else:
                builder.add_result(self.results[i], self.config_spaces[i], self.origins[i])
        warmstarted_model = builder.build()

        scores = list()
        for result, config_space, origin in eval_dataset:
            scores.append(self._evaluate(warmstarted_model, result, config_space, origin))
        return np.mean(scores)
    
    def _evaluate(self, warmstarted_model, result, config_space, origin):
        runs = sorted(result.get_all_runs(), key=lambda x: x.time_stamps["submitted"])
        id2config = result.get_id2config_mapping()

        warmstarted_model.clean()
        config_generator = self.get_config_generator(warmstarted_model, config_space)
        warmstarted_model.set_current_config_space(config_space, config_generator)

        # divide config ids in good and bad
        config_ids_to_best_run = dict()
        for run in runs:
            if ((run.config_id not in config_ids_to_best_run) or
                (config_generator.bigger_budget_is_better and run.budget > config_ids_to_best_run[run.config_id].budget) or
                (not config_generator.bigger_budget_is_better and run.loss is not None and run.loss < config_ids_to_best_run[run.config_id].loss)):
                config_ids_to_best_run[run.config_id] = run

        # iterate over runs and check the warmstarted model for ei
        scheduled_config_ids = set()
        score = 0
        for i, run in enumerate(runs):
            print(i, len(runs))
            if run.config_id not in scheduled_config_ids:
                scheduled_config_ids.add(run.config_id)
                config = ConfigSpace.Configuration(config_space, id2config[run.config_id]["config"])

                l = warmstarted_model["good"].pdf
                g = warmstarted_model["bad"].pdf
                ei_func = lambda x: max(1e-32, g(x))/max(l(x),1e-32)

                ei = ei_func(config.get_array())
                is_good = self.is_good(run.config_id, scheduled_config_ids, config_ids_to_best_run, config_generator.top_n_percent)
                score += ei * ((100 - config_generator.top_n_percent) if is_good else -config_generator.top_n_percent)
                print(score)

            # register result to cg --> update model weights
            job = Job(run.config_id)
            job.kwargs = {"config": config.get_dictionary(), "budget": run.budget}
            job.result = {"loss": run.loss}
            config_generator.new_result(job)
        return score
    
    def get_config_generator(self, warmstarted_model, configspace):
        kwargs = {
            "warmstarted_model": warmstarted_model,
            "bigger_budget_is_better": self.bigger_budget_is_better,
            "configspace": configspace,
            "min_points_in_model": None,
            "top_n_percent": 15,
            "num_samples": 64,
            "random_fraction": 1/3,
            "bandwidth_factor": 3,
            "min_bandwidth" : 1e-3
        }
        if self.cg_kwargs:
            kwargs.update(self.cg_kwargs)
        return MetaLearningBOHBConfigGenerator(**kwargs)
    
    def is_good(self, config_id, scheduled_set, config_ids_to_best_run, top_n_percent):
        sorted_config_ids = sorted(scheduled_set, key=lambda x: config_ids_to_best_run[x].loss)
        num_good = len(sorted_config_ids) * (top_n_percent / 100)
        print("Total:", len(sorted_config_ids))
        print("Num good:", num_good)
        print("Index:", sorted_config_ids.index(config_id))
        print("--->", sorted_config_ids.index(config_id) < num_good)
        return sorted_config_ids.index(config_id) < num_good