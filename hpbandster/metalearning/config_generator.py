from hpbandster.optimizers.config_generators.bohb import BOHB
from hpbandster.metalearning.util import make_vector_compatible
import ConfigSpace
import numpy as np
import math
from statsmodels.nonparametric.kernel_density import gpke, _adjust_shape, KDEMultivariate

class MetaLearningBOHBConfigGenerator(BOHB):
    def __init__(self, warmstarted_model, bigger_budget_is_better, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmstarted_model = warmstarted_model
        self.warmstarted_model.clean()
        self.bigger_budget_is_better = bigger_budget_is_better
        self.observations = dict()
        # self.num_nonzero_weight = False
        self.num_nonzero_weight = 15

    def new_result(self, job, *args, **kwargs):
        super().new_result(job, *args, **kwargs)

        # save for each config a loss value: either loss evaluated on heighest budget or best loss observed so far
        budget = job.kwargs["budget"]
        config = ConfigSpace.Configuration(configuration_space=self.configspace, values=job.kwargs["config"])
        loss = job.result["loss"] if (job.result is not None and "loss" in job.result) else float("inf")

        if config not in self.observations:
            self.observations[config] = list()
        self.observations[config].append((budget, loss))
    
    def get_config(self, *args, **kwargs):
        max_budget_with_model = self.warmstarted_model.get_min_budget()
        if self.kde_models:
            max_budget_with_model = max(self.kde_models.keys())

        # prepare model
        self.warmstarted_model.set_current_config_space(self.configspace, self)
        self.warmstarted_model.sample_budget = self.warmstarted_model.get_max_budget()
        max
        self.update_warmstarted_model_weights(
            select_observation_strategy=FilterObservations(max_budget_with_model, self),
            select_kde_budget=max_budget_with_model
        )

        # impute model
        self.kde_models[max_budget_with_model + 1] = self.warmstarted_model
        result = super().get_config(*args, **kwargs)
        del self.kde_models[max_budget_with_model + 1]
        return result
    
    def update_warmstarted_model_weights(self, select_observation_strategy, select_kde_budget):
        select_observation_strategy.set_observations(self.observations)
        config_to_loss = select_observation_strategy

        # get kdes from warmstarted model
        self.warmstarted_model.set_current_kdes(self.kde_models)
        kdes_good = self.warmstarted_model.get_good_kdes(select_kde_budget)
        kdes_bad  = self.warmstarted_model.get_bad_kdes(select_kde_budget)
        kde_configspaces = self.warmstarted_model.get_kde_configspaces()

        # read observations and transform to array
        train_configs = list(config_to_loss.keys())
        train_losses = np.array(list(map(lambda c:config_to_loss[c], train_configs)))
        train_configs = np.array(list(map(ConfigSpace.Configuration.get_array, train_configs)))

        # split into good and bad
        n_good = (self.top_n_percent * train_configs.shape[0]) // 100
        n_bad = ((100-self.top_n_percent)*train_configs.shape[0]) // 100

        idx = np.argsort(train_losses)

        train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
        train_data_bad  = self.impute_conditional_data(train_configs[idx[n_good:n_good+n_bad]])

        # update the weights
        self.warmstarted_model.update_weights(
            self._get_weights(train_data_good, train_data_bad, kdes_good, kdes_bad, kde_configspaces)
        )

    def _get_weights(self, train_data_good, train_data_bad, kdes_good, kdes_bad, kde_configspaces):
        # iterate over kdes
        likelihood_sums = np.zeros(len(kdes_good), dtype=float)
        for i, (good_kde, bad_kde, kde_configspace) in enumerate(zip(kdes_good, kdes_bad, kde_configspaces)):
            train_data_good_compatible = train_data_good
            train_data_bad_compatible = train_data_bad

            # compute likelihood of kde given observation
            pdf = pdf = KDEMultivariate.pdf  # leave_given_out_pdf
            if not self.warmstarted_model.is_current_kde(i):
                imputer = BOHB(kde_configspace).impute_conditional_data
                train_data_good_compatible = make_vector_compatible(train_data_good, self.configspace, kde_configspace, imputer)
                train_data_bad_compatible  = make_vector_compatible(train_data_bad, self.configspace, kde_configspace, imputer)
                pdf = KDEMultivariate.pdf

            good_kde_likelihoods = np.maximum(np.nan_to_num(pdf(good_kde, train_data_good_compatible)), 1e-32)
            bad_kde_likelihoods = np.maximum(np.nan_to_num(pdf(bad_kde, train_data_bad_compatible)), 1e-32)

            likelihood_sum = np.sum(np.append(good_kde_likelihoods, bad_kde_likelihoods))
            likelihood_sums[i] += likelihood_sum

        return self._likelihoods_to_weights(likelihood_sums, len(kdes_good))

    def _likelihoods_to_weights(self, likelihoods, num_weights):
        weights = likelihoods

        # if all weights are zero, all models are equally likely
        if np.sum(weights) == 0 and (not self.num_nonzero_weight or num_weights < self.num_nonzero_weight):
            weights = np.ones(num_weights / num_weights)

        # only num_nonzero_weight should have positive weight
        elif np.sum(weights) == 0:
            weights = np.zeros(num_weights)
            weights[np.random.choice(np.array(list(range(num_weights))), size=self.num_nonzero_weight, replace=False)] = 1 / self.num_nonzero_weight
        elif self.num_nonzero_weight:
            weights[np.argsort(weights)[:-self.num_nonzero_weight]] = 0
        return weights


# def leave_given_out_pdf(kde, data_predict):
#     data_predict = _adjust_shape(data_predict, kde.k_vars)

#     pdf_est = []
#     for i in range(np.shape(data_predict)[0]):
#         data = kde.data[np.sum(np.abs(kde.data - data_predict[i, :]), axis=1) != 0]

#         pdf_est.append(gpke(kde.bw, data=data,
#                             data_predict=data_predict[i, :],
#                             var_type=kde.var_type) / kde.nobs)

#     pdf_est = np.squeeze(pdf_est)
#     return pdf_est


class FilterObservations():
    def __init__(self, budget, cg):
        self.cg = cg
        self.budget = budget
    
    def set_observations(self, observations):
        self.observations = observations
    
    def __getitem__(self, key):
        return [config for budget, config in self.observations[key] if budget == self.budget][0]
    
    def keys(self):
        return [key for key, value in self.observations.items() for budget, config in value  if budget == self.budget]