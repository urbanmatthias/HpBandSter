from hpbandster.optimizers.config_generators.bohb import BOHB
from hpbandster.metalearning.util import make_vector_compatible, filter_constant
import ConfigSpace
import numpy as np
import math
from statsmodels.nonparametric.kernel_density import gpke, _adjust_shape, KDEMultivariate

class MetaLearningBOHBConfigGenerator(BOHB):
    def __init__(self, warmstarted_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmstarted_model = warmstarted_model
        self.warmstarted_model.clean()
        self.observations = dict()
        self.weights = None
        self.learning_rate = 0.01
        self.num_steps = 1000

    def new_result(self, job, *args, **kwargs):
        super().new_result(job, update_model=True, force_model_update=(self.warmstarted_model.choose_similarity_budget_strategy == "current"))

        budget = job.kwargs["budget"]
        config = ConfigSpace.Configuration(configuration_space=self.configspace, values=job.kwargs["config"])
        loss = job.result["loss"] if (job.result is not None and "loss" in job.result) else float("inf")

        if config not in self.observations:
            self.observations[config] = list()
        self.observations[config].append((budget, loss))
    
    def get_config(self, budget, *args, **kwargs):
        max_budget_with_model = self.warmstarted_model.get_min_budget()
        if self.kde_models:
            max_budget_with_model = max(self.kde_models.keys())

        similarity_budget = sample_budget = budget
        if self.warmstarted_model.choose_similarity_budget_strategy == "max_with_model" or budget > max_budget_with_model:
            similarity_budget = max_budget_with_model
        if self.warmstarted_model.choose_sample_budget_strategy == "max_available":
            sample_budget = self.warmstarted_model.get_max_budget()

        # prepare model
        self.warmstarted_model.set_current_config_space(self.configspace, self)
        self.warmstarted_model.sample_budget = sample_budget

        self.update_warmstarted_model_weights(
            select_observation_strategy=FilterObservations(similarity_budget, self),
            select_kde_budget=similarity_budget
        )

        # impute model
        self.kde_models[max_budget_with_model + 1] = self.warmstarted_model
        result = super().get_config(budget=budget, *args, **kwargs)
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
        train_configs = np.array(list(map(
            lambda x: filter_constant(ConfigSpace.Configuration.get_array(x), self.configspace), train_configs)))

        # split into good and bad
        n_good = (self.top_n_percent * train_configs.shape[0]) // 100
        n_bad = ((100-self.top_n_percent)*train_configs.shape[0]) // 100

        idx = np.argsort(train_losses)

        train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
        train_data_bad  = self.impute_conditional_data(train_configs[idx[n_good:n_good+n_bad]])

        self._update_weights(train_data_good, train_data_bad, kdes_good, kdes_bad, kde_configspaces, select_kde_budget)
    
    def _update_weights(self, train_data_good, train_data_bad, kdes_good, kdes_bad, kde_configspaces, select_kde_budget):
        # no data to set weight 
        if self.weights is None or train_data_good.shape[0] + train_data_bad.shape[0] == 0:
            self.weights = np.ones(len(kdes_good))

        # maximize likelihood of ensemble to set the weights
        elif self.warmstarted_model.weight_type == "max_likelihood":
            self.weights = np.zeros(len(kdes_good))
            self.weights[np.random.choice(len(kdes_good))] = 1
            matrix = self._get_likelihood_matrix(train_data_good, train_data_bad, kdes_good, kdes_bad, kde_configspaces)
            for _ in range(self.num_steps):
                gradient = self._get_weight_gradient(matrix)
                self.weights = self.weights + gradient * self.learning_rate

                # project back in allowed weight space
                sorted_indices = np.argsort(self.weights)
                weight_sum = np.sum(self.weights)
                for i in range(sorted_indices.shape[0]):
                    num_pos = sorted_indices.shape[0] - i
                    reduce = self.weights[sorted_indices[i]]
                    if weight_sum - reduce * num_pos <= 1:
                        self.weights += (1 - weight_sum) / self.weights.shape[0]
                        break
                    weight_sum -= reduce * num_pos
                    self.weights[sorted_indices[i:]] -= reduce

        # use the likelihood or log_likelihood as weight
        elif self.warmstarted_model.weight_type == "likelihood":
            self.weights = np.sum(np.log(np.maximum(1e-32, self._get_likelihood_matrix(
                train_data_good, train_data_bad, kdes_good, kdes_bad, kde_configspaces))), axis=0)
            self.weights = np.exp(self.weights - np.max(self.weights))  # subtract maximum for numerical reasons
          
        self.warmstarted_model.update_weights(self.weights, 
                                              similarity_budget=select_kde_budget,
                                              num_observation=train_data_good.shape[0] + train_data_bad.shape[0])

    def _get_weight_gradient(self, matrix):
        sum_over_models = np.sum(matrix, axis=1).reshape((-1, 1))
        gradient = np.sum(matrix / sum_over_models, axis=0)
        return gradient
    
    def _get_likelihood_matrix(self, train_data_good, train_data_bad, kdes_good, kdes_bad, kde_configspaces):  # datapoints x models
        likelihood_matrix = np.empty((train_data_good.shape[0] + train_data_bad.shape[0], len(kdes_good)))
        for i, (good_kde, bad_kde, kde_configspace) in enumerate(zip(kdes_good, kdes_bad, kde_configspaces)):
            train_data_good_compatible = train_data_good
            train_data_bad_compatible = train_data_bad

            # compute likelihood of kde given observation
            pdf = KDEMultivariate.pdf  # leave_given_out_pdf
            if not self.warmstarted_model.is_current_kde(i):
                imputer = BOHB(kde_configspace).impute_conditional_data
                train_data_good_compatible = make_vector_compatible(train_data_good, self.configspace, kde_configspace, imputer)
                train_data_bad_compatible  = make_vector_compatible(train_data_bad, self.configspace, kde_configspace, imputer)
                pdf = KDEMultivariate.pdf

            good_kde_likelihoods = np.nan_to_num(pdf(good_kde, train_data_good_compatible))
            bad_kde_likelihoods = np.nan_to_num(pdf(bad_kde, train_data_bad_compatible))
            likelihood_matrix[:, i] = np.append(good_kde_likelihoods, bad_kde_likelihoods)
        return likelihood_matrix


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
