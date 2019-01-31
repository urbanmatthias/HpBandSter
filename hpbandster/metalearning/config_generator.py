from hpbandster.optimizers.config_generators.bohb import BOHB
from hpbandster.metalearning.util import make_vector_compatible
import numpy as np

class MetaLearningBOHBConfigGenerator(BOHB):
    def __init__(self, warmstarted_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmstarted_model = warmstarted_model

    def new_result(self, *args, **kwargs):
        super().new_result(*args, **kwargs)
        self.warmstarted_model.set_current_kdes(self.kde_models)

        # calculate weights
        kdes_good = self.warmstarted_model.get_good_kdes()
        kdes_bad  = self.warmstarted_model.get_bad_kdes()
        kde_configspaces = self.warmstarted_model.get_kde_configspaces()

        budgets = sorted(self.configs.keys())
        weights = np.zeros(len(kdes_good), dtype=float)

        # consider all models of all budgets
        for budget in budgets:
            train_configs = np.array(self.configs[budget])
            train_losses =  np.array(self.losses[budget])

            n_good= (self.top_n_percent * train_configs.shape[0]) // 100
            n_bad = ((100-self.top_n_percent)*train_configs.shape[0]) // 100

            idx = np.argsort(train_losses)

            train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
            train_data_bad  = self.impute_conditional_data(train_configs[idx[n_good:n_good+n_bad]])

            # calculate the sum of likelihoods
            for i, (good_kde, bad_kde, kde_configspace) in enumerate(zip(kdes_good, kdes_bad, kde_configspaces)):
                imputer = BOHB(kde_configspace).impute_conditional_data
                train_data_good_compatible = make_vector_compatible(train_data_good, self.configspace, kde_configspace, imputer)
                train_data_bad__compatible  = make_vector_compatible(train_data_bad, self.configspace, kde_configspace, imputer)

                good_kde_likelihoods = np.maximum(np.nan_to_num(good_kde.pdf(train_data_good_compatible)), 0)
                bad_kde_likelihoods = np.maximum(np.nan_to_num(bad_kde.pdf(train_data_bad__compatible)), 0)

                sum_of_likelihood = np.sum(good_kde_likelihoods) + np.sum(bad_kde_likelihoods)
                weights[i] += budget * sum_of_likelihood
        if np.sum(weights) == 0:
            weights = np.ones(len(kdes_good))
        self.warmstarted_model.update_weights(weights)
    
    def get_config(self, *args, **kwargs):
        budget = 0
        self.warmstarted_model.set_current_config_space(self.configspace, self)
        if len(self.kde_models.keys()) > 0:
            budget = max(self.kde_models.keys())
        self.kde_models[budget + 1] = self.warmstarted_model
        result = super().get_config(*args, **kwargs)
        del self.kde_models[budget + 1]
        return result