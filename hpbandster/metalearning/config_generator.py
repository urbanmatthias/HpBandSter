from hpbandster.optimizers.config_generators.bohb import BOHB
import numpy as np

class MetaLearningBOHBConfigGenerator(BOHB):
    def __init__(self, warmstarted_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmstarted_model = warmstarted_model

    def new_result(self, *args, **kwargs):
        super().new_result(*args, **kwargs)

        self.warmstarted_model.set_current_kdes(self.kde_models, self.configspace)

        kdes_good = self.warmstarted_model.good_kdes + self.warmstarted_model.current_good_kdes
        kdes_bad  = self.warmstarted_model.bad_kdes  + self.warmstarted_model.current_bad_kdes

        weights = np.array([0] * len(kdes_good))
        budgets = sorted(self.configs.keys())
        for budget in budgets:
            train_configs = np.array(self.configs[budget])
            train_losses =  np.array(self.losses[budget])

            n_good= max(self.min_points_in_model, (self.top_n_percent * train_configs.shape[0])//100 )
            n_bad = max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100)

            # Refit KDE for the current budget
            idx = np.argsort(train_losses)

            train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
            train_data_bad  = self.impute_conditional_data(train_configs[idx[n_good:n_good+n_bad]])

            for i, (good_kde, bad_kde) in enumerate(zip(kdes_good, kdes_bad)):
                good_kde_likelihoods = np.maximum(good_kde.pdf(train_data_good), 1e-32)
                bad_kde_likelihoods = np.maximum(bad_kde.pdf(train_data_bad), 1e-32)
                likelihood = np.prod(good_kde_likelihoods) * np.prod(bad_kde_likelihoods)
                weights[i] += (budget * likelihood)
        self.warmstarted_model.update_weights(weights)
    
    def get_config(self, *args, **kwargs):
        budget = 0
        if len(self.kde_models.keys()) > 0:
            budget = max(self.kde_models.keys())
        self.kde_models[budget + 1] = self.warmstarted_model
        super().get_config(*args, **kwargs)
        del self.kde_models[budget + 1]