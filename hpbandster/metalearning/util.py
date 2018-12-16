import ConfigSpace
import numpy as np
from hpbandster.optimizers.config_generators.bohb import BOHB as BohbConfigGenerator

def make_config_compatible(config, config_space):
    if isinstance(config, dict):
        config = fix_boolean_config(config)
    else:
        config = config.get_dictionary()

    # remove illegal values
    config = {k: v for k, v in config.items() 
        if k in config_space.get_hyperparameter_names()
        and config_space.get_hyperparameter(k).is_legal(v)}

    # add values missing for current config space: random value
    for hp in config_space.get_hyperparameters():
        if hp.name not in config:
            config[hp.name] = hp.sample(config_space.random)

    # delete values for inactive hyperparameters
    config = ConfigSpace.util.deactivate_inactive_hyperparameters(
									configuration_space=config_space,
									configuration=fix_boolean_config(config)
									)
    return ConfigSpace.Configuration(config_space, config)

def make_vector_compatible(vector, from_configspace, to_configspace):
    x = np.array(vector).reshape((-1, len(from_configspace.get_hyperparameters())))
    c = np.zeros((x.shape[0], len(to_configspace.get_hyperparameters()))) * np.nan

    # copy given values at correct index
    for i in range(x.shape[1]):
        j = transform_hyperparameter_index(i, from_configspace, to_configspace)
        if j is not None:
            c[:, j] = x[:, i]
    cg = BohbConfigGenerator(to_configspace)
    return cg.impute_conditional_data(c)

def transform_hyperparameter_index(idx, from_configspace, to_configspace):
    hp_name = from_configspace.get_hyperparameter_by_idx(idx)
    try:
        return to_configspace.get_idx_by_hyperparameter_name(hp_name)
    except:
        return None

def fix_boolean_config(config):
    return {k: v if not isinstance(v, bool) else str(v) for k, v in config.items()}