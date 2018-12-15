import ConfigSpace

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

def make_vector_compatible(vector, from_searchspace, to_searchspace):
    config = ConfigSpace.Configuration(configuration_space=from_searchspace, vector=vector, allow_inactive_with_values=True)
    return make_config_compatible(config, to_searchspace).get_array()

def fix_boolean_config(config):
    return {k: v if not isinstance(v, bool) else str(v) for k, v in config.items()}