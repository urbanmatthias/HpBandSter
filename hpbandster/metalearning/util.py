import ConfigSpace

def make_config_compatible(config, config_space):
    # remove illegal values
    config = {k: v for k, v in config.items() 
        if k in config_space.get_hyperparameter_names()
        and config_space.get_hyperparameter(k).is_legal(v)}

    # add values missing for current config space: random value
    for hp in config_space.get_hyperparameters():
        if hp.name not in config:
            config[hp.name] = hp.sample(config_space.random)

    # delete values for inactive hyperparameters
    config_tmp = dict()
    config_obj = ConfigSpace.Configuration(config_space, config, allow_inactive_with_values=True)
    for hp_name in config_space.get_active_hyperparameters(config_obj):
        config_tmp[hp_name] = config[hp_name]
    while set(config.keys()) != set(config_tmp.keys()):
        config = config_tmp
        config_tmp = dict()
        config_obj = ConfigSpace.Configuration(config_space, config, allow_inactive_with_values=True)
        for hp_name in config_space.get_active_hyperparameters(config_obj):
            config_tmp[hp_name] = config[hp_name]
    return ConfigSpace.Configuration(config_space, config)