import ConfigSpace
import numpy as np
import numpy.random

def make_config_compatible(config, config_space):
    if isinstance(config, dict):
        config = config
    else:
        config = config.get_dictionary()

    # remove illegal values
    config = {k: v for k, v in config.items() 
        if k in config_space.get_hyperparameter_names()
        and config_space.get_hyperparameter(k).is_legal(v)}

    # add values missing for current config space: random value
    for hp in config_space.get_hyperparameters():
        if hp.name not in config and isinstance(hp, ConfigSpace.hyperparameters.Constant):
            config[hp.name] = hp.value
        elif hp.name not in config:
            config[hp.name] = hp.sample(config_space.random)

    # delete values for inactive hyperparameters
    config = ConfigSpace.util.deactivate_inactive_hyperparameters(
									configuration_space=config_space,
									configuration=config)
    return ConfigSpace.Configuration(config_space, config)

def make_bw_compatible(bw, from_configspace, to_configspace):
    bw = insert_constant(bw, from_configspace)
    result = np.zeros(len(to_configspace.get_hyperparameter_names()))
    for i in range(len(bw)):
        j = transform_hyperparameter_index(i, from_configspace, to_configspace)
        if j is not None:
            result[j] = bw[i]
    return filter_constant(result, to_configspace)

def make_vector_compatible(vector, from_configspace, to_configspace, imputer):
    vector = np.asanyarray(vector)
    vector = insert_constant(vector, from_configspace)
    x = np.array(vector).reshape((-1, len(from_configspace.get_hyperparameters())))
    c = np.zeros((x.shape[0], len(to_configspace.get_hyperparameters()))) * np.nan

    # copy given values at correct index
    for i in range(x.shape[1]):
        j = transform_hyperparameter_index(i, from_configspace, to_configspace)
        if j is not None:
            c[:, j] = transform_hyperparameter(from_configspace, to_configspace, i, j, x[:, i])
    return imputer(filter_constant(c, to_configspace))

def transform_hyperparameter_index(idx, from_configspace, to_configspace):
    hp_name = from_configspace.get_hyperparameter_by_idx(idx)
    try:
        return to_configspace.get_idx_by_hyperparameter_name(hp_name)
    except:
        return None

def transform_hyperparameter(from_configspace, to_configspace, from_idx, to_idx, vector):
    from_hp = from_configspace.get_hyperparameter(from_configspace.get_hyperparameter_by_idx(from_idx))
    to_hp   = to_configspace  .get_hyperparameter(to_configspace  .get_hyperparameter_by_idx(to_idx))
    result = np.ones(vector.shape) * np.nan
    for i, v in enumerate(vector):
        try:
            transformed = from_hp._transform(v)
        except:
            print("\nvalue:", v)
            print("hp:", from_hp)
            print("to hp:", to_hp)
            raise
        transformed = transformed[0] if isinstance(transformed, np.ndarray) else transformed
        if to_hp.is_legal(transformed):
            result[i] = to_hp._inverse_transform(transformed)
    return result


def num_non_constant_hps(cs):
    return np.sum(~constant_hypers(cs))


def filter_constant(array, cs):
    if len(array.shape) == 1:
        return array[~constant_hypers(cs)]
    else:
        return array[:, ~constant_hypers(cs)]


def constant_hypers(cs):
    constant_idxs = []
    for hyper in cs.get_hyperparameters():
        idx = cs.get_idx_by_hyperparameter_name(hyper.name)
        if is_constant(hyper):
            constant_idxs.append(idx)
    return np.array([i in constant_idxs for i in range(len(cs.get_hyperparameters()))])


def is_constant(hyper):
    if isinstance(hyper, ConfigSpace.hyperparameters.Constant):
        return True

    elif isinstance(hyper, ConfigSpace.hyperparameters.CategoricalHyperparameter):
        if len(hyper.choices) == 1:
            return True

    return False


def insert_constant(array, cs):
    if len(array.shape) == 1:
        result = np.zeros(len(cs.get_hyperparameters()))
    else:
        result = np.zeros((array.shape[0], len(cs.get_hyperparameters())))

    non_constant_pointer = 0
    for i, constant in enumerate(constant_hypers(cs)):
        if not constant and len(array.shape) == 1:
            result[i] = array[non_constant_pointer]
            non_constant_pointer += 1
        elif not constant:
            result[:, i] = array[:, non_constant_pointer]
            non_constant_pointer += 1
    return result
