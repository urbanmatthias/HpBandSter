import unittest
import ConfigSpace
from hpbandster.metalearning.util import make_config_compatible, make_vector_compatible

class TestMetaLearning(unittest.TestCase):

    def test_util(self):
        h1 = ConfigSpace.hyperparameters.UniformFloatHyperparameter("h1", lower=0, upper=1)
        h2a = ConfigSpace.hyperparameters.CategoricalHyperparameter("h2", choices=["B", "C"])
        h2b = ConfigSpace.hyperparameters.CategoricalHyperparameter("h2", choices=["A", "B", "C"])
        h3 = ConfigSpace.hyperparameters.UniformIntegerHyperparameter("h3", lower=0, upper=100)
        h4 = ConfigSpace.hyperparameters.UniformIntegerHyperparameter("h4", lower=0, upper=100)
        h5 = ConfigSpace.hyperparameters.UniformIntegerHyperparameter("h5", lower=0, upper=100)
        cs1 = ConfigSpace.ConfigurationSpace()
        cs1.add_hyperparameters([h1, h2a, h4, h5])
        cs1.add_conditions([
            ConfigSpace.EqualsCondition(h4, parent=h2a, value="B"),
            ConfigSpace.EqualsCondition(h5, parent=h2a, value="C")
        ])
        cs2 = ConfigSpace.ConfigurationSpace()
        cs2.add_hyperparameters([h2b, h3, h4, h5])
        cs2.add_conditions([
            ConfigSpace.EqualsCondition(h3, parent=h2b, value="A"),
            ConfigSpace.EqualsCondition(h4, parent=h2b, value="B"),
            ConfigSpace.EqualsCondition(h5, parent=h2b, value="C"),
        ])

        config = ConfigSpace.Configuration(cs1, {"h1": 0.2, "h2": "B", "h4": 42})
        compatible = make_config_compatible(config, cs2)
        self.assertEqual(compatible["h2"], "B")
        self.assertEqual(compatible["h4"], 42)
        self.assertEqual(len(compatible.get_dictionary()), 2)

        vector = config.get_array()
        compatible_vector = make_vector_compatible(vector, cs1, cs2)
        compatible_config = ConfigSpace.Configuration(cs2, vector=compatible_vector.reshape((-1, )), allow_inactive_with_values=True)
        compatible_config = ConfigSpace.util.deactivate_inactive_hyperparameters(compatible_config, cs2)
        self.assertEqual(compatible, compatible_config)

        config = ConfigSpace.Configuration(cs2, {"h2": "A", "h3": 97})
        compatible = make_config_compatible(config, cs1)
        self.assertTrue("h1" in compatible)
        self.assertTrue(compatible["h2"] in ["B", "C"])
        self.assertTrue("h4" in compatible if compatible["h2"] == "B" else "h5" in compatible)
        self.assertEqual(len(compatible.get_dictionary()), 3)

        vector = config.get_array()
        compatible_vector = make_vector_compatible(vector, cs2, cs1)
        compatible_config = ConfigSpace.Configuration(cs1, vector=compatible_vector.reshape((-1, )), allow_inactive_with_values=True)
        compatible_config = ConfigSpace.util.deactivate_inactive_hyperparameters(compatible_config, cs1)
        self.assertTrue("h1" in compatible_config)
        self.assertTrue(compatible_config["h2"] in ["B", "C"])
        self.assertTrue("h4" in compatible_config if compatible_config["h2"] == "B" else "h5" in compatible_config)
        self.assertEqual(len(compatible_config.get_dictionary()), 3)