import unittest
import os
import ConfigSpace
import numpy as np
from hpbandster.metalearning.util import make_config_compatible, make_vector_compatible
from hpbandster.metalearning.initial_design import Hydra
from hpbandster.metalearning.model_warmstarting import WarmstartedModelBuilder, WarmstartedModel
from hpbandster.core.result import logged_results_to_HBS_result
from sklearn.linear_model import LinearRegression

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
    
    def test_initial_design(self):
        # test train cost estimation model
        hydra = Hydra(cost_estimation_model=LinearRegression())
        result1 = logged_results_to_HBS_result(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result1"))
        result2 = logged_results_to_HBS_result(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result2"))
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameters([
            ConfigSpace.hyperparameters.UniformFloatHyperparameter("A", lower=0, upper=10),
            ConfigSpace.hyperparameters.UniformFloatHyperparameter("B", lower=0, upper=10)
        ])
        model, _ = hydra._train_cost_estimation_model(result1, cs)
        np.testing.assert_array_almost_equal(model.coef_, np.array([0., -1]))
        np.testing.assert_almost_equal(model.intercept_, 0.8)

        # test get cost matrix column
        hydra = Hydra(cost_estimation_model=LinearRegression())
        class ExactCostModel():
            def __init__(self, idx):
                self.idx = idx

            def __enter__(self):
                return self
            
            def __exit__(self, error_type, error_value, error_traceback):
                return error_type is None
            
            def evaluate(self, incumbent, budget):
                return abs(8 - incumbent["B"]) if self.idx == 1 else abs(8 - (incumbent["B"] - incumbent["A"]))
    
        hydra.incumbents = [ConfigSpace.Configuration(cs, values={"A": 9, "B": 9}),
                            ConfigSpace.Configuration(cs, values={"A": 0, "B": 10}),
                            ConfigSpace.Configuration(cs, values={"A": 2, "B": 8})]
        hydra.budgets = [0, 0, 0]
        r = hydra._get_cost_matrix_column((0, (result1, cs, None)))
        np.testing.assert_array_almost_equal(r, np.array([[1], [0], [2]]))
        r = hydra._get_cost_matrix_column((0, (result1, cs, ExactCostModel(1))))
        np.testing.assert_array_almost_equal(r, np.array([[1], [2], [0]]))

        # test get_incumbents
        hydra = Hydra(cost_estimation_model=LinearRegression())
        hydra.results = [result1, result2]
        hydra.config_spaces = [cs, cs]
        hydra.origins = ["result1", "result2"]
        hydra._get_incumbents()
        self.assertEqual(hydra.incumbents, [ConfigSpace.Configuration(cs, values={"A": 2, "B": 8}),
                                            ConfigSpace.Configuration(cs, values={"A": 2, "B": 10})])
        self.assertEqual(hydra.budgets, [9.0, 9.0])

        # test get_cost_matrix
        hydra = Hydra(cost_estimation_model=LinearRegression())
        hydra.results = [result1, result2]
        hydra.config_spaces = [cs, cs]
        hydra.origins = ["result1", "result2"]
        hydra.exact_cost_models = [None, None]
        r = hydra._get_cost_matrix()
        np.testing.assert_array_almost_equal(r, np.array([[1, 1], [0, 0]]))
        hydra.exact_cost_models = [ExactCostModel(1), ExactCostModel(2)]
        r = hydra._get_cost_matrix()
        np.testing.assert_array_almost_equal(r, np.array([[0, 1], [1, 0]]))

        # test hydra alg
        hydra = Hydra(cost_estimation_model=LinearRegression())
        hydra.incumbents = [0, 1, 2, 3]
        cost_matrix = np.array([[0.1, 0.8, 0.7, 0.6],
                                [0.4, 0.0, 0.7, 0.5],
                                [0.9, 0.3, 0.2, 0.3],
                                [0.2, 0.9, 0.9, 0.1]])
        self.assertAlmostEqual(hydra._cost([0], cost_matrix), 2.2 / 4)
        self.assertAlmostEqual(hydra._cost([1, 2], cost_matrix), 0.9 / 4)
        self.assertAlmostEqual(hydra._cost([0, 1, 2, 3], cost_matrix), 0.1)
        self.assertEqual(hydra._greedy_step([], cost_matrix), 1)
        self.assertEqual(hydra._greedy_step([1], cost_matrix), 2)
        self.assertEqual(hydra._greedy_step([1, 2], cost_matrix), 3)
        self.assertEqual(hydra._greedy_step([1, 2, 3], cost_matrix), 0)

        # learn 
        hydra = Hydra(cost_estimation_model=LinearRegression())
        hydra.add_result(result1, cs, "result1", None)
        hydra.add_result(result2, cs, "result2", None)
        self.assertEqual(list(hydra.learn())[0][0].get_dictionary(), {"A": 2.0, "B": 10.0})
        self.assertEqual(list(hydra.learn())[1][0].get_dictionary(), {"A": 2.0, "B": 8.0})

        hydra = Hydra(cost_estimation_model=LinearRegression())
        hydra.add_result(result1, cs, "result1", ExactCostModel(1))
        hydra.add_result(result2, cs, "result2", ExactCostModel(2))
        self.assertEqual(list(hydra.learn())[0][0].get_dictionary(), {"A": 2.0, "B": 8.0})
        self.assertEqual(list(hydra.learn())[1][0].get_dictionary(), {"A": 2.0, "B": 10.0})
    
    def test_model_warmstarting(self):
        hydra = Hydra(cost_estimation_model=LinearRegression())
        result1 = logged_results_to_HBS_result(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result1"))
        result2 = logged_results_to_HBS_result(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result2"))
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameters([
            ConfigSpace.hyperparameters.UniformFloatHyperparameter("A", lower=0, upper=10),
            ConfigSpace.hyperparameters.UniformFloatHyperparameter("B", lower=0, upper=10)
        ])

        # train model
        builder = WarmstartedModelBuilder()
        r = builder.train_kde(result1, cs)
        self.assertEqual(len(r[0]),1)
        self.assertEqual(len(r[1]), 1)
        self.assertEqual(r[0][0].data.shape, (3, 2))
        self.assertEqual(r[1][0].data.shape, (6, 2))

        # build
        builder.add_result(result1, cs, "result1")
        builder.add_result(result2, cs, "result2")
        r = builder.build()
        self.assertEqual(len(r.good_kdes), 2)
        self.assertEqual(len(r.bad_kdes), 2)
        self.assertEqual(len(r.kde_config_spaces), 2)