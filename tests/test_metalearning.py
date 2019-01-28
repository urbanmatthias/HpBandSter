import unittest
import os
import ConfigSpace
import numpy as np
from hpbandster.metalearning.util import make_config_compatible, make_vector_compatible
from hpbandster.metalearning.initial_design import Hydra, LossMatrixComputation, rank
from hpbandster.metalearning.model_warmstarting import WarmstartedModelBuilder, WarmstartedModel
from hpbandster.core.result import logged_results_to_HBS_result
from sklearn.linear_model import LinearRegression
from hpbandster.optimizers.config_generators.bohb import BOHB as BohbConfigGenerator

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
        imputer = BohbConfigGenerator(cs2).impute_conditional_data
        compatible_vector = make_vector_compatible(vector, cs1, cs2, imputer)
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
        imputer = BohbConfigGenerator(cs1).impute_conditional_data
        compatible_vector = make_vector_compatible(vector, cs2, cs1, imputer)
        compatible_config = ConfigSpace.Configuration(cs1, vector=compatible_vector.reshape((-1, )), allow_inactive_with_values=True)
        compatible_config = ConfigSpace.util.deactivate_inactive_hyperparameters(compatible_config, cs1)
        self.assertTrue("h1" in compatible_config)
        self.assertTrue(compatible_config["h2"] in ["B", "C"])
        self.assertTrue("h4" in compatible_config if compatible_config["h2"] == "B" else "h5" in compatible_config)
        self.assertEqual(len(compatible_config.get_dictionary()), 3)
    
    def test_initial_design(self):
        loss_matrix_computation = LossMatrixComputation()
        losses = loss_matrix_computation.read_loss(os.path.join(os.path.dirname(os.path.abspath(__file__)), "losses.txt"))[0]

        # test train cost estimation model
        hydra = Hydra(normalize_loss=rank)
        self.assertEquals(hydra.get_num_configs_per_sh_iter(9, 1, 3), [9, 3, 1])
        self.assertEquals(hydra.get_num_configs_per_sh_iter(9, 3, 2), [9, 3])
        self.assertEquals(hydra.get_num_configs_per_sh_iter(27, 3, 3), [27, 9, 3])
        self.assertEquals(hydra.get_num_configs_per_sh_iter(1, 3, 5), [1, 1, 1, 1, 1])

        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameters([
            ConfigSpace.hyperparameters.UniformFloatHyperparameter("A", lower=0, upper=10)
        ])
        incumbent_dict = {"1": ConfigSpace.Configuration(configuration_space=cs, values={"A": 1}),
                          "2": ConfigSpace.Configuration(configuration_space=cs, values={"A": 2}),
                          "3": ConfigSpace.Configuration(configuration_space=cs, values={"A": 3}),
                          "4": ConfigSpace.Configuration(configuration_space=cs, values={"A": 4})}
        hydra.set_incumbent_losses(losses, incumbent_dict)
        self.assertEqual(hydra.origins, ["1", "2", "3", "4"])
        self.assertEqual(list(map(lambda x: x.get_dictionary()["A"], hydra.incumbents)), [1, 2, 3, 4])

        np.testing.assert_almost_equal(hydra.loss_matrices[1],
            np.array([[0.1, 1.0, 1.0, 0.7],
                      [0.7, 0.2, 0.5, 0.7],
                      [0.2, 0.3, 0.2, 0.8],
                      [0.3, 0.8, 0.4, 0.1]]))
        np.testing.assert_almost_equal(hydra.loss_matrices[2],
            np.array([[0.1, 0.1, 0.6, 0.2],
                      [0.0, 0.1, 0.1, 0.4],
                      [0.4, 0.5, 0.2, 0.3],
                      [0.5, 0.5, 0.4, 0.1]]))
        np.testing.assert_almost_equal(hydra.loss_matrices[4],
            np.array([[0.0, 0.8, 0.7, 0.2],
                      [0.0, 0.2, 0.1, 0.8],
                      [0.1, 0.0, 0.1, 0.2],
                      [0.6, 0.0, 0.8, 0.0]]))
        
        # test get_budgets
        self.assertEqual(hydra.get_budgets(1), [4])
        self.assertEqual(hydra.get_budgets(2), [2, 4])
        self.assertEqual(hydra.get_budgets(3), [1, 2, 4])

        # test get_total_budget
        self.assertEqual(hydra.get_total_budget(27, 3, 3), 27 * 1 + 9 * 2 + 3 * 4)
        self.assertEqual(hydra.get_total_budget(9, 3, 2), 9 * 2 + 3 * 4)
        self.assertEqual(hydra.get_total_budget(9, 1, 3), 9 * 1 + 3 * 2 + 1 * 4)
        self.assertEqual(hydra.get_total_budget(1, 3, 3), 1 * 1 + 1 * 2 + 1 * 4)
        
        # test _cost
        self.assertEqual(hydra._cost([0], [1, 1, 1]), 6/4)
        self.assertEqual(hydra._cost([2], [1, 1, 1]), 5/4)
        self.assertEqual(hydra._cost([0, 2], [2, 1, 1]), 2/4)
        self.assertEqual(hydra._cost([0, 2], [2, 2, 1]), 5/4)
        self.assertEqual(hydra._cost([0, 1, 2, 3], [4, 2, 1]), 3/4)
        self.assertEqual(hydra._cost([0, 1], [2, 2, 2]), 3/4)

        # test _greedy_step
        self.assertEqual(hydra._greedy_step([], 1, 3), (2, (5/4)))
        self.assertEqual(hydra._greedy_step([2], 1, 3), (0, 0.5))
        self.assertEqual(hydra._greedy_step([0, 2], 1, 3), (3, 0.25))

        # test learn
        initial_design, cost = hydra._learn(convergence_threshold=0.3, max_total_budget=50, num_max_budget=1, num_sh_iter=3, max_size=4)
        self.assertEqual(initial_design.origins, ["3", "1", "4"])
        self.assertEqual(list(map(lambda x: x.get_dictionary()["A"], initial_design.configs)), [3, 1, 4])
        self.assertEqual(initial_design.num_configs_per_sh_iter, [3, 1, 1])

        initial_design, cost = hydra.learn(convergence_threshold=0.3, max_total_budget=50)
        self.assertEqual(initial_design.origins, ["3", "1", "4"])
        self.assertEqual(list(map(lambda x: x.get_dictionary()["A"], initial_design.configs)), [3, 1, 4])
        self.assertEqual(initial_design.num_configs_per_sh_iter, [3, 1, 1])
    
    def test_loss_matrix_computation(self):
        result1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result1")
        result2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result2")
        empty_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "empty_result")

        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameters([
            ConfigSpace.hyperparameters.UniformFloatHyperparameter("A", lower=0, upper=10),
            ConfigSpace.hyperparameters.UniformFloatHyperparameter("B", lower=0, upper=10)
        ])

        class ExactCostModel():
            def __init__(self, idx):
                self.idx = idx

            def __enter__(self):
                return self
            
            def __exit__(self, error_type, error_value, error_traceback):
                return error_type is None
            
            def evaluate(self, incumbent, budget):
                return (abs(8 - incumbent["B"]) if self.idx == 1 else abs(8 - (incumbent["B"] - incumbent["A"])))


        loss_matrix_computation = LossMatrixComputation()
        self.assertTrue( loss_matrix_computation.add_result(result1_path, cs, "result1", ExactCostModel(1)))
        self.assertTrue( loss_matrix_computation.add_result(result2_path, cs, "result2", ExactCostModel(2)))
        self.assertFalse(loss_matrix_computation.add_result(empty_path, cs, "empty", ExactCostModel(0)))
        self.assertEqual(loss_matrix_computation.results, [result1_path, result2_path])
        self.assertEqual(loss_matrix_computation.config_spaces, [cs, cs])
        self.assertEqual(loss_matrix_computation.budgets, [1.0, 3.0, 9.0])
        self.assertEqual(loss_matrix_computation.origins, ["result1", "result2"])

        open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_losses.txt"), "w").close()

        for i in range(2 * 2):
            loss_matrix_computation.write_loss(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_losses.txt"), i + 1)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_losses.txt"), "r") as f1:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_losses.txt"), "r") as f2:
                self.assertEqual(f1.readlines(), f2.readlines())

        os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_losses.txt"))

    def test_model_warmstarting(self):
        result1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result1")
        result2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result2")
        empty_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "empty_result")
        result1 = logged_results_to_HBS_result(result1_path)
        result2 = logged_results_to_HBS_result(result2_path)

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
        builder.add_result(empty_path, cs, "empty")
        builder.add_result(result1, cs, "result1")
        builder.add_result(result2, cs, "result2")
        r = builder.build()
        self.assertEqual(len(r.good_kdes), 2)
        self.assertEqual(len(r.bad_kdes), 2)
        self.assertEqual(len(r.kde_config_spaces), 2)