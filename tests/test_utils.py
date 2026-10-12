import pytest
import math
from torch import Tensor, rand, floor
from shared import utils

p_list =  [[0.2, 0.5, 0.8], [0.9, 0.7, 0.8]]
o_list = [1, 1]
class TestComputeBrierScore:
    
    @pytest.mark.parametrize(
            "P, O, var_gt, bias_gt",
            [
                (Tensor(p_list), Tensor(o_list), 1/30, 0.145),
                (Tensor([p_list]), Tensor([o_list]), 1/30, 0.145),
                (p_list, o_list, 1/30, 0.145),
                (0.5, 1.0, 0, 0.25),
                (Tensor([0.2, 0.5, 0.8]), Tensor([1, 1, 0]), 0, 0.51)
            ]
    )
    def test_score_computes_correctly(self, P, O, var_gt, bias_gt):
        # Act
        var, bias = utils.compute_brier_decomposition(P, O)

        # Assert
        assert math.isclose(var, var_gt, rel_tol=1e-5)
        assert math.isclose(bias, bias_gt, rel_tol=1e-5)

    @pytest.mark.parametrize("dim_o, dim_p", [(3, 1), (2, 3), (3, 1)])
    def test_large_dim_raise_error(self, dim_o, dim_p):
        # Arrange
        P = rand(*([10]*dim_p))
        O = floor(2 * rand(*([10]*dim_o)))

        # Assert
        with pytest.raises(ValueError) as info:
            utils.compute_brier_decomposition(P, O)

        assert f"provided: dim(P)={dim_p}, dim(O)={dim_o}" in str(info.value)

    @pytest.mark.parametrize("P, O", [("0.0", 1.0), (0.9, "1.0"), (0.9, 1), (1.1, 1.9)])
    def test_incorrect_type_raises_error(self, P, O):
        with pytest.raises(TypeError) as info:
            utils.compute_brier_decomposition(P, O)

        assert "type not supported" in str(info.value)

    @pytest.mark.parametrize("n, t", [(2, 3), (10, 2)])
    def test_brier_sum_matches(self, n, t):
        # Arrange 
        P = rand(n, t)
        O = floor(2 * rand(n))

        # Act 
        var, bias = utils.compute_brier_decomposition(P, O)
        brier = ((P - O[:, None])**2).mean()

        # Assert
        assert math.isclose(var + bias, brier, rel_tol=1e-5)         