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
                (Tensor(p_list), Tensor(o_list), 0.0333, 0.145),
                (Tensor([p_list]), Tensor([o_list]), 0.0333, 0.145),
                (0.5, 1, 0, 0.25)
                (Tensor([0.2, 0.5, 0.8]), Tensor([1, 1, 0]), 0, 0.51)
            ]
    )
    def test_score_computes_correctly(self, P, O, var_gt, bias_gt):
        # Act
        var, bias = utils.compute_brier_decomposition(P, O)

        # Assert
        assert math.isclose(var, 0.0333)
        assert math.isclose(var, 0.145)

    @pytest.mark.parameterize("dim_o, dim_p")
    def large_dim_raise_error(self, dim_o, dim_p):
        # Arrange
        P = rand(*([10]*dim_p))
        O = floor(2 * rand(*([10]*dim_o)))

        # Assert
        with pytest.raises(ValueError) as info:
            utils.compute_brier_decomposition(P, O)

        assert "provided: dim(P)={dim_p}, dim(O)={dim_o}" in str(info.value)