from pharmbio import data
import pytest
from ...help_utils import get_resource

class Test_regression():

    def test_load(self):
        (y,predictions,signs) = data.load_regression(get_resource('cpsign_reg_predictions.csv'),'y')
        assert len(y) == len(predictions)
        assert predictions.shape[2] ==len(signs)
        assert pytest.approx(1) == signs[0]
        
