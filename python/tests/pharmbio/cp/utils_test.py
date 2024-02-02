from pharmbio.cp import utils

import numpy as np
import pandas as pd
import pytest

def test_validate_sign_single():
    # Test single value
    # this should be fine
    utils.validate_sign(0)
    utils.validate_sign(.1)
    utils.validate_sign(1)
    
    
    # Too low value
    with pytest.raises(ValueError):
        utils.validate_sign(-.1)
    # Too high value
    with pytest.raises(ValueError):
        utils.validate_sign(1.001)

def test_validate_sign_numpy():
    # Should all be OK
    utils.validate_sign(np.asarray(0))
    utils.validate_sign(np.asarray(.1))
    utils.validate_sign(np.asarray(1))
    utils.validate_sign(np.asarray((0,.1,.2,.3,.5,.9,.99,1)))
    
    # Too low value
    with pytest.raises(ValueError):
        utils.validate_sign(np.asarray((-.1,.2)))
    # Too high value
    with pytest.raises(ValueError):
        utils.validate_sign(np.asarray((0.1,.5,1.001)))
    
    # Invalid shape
    with pytest.raises(ValueError):
        utils.validate_sign(np.asarray(1.001).reshape((1,1)))

def test_validate_sign_pd_series():
    # Should all be OK
    utils.validate_sign(pd.Series(0))
    utils.validate_sign(pd.Series(.1))
    utils.validate_sign(pd.Series(1))
    utils.validate_sign(pd.Series((0,.1,.2,.3,.5,.9,.99,1)))
    
    # Too low value
    with pytest.raises(ValueError):
        utils.validate_sign(pd.Series((-.1,.2)))
    # Too high value
    with pytest.raises(ValueError):
        utils.validate_sign(pd.Series((0.1,.5,1.001)))
    
    # Invalid shape
    with pytest.raises(TypeError):
        utils.validate_sign(pd.DataFrame(data=[[0.1]]))
