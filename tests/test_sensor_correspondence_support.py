import numpy as np
from clearvla.vision.sensor_geometry import photometric_support


def test_photometry_only_removes_wrong_or_unsupported_pairs():
    source=np.array([[[220,30,20],[20,20,220],[50,50,50]]],dtype=np.uint8)
    target=source.copy();target[0,1]=[220,30,20]
    law={'v':np.zeros(3,dtype=int),'u':np.arange(3),'accepted':np.array([True,True,False])}
    np.testing.assert_array_equal(photometric_support(law,source,target),[True,False,False])
    np.testing.assert_array_equal(law['accepted'],[True,True,False])
