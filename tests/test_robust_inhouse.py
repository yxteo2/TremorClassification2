import unittest
import numpy as np
from experiments.robust_inhouse import inner_splits, quaternion_quality, fit_model

class RobustTests(unittest.TestCase):
    def test_external_patients_train_only(self):
        y=np.tile([0,1,2],12)
        for tr,va in inner_splits(y,6):
            self.assertFalse(set(tr)&set(va))
            self.assertTrue(set(range(36,42)).issubset(tr))
            self.assertTrue((va<36).all())
    def test_bad_quaternions(self):
        q=np.tile([0.,0.,0.,1.],(10,3))
        self.assertFalse(any(quaternion_quality(q).values()))
        q[0,:4]=0
        self.assertTrue(quaternion_quality(q)['near_zero_norm'])
        q[0,0]=np.nan
        self.assertTrue(quaternion_quality(q)['nonfinite'])
    def test_robust_statistics_training_only(self):
        rng=np.random.default_rng(0);x=rng.normal(size=(36,36));y=np.tile([0,1,2],12)
        m=fit_model(x,y,inner_splits(y,0),robust=True,fixed=True)
        np.testing.assert_allclose(m['scale'].center_,np.median(x,axis=0))
        before=m['scale'].center_.copy();m.predict(np.full((2,36),1e6))
        np.testing.assert_array_equal(m['scale'].center_,before)

if __name__=='__main__':unittest.main()
