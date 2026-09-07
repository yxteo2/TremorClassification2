import unittest
from types import SimpleNamespace
import numpy as np
from experiments.task_power_benchmark import features, align, fit

class TaskPowerTests(unittest.TestCase):
    def test_amplitude_and_rotation(self):
        t=np.arange(1000)/100
        x=np.array([2*np.sin(2*np.pi*6*t)+np.sin(2*np.pi*4*t),np.cos(2*np.pi*8*t),np.sin(2*np.pi*11*t)])
        a=features(x);b=features(2*x)
        np.testing.assert_allclose(b[:4]-a[:4],np.log(4),atol=1e-10)
        q,_=np.linalg.qr(np.random.default_rng(0).normal(size=(3,3)))
        np.testing.assert_allclose(features(q@x),a,atol=1e-10)
    def test_missing_task_rejected(self):
        r=SimpleNamespace(subject='PADS_1',y=2,x=np.ones((3,500)))
        with self.assertRaises(ValueError):align([r],[],['PADS:PADS_1'],[2])
    def test_test_inputs_and_labels_cannot_change_validation(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(60,12));y=np.arange(60)%3
        tr=np.arange(36);va=np.arange(36,48);te=np.arange(48,60)
        a,_=fit(x,y,tr,va,te);x[te]*=100;y[te]=0
        b,_=fit(x,y,tr,va,te);np.testing.assert_array_equal(a,b)
    def test_invalid_signal(self):
        with self.assertRaises(ValueError):features(np.zeros((3,100)))

if __name__=='__main__':unittest.main()
