import unittest
from types import SimpleNamespace
import numpy as np
import torch
from experiments.out_transfer import windows,load_bags,Encoder,pack,pooled

class TransferTests(unittest.TestCase):
    def test_windows(self):
        x=np.random.default_rng(0).normal(size=(9,1000))
        self.assertEqual(windows(x).shape,(2,9,160))
        with self.assertRaises(ValueError):windows(x[:,:100])
    def test_other_action_rejected(self):
        with self.assertRaises(ValueError):load_bags([SimpleNamespace(condition='REST')],'2015')
    def test_pool_equal_windows(self):
        torch.manual_seed(0);enc=Encoder();bag=np.ones((1,9,160),dtype='float32')
        x,o=pack([bag,np.repeat(bag,3,axis=0)],np.ones((1,9,1)))
        h=pooled(enc,x,o,2);torch.testing.assert_close(h[0],h[1])
    def test_independent_embedding(self):
        torch.manual_seed(0);enc=Encoder();a=np.ones((1,9,160),dtype='float32')
        x,o=pack([a,a],1);before=pooled(enc,x,o,2)[0].detach()
        x[-1]*=100;after=pooled(enc,x,o,2)[0].detach();torch.testing.assert_close(before,after)

if __name__=='__main__':unittest.main()
