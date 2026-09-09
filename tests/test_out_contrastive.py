import unittest
import numpy as np
import torch
from experiments.out_contrastive import paired_loss,hierarchy,embed
from experiments.out_transfer import Encoder

class ContrastiveTests(unittest.TestCase):
    def test_aligned_pairs_better_than_wrong(self):
        a=torch.eye(6)[None]
        self.assertLess(paired_loss(a,a).item(),paired_loss(a,a.roll(1,1)).item())
    def test_symmetry_and_finite_gradient(self):
        torch.manual_seed(0);a=torch.randn(4,16,40,requires_grad=True);b=torch.randn(4,16,40)
        loss=hierarchy(a,b);torch.testing.assert_close(loss,hierarchy(b,a))
        loss.backward();self.assertTrue(torch.isfinite(a.grad).all())
    def test_embedding_patient_independence(self):
        torch.manual_seed(0);enc=Encoder().eval();x=np.ones((2,9,160),dtype='float32')
        a=embed(enc,[x,x],1)[0];b=embed(enc,[x,100*x],1)[0]
        np.testing.assert_allclose(a,b);self.assertEqual(a.shape,(32,))

if __name__=='__main__':unittest.main()
