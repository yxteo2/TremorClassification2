import unittest
from types import SimpleNamespace
import numpy as np
from experiments.inhouse_2015_explore import patient_features, Columns

class InhouseTests(unittest.TestCase):
    def record(self,subject,task,y=2):
        t=np.arange(500)/100
        return SimpleNamespace(subject=subject,condition=task,y=y,x=np.tile(np.sin(2*np.pi*6*t),(9,1)))
    def test_pool_trials_keep_task_order(self):
        r=[self.record('ET 1','OUT'),self.record('ET 1','REST')]
        a,y,p,d=patient_features(r,'2015')
        b,_,_,_=patient_features(r+r,'2015')
        self.assertEqual(a.shape,(1,36));self.assertEqual(p.tolist(),['2015:ET 1'])
        np.testing.assert_array_equal(a,b)
    def test_missing_task_audited(self):
        r=[self.record('ET 1','OUT'),self.record('ET 1','REST'),self.record('ET 2','OUT')]
        a,y,p,d=patient_features(r,'2015');self.assertEqual(d['excluded_missing_OUT_or_REST'],1)
    def test_conflicting_labels_rejected(self):
        with self.assertRaises(ValueError):patient_features([self.record('1','OUT',1),self.record('1','REST',2)],'2015')
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):patient_features([],'2015')
    def test_column_selection(self):
        x=np.arange(72).reshape(2,36)
        np.testing.assert_array_equal(Columns(tuple(range(6,12))).fit_transform(x),x[:,6:12])
if __name__=='__main__':unittest.main()
