import unittest
import numpy as np
import scipy.sparse as sp
import sys
import os

# Attempt to import pdhcg. If installed via pip, this works.
# If running from source without install, append the python directory to path.
try:
    import pdhcg
except ImportError:
    # Assuming the script is in PDHCGv2-C/test/ and python package is in PDHCGv2-C/python/
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))
    try:
        import pdhcg
    except ImportError:
        print("Error: Could not import pdhcg module. Please install it via 'pip install .' or ensure PYTHONPATH is set.")
        sys.exit(1)

from pdhcg import Model, PDHCG

class TestPDHCGInterface(unittest.TestCase):
    
    def setUp(self):
        # Suppress output for cleaner test logs by default, can be enabled for debugging
        self.default_params = {"OutputFlag": False, "OptimalityTol": 1e-6, "FeasibilityTol": 1e-6}

    def test_simple_qp_minimization(self):
        """
        Test a simple QP minimization problem.
        
        Minimize 0.5 * x^T Q x + c^T x
        
        Let Q = [[4, 0], [0, 2]]
        Let c = [1, 1]
        
        Objective: 0.5 * (4*x1^2 + 2*x2^2) + x1 + x2
                 = 2*x1^2 + x2^2 + x1 + x2
                 
        Subject to:
            x1 + x2 = 1
            x1, x2 >= 0
            
        Analytical Solution:
            Substitute x2 = 1 - x1 into Objective:
            f(x1) = 2*x1^2 + (1-x1)^2 + x1 + (1-x1)
                  = 2*x1^2 + (1 - 2*x1 + x1^2) + 1
                  = 3*x1^2 - 2*x1 + 2
            
            df/dx1 = 6*x1 - 2 = 0  =>  x1 = 1/3
            x2 = 1 - 1/3 = 2/3
            
            Constraints x >= 0 are satisfied.
            
            Optimal Obj = 3*(1/9) - 2*(1/3) + 2
                        = 1/3 - 2/3 + 2
                        = 5/3  (~1.6667)
        """
        print("\n[Test] Simple QP Minimization")
        
        # Q matrix (diagonal)
        Q = sp.diags([4.0, 2.0], format="csc")
        c = np.array([1.0, 1.0])
        
        # Constraint: x1 + x2 = 1
        A = np.array([[1.0, 1.0]])
        l = np.array([1.0])
        u = np.array([1.0])
        
        # Bounds: x >= 0
        lb = np.array([0.0, 0.0])
        
        # Pass Q to Model. Assuming argument name is 'Q' or 'quadratic_matrix'.
        # Based on bindings, 'Q' is the direct argument name.
        m = Model(objective_matrix=Q, 
                  objective_vector=c, 
                  constraint_matrix=A,
                  constraint_lower_bound=l, 
                  constraint_upper_bound=u,
                  variable_lower_bound=lb)
        
        m.setParams(**self.default_params)
        m.optimize()
        
        self.assertEqual(m.Status, "OPTIMAL")
        self.assertAlmostEqual(m.ObjVal, 5.0/3.0, places=4)
        
        if m.X is not None:
            self.assertAlmostEqual(m.X[0], 1.0/3.0, places=4)
            self.assertAlmostEqual(m.X[1], 2.0/3.0, places=4)

if __name__ == '__main__':
    unittest.main()
