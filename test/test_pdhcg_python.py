import unittest
import numpy as np
import scipy.sparse as sp
import sys
import os

# Attempt to import pdhcg.
try:
    import pdhcg
except ImportError:
    # Assuming the script is in PDHCGv2-C/test/ and python package is in PDHCGv2-C/python/
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))
    try:
        import pdhcg
    except ImportError:
        print("Error: Could not import pdhcg module.")
        sys.exit(1)

from pdhcg import Model

class TestPDHCGInterface(unittest.TestCase):
    
    def setUp(self):
        # Suppress output for cleaner test logs
        self.default_params = {
            "OutputFlag": True, 
            "OptimalityTol": 1e-6, 
            "FeasibilityTol": 1e-6
        }



    def test_lowrank_qp(self):
        """
        Test Low-Rank QP: Minimize 0.5 * ||Rx||^2 + c^T x
        
        Let R = Identity (2x2). ||Rx||^2 = x1^2 + x2^2.
        Objective term: 0.5 * (x1^2 + x2^2)
        c = [0, 0]
        
        Subject to: x1 + x2 = 1, x >= 0
        
        Analytical Solution:
        Minimize 0.5(x1^2 + x2^2) s.t. x1+x2=1.
        By symmetry, x1 = x2 = 0.5.
        Obj = 0.5 * (0.25 + 0.25) = 0.25.
        """
        print("\n[Test] Low-Rank QP Minimization (R only)")
        
        # R is 2x2 Identity
        R = sp.eye(2, format="csc")
        c = np.array([0.0, 0.0])
        
        # Constraints: x1 + x2 = 1
        A = np.array([[1.0, 1.0]])
        l = np.array([1.0])
        u = np.array([1.0])
        lb = np.array([0.0, 0.0])

        # Note: We pass R to `objective_matrix_low_rank`, Q is None
        m = Model(objective_vector=c,
                  constraint_matrix=A,
                  constraint_lower_bound=l,
                  constraint_upper_bound=u,
                  objective_matrix=None,
                  objective_matrix_low_rank=R,  # <--- Input R here
                  variable_lower_bound=lb)
        
        m.setParams(**self.default_params)
        m.optimize()

        self.assertEqual(m.Status, "OPTIMAL")
        self.assertAlmostEqual(m.ObjVal, 0.25, places=4)
        if m.X is not None:
            self.assertAlmostEqual(m.X[0], 0.5, places=4)
            self.assertAlmostEqual(m.X[1], 0.5, places=4)

    def test_mixed_qp_sparse_and_lowrank(self):
        """
        Test Mixed QP: Minimize 0.5 * x^T Q x + 0.5 * ||Rx||^2 + c^T x
        
        Let Q = diag(2, 2). x^T Q x = 2*x1^2 + 2*x2^2.
        Let R = Identity(2). ||Rx||^2 = x1^2 + x2^2.
        
        Total Quadratic Term = 0.5*(2x^2) + 0.5*(1x^2) = 1.5 * (x1^2 + x2^2)
        
        Subject to: x1 + x2 = 1, x >= 0
        c = [0, 0]
        
        Analytical Solution:
        Min 1.5(x1^2 + x2^2) s.t. x1+x2=1.
        Symmetry -> x1 = x2 = 0.5.
        Obj = 1.5 * (0.25 + 0.25) = 1.5 * 0.5 = 0.75.
        """
        print("\n[Test] Mixed QP (Sparse Q + Low-Rank R)")
        
        Q = sp.diags([2.0, 2.0], format="csc")
        R = sp.eye(2, format="csc")
        c = np.array([0.0, 0.0])
        
        A = np.array([[1.0, 1.0]])
        l = np.array([1.0])
        u = np.array([1.0])
        lb = np.array([0.0, 0.0])

        m = Model(objective_vector=c,
                  constraint_matrix=A,
                  constraint_lower_bound=l,
                  constraint_upper_bound=u,
                  objective_matrix=Q,            # <--- Input Q
                  objective_matrix_low_rank=R,   # <--- Input R
                  variable_lower_bound=lb)
        
        m.setParams(**self.default_params)
        m.optimize()

        self.assertEqual(m.Status, "OPTIMAL")
        self.assertAlmostEqual(m.ObjVal, 0.75, places=4)
        if m.X is not None:
            self.assertAlmostEqual(m.X[0], 0.5, places=4)
            self.assertAlmostEqual(m.X[1], 0.5, places=4)

    def test_qp_no_constraints_A(self):
        """
        Test QP without matrix A (Box constrained or Unconstrained).
        
        Minimize 0.5 * x^T Q x + c^T x
        
        Let Q = diag(2, 2).
        Let c = [-2, -2].
        
        Objective: x1^2 + x2^2 - 2x1 - 2x2
                 = (x1 - 1)^2 - 1 + (x2 - 1)^2 - 1
                 = (x1 - 1)^2 + (x2 - 1)^2 - 2
        
        Global Minimum is at x = [1, 1], Obj = -2.
        
        We provide NO constraint matrix A.
        Bounds: lb = [-inf, -inf], ub = [inf, inf] (default in Model if None)
        """
        print("\n[Test] QP without Constraint Matrix A")
        
        Q = sp.diags([2.0, 2.0], format="csc")
        c = np.array([-2.0, -2.0])
        
        # Note: constraint_matrix is None.
        # constraint bounds must also be None.
        m = Model(objective_vector=c,
                  objective_matrix=Q,
                  constraint_matrix=None,       # <--- No A
                  constraint_lower_bound=None,
                  constraint_upper_bound=None,
                  variable_lower_bound=None,    # Unbounded variables
                  variable_upper_bound=None)
        
        m.setParams(**self.default_params)
        m.optimize()

        self.assertEqual(m.Status, "OPTIMAL")
        self.assertAlmostEqual(m.ObjVal, -2.0, places=4)
        if m.X is not None:
            self.assertAlmostEqual(m.X[0], 1.0, places=4)
            self.assertAlmostEqual(m.X[1], 1.0, places=4)
    # def test_simple_qp_minimization(self):
    #     """
    #     Existing test: Minimize 0.5 * x^T Q x + c^T x s.t. x1+x2=1, x>=0
    #     Q = diag(4, 2), c = [1, 1]
    #     """
    #     print("\n[Test] Simple QP Minimization (Sparse Q)")
    #     Q = sp.diags([4.0, 2.0], format="csc")
    #     c = np.array([1.0, 1.0])
    #     A = np.array([[1.0, 1.0]])
    #     l = np.array([1.0])
    #     u = np.array([1.0])
    #     lb = np.array([0.0, 0.0])
        
    #     m = Model(objective_matrix=Q, 
    #               objective_vector=c, 
    #               constraint_matrix=A,
    #               constraint_lower_bound=l, 
    #               constraint_upper_bound=u,
    #               variable_lower_bound=lb)
        
    #     m.setParams(**self.default_params)
    #     m.optimize()
        
    #     self.assertEqual(m.Status, "OPTIMAL")
    #     # Analytical result: 5/3 ~ 1.6667
    #     self.assertAlmostEqual(m.ObjVal, 5.0/3.0, places=4)
    #     if m.X is not None:
    #         self.assertAlmostEqual(m.X[0], 1.0/3.0, places=4)
    #         self.assertAlmostEqual(m.X[1], 2.0/3.0, places=4)
if __name__ == '__main__':
    unittest.main()