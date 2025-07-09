import numpy as np
import particles_cdssm.sdes as sdes
import unittest
"""
This module contains unit tests for the linear SDEs implemented in particles_cdssm.sdes.

In particular, we test whether the calcuation of the transition density is correct: 

 
The linear SDE for step size t has a Gaussian transition density:

X_t | X_0 = x_0 ~ N(Ax_0 + b, V)

The values of A, b and V are given by the methods _a, _b and _v of a given linear SDE class.

You can run all the test by using the command:

python -m unittest test_linear_sdes.py
"""

class TestMvLinearSDE(unittest.TestCase):
    """
    Use this syntax to come up with the test cases:
    """
    test_cases = []
    true_values = []

    @classmethod
    def setUpClass(cls): # Here, we can provide class level attributes to the object.
        print("Setting up tests for test class: ", cls.__name__)

    def setUp(self):
        # Run before each individual test in the class
        self.n_tests = len(self.test_cases)
        self.test_sdes = [self.LinearSDEClass(**test_case) for test_case in self.test_cases]

    def run_method_tests(self, method_name):
        for i, sde, true_value_dict in zip(range(self.n_tests), self.test_sdes, self.true_values):
            method = getattr(sde, method_name)
            test_value = method(0., 1.) #delta_s = 1 for all tests
            true_value = true_value_dict[method_name]
            test_result = np.all(np.isclose(test_value, true_value))
            self.assertTrue(test_result, msg=self.fail_message(method_name, i, true_value, test_value))

    def fail_message(self, method_name, case_id, true_value, test_value):
        f"Test failed for method {method_name} case_id: {case_id} expected: {true_value}, got: {test_value}"
            
    def test_a(self):
        self.run_method_tests('_a')
        
    def test_b(self):   
        self.run_method_tests('_b')
        
    def test_v(self):
        self.run_method_tests('_v')
        
    def tearDown(self):
        # Run after each individual test in the class
        # Cleanup after each test
        self.test_sdes = None

    @classmethod
    def tearDownClass(cls):
        print(f"\nFinished running tests for test class: {cls.__name__}")

class TestMvBrownianMotion(TestMvLinearSDE):

    LinearSDEClass = sdes.MvBrownianMotion
    
    test_cases = [{'N': 1, 'dimX': 2, 'm': np.zeros((1, 2)), 's': np.eye(2)}, 
                  {'N': 1, 'dimX': 2, 'm': np.array([[0.5, 1.0]]), 's': 2.*np.eye(2)},
                  {'N': 1, 'dimX': 2, 'm': np.array([[0.5, 1.0]]), 's': np.array([[1., 0.], [0.9, np.sqrt(1.-0.9 ** 2)]])},
                  {'N': 1, 'dimX': 5, 'm': np.array([[0., 1., 2., 3., 4.]]), 's': 2.*np.eye(5)},
                  {'N': 100, 'dimX': 2, 'm': np.concatenate([2.*np.ones((50, 2)), np.ones((50, 2))]), 's': np.stack([np.eye(2)]*50 + [2.*np.eye(2)]*50)},
                  {'N': 100, 'dimX': 2, 'm': np.concatenate([2.*np.ones((50, 2)), np.ones((50, 2))]), 's': np.stack([np.array([[1., 0.], [0.9, np.sqrt(1.-0.9 ** 2)]])]*50 + [2.*np.eye(2)]*50)}
                  ] 

    true_values = [{'_a': np.eye(2).reshape(1, 2, 2), '_b': np.zeros((1, 2)), '_v': np.eye(2).reshape(1, 2, 2)},
                   {'_a': np.eye(2).reshape(1, 2, 2), '_b': np.array([[0.5, 1.0]]), '_v': 4.*np.eye(2).reshape(1, 2, 2)},
                   {'_a': np.eye(2).reshape(1, 2, 2), '_b': np.array([[0.5, 1.0]]), '_v': np.array([[1., 0.9], [0.9, 1.]])},
                   {'_a': np.eye(5).reshape(1, 5, 5), '_b': np.array([[0., 1., 2., 3., 4.]]), '_v': 4.* np.eye(5)},
                   {'_a': np.stack([np.eye(2)]*100), '_b': np.concatenate([2.*np.ones((50, 2)), np.ones((50, 2))]), '_v': np.stack([np.eye(2)]*50 + [4.*np.eye(2)]*50)},
                   {'_a': np.stack([np.eye(2)]*100), '_b': np.concatenate([2.*np.ones((50, 2)), np.ones((50, 2))]), '_v': np.stack([np.array([[1., 0.9], [0.9, 1.]])]*50 + [4.*np.eye(2)]*50)},
                   ]

class TestMvIndepBrownianMotion(TestMvLinearSDE):
    
    LinearSDEClass = sdes.MvIndepBrownianMotion
    
    test_cases = [{'N': 1, 'dimX': 2, 'm': np.zeros((1, 2)), 's': np.ones((1, 2))},
                  {'N': 1, 'dimX': 2, 'm': np.array([[0.5, 1.0]]), 's': 2.*np.ones((1, 2))},
                  {'N': 1, 'dimX': 5, 'm': np.array([[0., 1., 2., 3., 4.]]), 's': 2.*np.ones((1, 5))},
                  {'N': 100, 'dimX': 2, 'm': np.concatenate([2.*np.ones((50, 2)), np.ones((50, 2))]), 's': np.concatenate([np.ones((50, 2)), 2.*np.ones((50, 2))])}
                  ]

    true_values = [{'_a': np.eye(2).reshape(1, 2, 2), '_b': np.zeros((1, 2)), '_v': np.eye(2).reshape(1, 2, 2)},
                   {'_a': np.eye(2).reshape(1, 2, 2), '_b': np.array([[0.5, 1.0]]), '_v': 4.*np.eye(2).reshape(1, 2, 2)},
                   {'_a': np.eye(5).reshape(1, 5, 5), '_b': np.array([[0., 1., 2., 3., 4.]]), '_v': 4.* np.eye(5)},
                   {'_a': np.stack([np.eye(2)]*100), '_b': np.concatenate([2.*np.ones((50, 2)), np.ones((50, 2))]), '_v': np.stack([np.eye(2)]*50 + [4.*np.eye(2)]*50)}
                   ]
    
class TestIntegratedBrownianMotion(TestMvLinearSDE):
    
    LinearSDEClass = sdes.IntegratedBrownianMotion
    
    test_cases = [{'N': 1, 'dimX': 2, 'm': np.zeros((1, 1)), 's': np.ones((1, 1))},
                  {'N': 1, 'dimX': 4, 'm': np.ones((1, 2)), 's': np.eye(2)},
                  {'N': 1, 'dimX': 2, 'm': 0.5*np.ones((1, 1)), 's': np.ones((1, 1))},
                  {'N': 1, 'dimX': 4, 'm': np.array([[0.5, 1.]]), 's': np.array([[1., 0.], [0.9, np.sqrt(1.-0.9 ** 2)]])},
                  {'N': 100, 'dimX': 2, 'm': np.zeros((100, 1)), 's': np.stack([np.ones((1, 1))]*50 + [2.*np.ones((1, 1))]*50, axis=0)},
                  {'N': 100, 'dimX': 2, 'm': np.concatenate([1.*np.ones((50, 1)), 2.*np.ones((50, 1))]), 's': np.stack([np.ones((1, 1))]*50 + [2.*np.ones((1, 1))]*50, axis=0)},
                  {'N': 100, 'dimX': 4, 'm': np.concatenate([1.*np.ones((50, 2)), 2.*np.ones((50, 2))]), 's': np.stack([np.eye(2)]*50 + [2.*np.eye(2)]*50, axis=0)},
                  {'N': 100, 'dimX': 4, 'm': np.concatenate([1.*np.ones((50, 2)), 2.*np.ones((50, 2))]), 's': np.stack([np.eye(2)]*50 + [np.array([[1., 0.], [0.9, np.sqrt(1.-0.9 ** 2)]])]*50, axis=0)}
                  ]

    true_values = [{'_a': np.array([[[1., 1.], [0., 1.]]]), '_b': np.zeros((1, 2)), '_v': np.array([[[1/3, 0.5], [0.5, 1.]]])},
                   {'_a': np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 1.]]), 
                    '_b': np.array([[0.5, 0.5, 1., 1.]]),
                    '_v': np.array([[[1/3, 0., 0.5, 0.], [0., 1/3, 0., 0.5], [0.5, 0., 1., 0.], [0., 0.5, 0., 1.]]])},
                   {'_a': np.array([[[1., 1.], [0., 1.]]]), '_b': np.array([[0.25, 0.5]]), '_v': np.array([[[1/3, 0.5], [0.5, 1.]]])},
                   {'_a': np.array([[[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 1.]]]),
                    '_b': np.array([[0.25, 0.5, 0.5, 1.]]), 
                    '_v': np.array([[[1/3, 0.3, 0.5, 0.45], [0.3, 1/3, 0.45, 0.5], [0.5, 0.45, 1., 0.9], [0.45, 0.5, 0.9, 1.]]])},
                   {'_a': np.stack([np.array([[1., 1.], [0., 1.]])]*100), 
                    '_b': np.zeros((100, 2)), 
                    '_v': np.stack([np.array([[1/3, 0.5], [0.5, 1.]])]*50 + [4.*np.array([[1/3, 0.5], [0.5, 1.]])]*50)}, 
                   {'_a': np.stack([np.array([[1., 1.], [0., 1.]])]*100), 
                    '_b': np.concatenate([np.array([[0.5, 1.]])]*50 + [np.array([[1., 2.]])]*50), 
                    '_v': np.stack([np.array([[1/3, 0.5], [0.5, 1.]])]*50 + [4.*np.array([[1/3, 0.5], [0.5, 1.]])]*50)}, 
                   {'_a': np.stack([np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 1.]])]*100),
                    '_b': np.concatenate([np.array([[0.5, 0.5, 1., 1.]])]*50 + [np.array([[1., 1., 2., 2.]])]*50),
                    '_v': np.concatenate([np.array([[[1/3, 0., 0.5, 0.], [0., 1/3, 0., 0.5], [0.5, 0., 1., 0.], [0., 0.5, 0., 1.]]])] * 50 +
                           [4.* np.array([[[1/3, 0., 0.5, 0.], [0., 1/3, 0., 0.5], [0.5, 0., 1., 0.], [0., 0.5, 0., 1.]]])] * 50)},
                   {'_a': np.stack([np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 1.]])]*100),
                    '_b': np.concatenate([np.array([[0.5, 0.5, 1., 1.]])]*50 + [np.array([[1., 1., 2., 2.]])]*50),
                    '_v': np.concatenate([np.array([[[1/3, 0., 0.5, 0.], [0., 1/3, 0., 0.5], [0.5, 0., 1., 0.], [0., 0.5, 0., 1.]]])] * 50 +
                           [np.array([[[1/3, 0.3, 0.5, 0.45], [0.3, 1/3, 0.45, 0.5], [0.5, 0.45, 1., 0.9], [0.45, 0.5, 0.9, 1.]]])] * 50)}
                    ]
                   
class TestIntegratedIndepBrownianMotion(TestMvLinearSDE):

    LinearSDEClass = sdes.IntegratedIndepBrownianMotion
    
    test_cases = [{'N': 1, 'dimX': 2, 'm': np.zeros((1, 1)), 's': np.ones((1, 1))},
                  {'N': 1, 'dimX': 4, 'm': np.ones((1, 2)), 's': np.ones((1, 2))},
                  {'N': 1, 'dimX': 2, 'm': 0.5*np.ones((1, 1)), 's': np.ones((1, 1))},
                  {'N': 100, 'dimX': 2, 'm': np.zeros((100, 1)), 's': np.concatenate([np.ones((1, 1))]*50 + [2.*np.ones((1, 1))]*50, axis=0)},
                  {'N': 100, 'dimX': 2, 'm': np.concatenate([1.*np.ones((50, 1)), 2.*np.ones((50, 1))]), 's': np.concatenate([np.ones((1, 1))]*50 + [2.*np.ones((1, 1))]*50, axis=0)},
                  {'N': 100, 'dimX': 4, 'm': np.concatenate([1.*np.ones((50, 2)), 2.*np.ones((50, 2))]), 's': np.concatenate([np.ones((1, 2))]*50 + [2.*np.ones((1, 2))]*50, axis=0)},
                  ]

    true_values = [{'_a': np.array([[[1., 1.], [0., 1.]]]), '_b': np.zeros((1, 2)), '_v': np.array([[[1/3, 0.5], [0.5, 1.]]])},
                   {'_a': np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 1.]]), 
                    '_b': np.array([[0.5, 0.5, 1., 1.]]),
                    '_v': np.array([[[1/3, 0., 0.5, 0.], [0., 1/3, 0., 0.5], [0.5, 0., 1., 0.], [0., 0.5, 0., 1.]]])},
                   {'_a': np.array([[[1., 1.], [0., 1.]]]), '_b': np.array([[0.25, 0.5]]), '_v': np.array([[[1/3, 0.5], [0.5, 1.]]])},
                   {'_a': np.stack([np.array([[1., 1.], [0., 1.]])]*100), 
                    '_b': np.zeros((100, 2)), 
                    '_v': np.stack([np.array([[1/3, 0.5], [0.5, 1.]])]*50 + [4.*np.array([[1/3, 0.5], [0.5, 1.]])]*50)}, 
                   {'_a': np.stack([np.array([[1., 1.], [0., 1.]])]*100), 
                    '_b': np.concatenate([np.array([[0.5, 1.]])]*50 + [np.array([[1., 2.]])]*50), 
                    '_v': np.stack([np.array([[1/3, 0.5], [0.5, 1.]])]*50 + [4.*np.array([[1/3, 0.5], [0.5, 1.]])]*50)}, 
                   {'_a': np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
                    '_b': np.concatenate([np.array([[0.5, 0.5, 1., 1.]])]*50 + [np.array([[1., 1., 2., 2.]])]*50),
                    '_v': np.concatenate([np.array([[[1/3, 0., 0.5, 0.], [0., 1/3, 0., 0.5], [0.5, 0., 1., 0.], [0., 0.5, 0., 1.]]])] * 50 +
                           [4.* np.array([[[1/3, 0., 0.5, 0.], [0., 1/3, 0., 0.5], [0.5, 0., 1., 0.], [0., 0.5, 0., 1.]]])] * 50)},
                    ]    


class TestTwiceIntegratedBrownianMotion(TestMvLinearSDE):
    
    LinearSDEClass = sdes.TwiceIntegratedBrownianMotion
    
    test_cases = [{'N': 1, 'dimX': 3, 'm': np.zeros((1, 1)), 's': np.ones((1, 1))}, 
                  {'N': 1, 'dimX': 6, 'm': np.ones((1, 2)), 's': np.eye(2)},
                  {'N': 1, 'dimX': 3, 'm': 0.5*np.ones((1, 1)), 's': np.ones((1, 1))},
                  {'N': 1, 'dimX': 6, 'm': np.array([[0.5, 1.]]), 's': np.array([[1., 0.], [0.9, np.sqrt(1.-0.9 ** 2)]])},
                  {'N': 100, 'dimX': 3, 'm': np.zeros((100, 1)), 's': np.stack([np.ones((1, 1))]*50 + [2.*np.ones((1, 1))]*50, axis=0)},
                  {'N': 100, 'dimX': 3, 'm': np.concatenate([1.*np.ones((50, 1)), 2.*np.ones((50, 1))]), 's': np.stack([np.ones((1, 1))]*50 + [2.*np.ones((1, 1))]*50, axis=0)},
                  {'N': 100, 'dimX': 6, 'm': np.concatenate([1.*np.ones((50, 2)), 2.*np.ones((50, 2))]), 's': np.stack([np.eye(2)]*50 + [2.*np.eye(2)]*50, axis=0)},
                  {'N': 100, 'dimX': 6, 'm': np.concatenate([1.*np.ones((50, 2)), 2.*np.ones((50, 2))]), 's': np.stack([np.eye(2)]*50 + [np.array([[1., 0.], [0.9, np.sqrt(1.-0.9 ** 2)]])]*50, axis=0)}
                  ]

    true_values = [{'_a': np.array([[[1., 1., 0.5], [0., 1., 1.], [0., 0., 1.]]]), '_b': np.zeros((1, 3)), '_v': np.array([[[0.05, 0.125, 1/6], [0.125, 1/3, 0.5], [1/6, 0.5, 1.]]])},
                   {'_a': np.array([[[1., 0., 1., 0., 0.5, 0.], [0., 1., 0., 1., 0., 0.5], [0., 0., 1., 0., 1., 0.], [0., 0., 0., 1., 0., 1.], [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 1.]]]), 
                    '_b': np.array([[1/6, 1/6, 0.5, 0.5, 1., 1.]]),
                    '_v': np.array([[[0.05, 0., 0.125, 0., 1/6, 0.], [0., 0.05, 0., 0.125, 0., 1/6], [0.125, 0., 1/3, 0., 0.5, 0.], [0., 0.125, 0., 1/3, 0., 0.5], [1/6, 0., 0.5, 0., 1., 0.], [0., 1/6, 0., 0.5, 0., 1.]]])},
                   {'_a': np.array([[[1., 1., 0.5], [0., 1., 1.], [0., 0., 1.]]]), '_b': np.array([[1/12, 0.25, 0.5]]), '_v': np.array([[[0.05, 0.125, 1/6], [0.125, 1/3, 0.5], [1/6, 0.5, 1.]]])},
                   {'_a': np.array([[[1., 0., 1., 0., 0.5, 0.], 
                                     [0., 1., 0., 1., 0., 0.5],
                                     [0., 0., 1., 0., 1., 0.], 
                                     [0., 0., 0., 1., 0., 1.], 
                                     [0., 0., 0., 0., 1., 0.], 
                                     [0., 0., 0., 0., 0., 1.]]]),
                    '_b': np.array([[1/12, 1/6, 0.25, 0.5, 0.5, 1.]]), 
                    '_v': np.array([[[0.05, 0.045, 0.125, 0.1125, 1/6, 0.15],
                                     [0.045, 0.05, 0.1125, 0.125, 0.15, 1/6],
                                     [0.125, 0.1125, 1/3, 0.3, 0.5, 0.45],
                                     [0.1125, 0.125, 0.3, 1/3, 0.45, 0.5],
                                     [1/6, 0.15, 0.5, 0.45, 1., 0.9], 
                                     [0.15, 1/6, 0.45, 0.5, 0.9, 1.]]])},
                   {'_a': np.stack([np.array([[[1., 1., 0.5], [0., 1., 1.], [0., 0., 1.]]])]*100), 
                    '_b': np.zeros((100, 3)), 
                    '_v': np.concatenate([np.array([[[0.05, 0.125, 1/6], [0.125, 1/3, 0.5], [1/6, 0.5, 1.]]])]*50 + [4.*np.array([[[0.05, 0.125, 1/6], [0.125, 1/3, 0.5], [1/6, 0.5, 1.]]])]*50)}, 
                   {'_a': np.stack([np.array([[[1., 1., 0.5], [0., 1., 1.], [0., 0., 1.]]])]*100), 
                    '_b': np.concatenate([np.array([[1/6, 0.5, 1.]])]*50 + [np.array([[1/3, 1., 2.]])]*50), 
                    '_v': np.concatenate([np.array([[[0.05, 0.125, 1/6], [0.125, 1/3, 0.5], [1/6, 0.5, 1.]]])]*50 + [4.*np.array([[[0.05, 0.125, 1/6], [0.125, 1/3, 0.5], [1/6, 0.5, 1.]]])]*50)}, 
                   {'_a': np.concatenate([np.array([[[1., 0., 1., 0., 0.5, 0.], 
                                                    [0., 1., 0., 1., 0., 0.5],
                                                    [0., 0., 1., 0., 1., 0.], 
                                                    [0., 0., 0., 1., 0., 1.], 
                                                    [0., 0., 0., 0., 1., 0.], 
                                                    [0., 0., 0., 0., 0., 1.]]])]*100),
                    '_b': np.concatenate([np.array([[1/6, 1/6, 0.5, 0.5, 1., 1.]])]*50 + [np.array([[1/3, 1/3, 1., 1., 2., 2.]])]*50),
                    '_v': np.concatenate([np.array([[[0.05, 0., 0.125, 0., 1/6, 0.], [0., 0.05, 0., 0.125, 0., 1/6], [0.125, 0., 1/3, 0., 0.5, 0.], [0., 0.125, 0., 1/3, 0., 0.5], [1/6, 0., 0.5, 0., 1., 0.], [0., 1/6, 0., 0.5, 0., 1.]]])] * 50 +
                           [4.* np.array([[[0.05, 0., 0.125, 0., 1/6, 0.], [0., 0.05, 0., 0.125, 0., 1/6], [0.125, 0., 1/3, 0., 0.5, 0.], [0., 0.125, 0., 1/3, 0., 0.5], [1/6, 0., 0.5, 0., 1., 0.], [0., 1/6, 0., 0.5, 0., 1.]]])] * 50)},
                   {'_a': np.concatenate([np.array([[[1., 0., 1., 0., 0.5, 0.], 
                                                    [0., 1., 0., 1., 0., 0.5],
                                                    [0., 0., 1., 0., 1., 0.], 
                                                    [0., 0., 0., 1., 0., 1.], 
                                                    [0., 0., 0., 0., 1., 0.], 
                                                    [0., 0., 0., 0., 0., 1.]]])]*100),
                    '_b': np.concatenate([np.array([[1/6, 1/6, 0.5, 0.5, 1., 1.]])]*50 + [np.array([[1/3, 1/3, 1., 1., 2., 2.]])]*50),
                    '_v': np.concatenate([np.array([[[0.05, 0., 0.125, 0., 1/6, 0.], [0., 0.05, 0., 0.125, 0., 1/6], [0.125, 0., 1/3, 0., 0.5, 0.], [0., 0.125, 0., 1/3, 0., 0.5], [1/6, 0., 0.5, 0., 1., 0.], [0., 1/6, 0., 0.5, 0., 1.]]])] * 50 +
                           [np.array([[[0.05, 0.045, 0.125, 0.1125, 1/6, 0.15],
                                        [0.045, 0.05, 0.1125, 0.125, 0.15, 1/6],
                                        [0.125, 0.1125, 1/3, 0.3, 0.5, 0.45],
                                        [0.1125, 0.125, 0.3, 1/3, 0.45, 0.5],
                                        [1/6, 0.15, 0.5, 0.45, 1., 0.9], 
                                        [0.15, 1/6, 0.45, 0.5, 0.9, 1.]]])] * 50)}
                    ]

class TestTwiceIntegratedIndepBrownianMotion(TestMvLinearSDE):
    
    LinearSDEClass = sdes.TwiceIntegratedIndepBrownianMotion
    
    test_cases = [{'N': 1, 'dimX': 3, 'm': np.zeros((1, 1)), 's': np.ones((1, 1))}, 
                  {'N': 1, 'dimX': 6, 'm': np.ones((1, 2)), 's': np.ones((1, 2))},
                  {'N': 1, 'dimX': 3, 'm': 0.5*np.ones((1, 1)), 's': np.ones((1, 1))},
                  {'N': 100, 'dimX': 3, 'm': np.zeros((100, 1)), 's': np.concatenate([np.ones((1, 1))]*50 + [2.*np.ones((1, 1))]*50, axis=0)},
                  {'N': 100, 'dimX': 3, 'm': np.concatenate([1.*np.ones((50, 1)), 2.*np.ones((50, 1))]), 's': np.concatenate([np.ones((1, 1))]*50 + [2.*np.ones((1, 1))]*50, axis=0)},
                  {'N': 100, 'dimX': 6, 'm': np.concatenate([1.*np.ones((50, 2)), 2.*np.ones((50, 2))]), 's': np.concatenate([np.ones((1, 2))]*50 + [2.*np.ones((1, 2))]*50, axis=0)},
                  ]

    true_values = [{'_a': np.array([[[1., 1., 0.5], [0., 1., 1.], [0., 0., 1.]]]), '_b': np.zeros((1, 3)), '_v': np.array([[[0.05, 0.125, 1/6], [0.125, 1/3, 0.5], [1/6, 0.5, 1.]]])},
                   {'_a': np.array([[[1., 0., 1., 0., 0.5, 0.], [0., 1., 0., 1., 0., 0.5], [0., 0., 1., 0., 1., 0.], [0., 0., 0., 1., 0., 1.], [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 1.]]]), 
                    '_b': np.array([[1/6, 1/6, 0.5, 0.5, 1., 1.]]),
                    '_v': np.array([[[0.05, 0., 0.125, 0., 1/6, 0.], [0., 0.05, 0., 0.125, 0., 1/6], [0.125, 0., 1/3, 0., 0.5, 0.], [0., 0.125, 0., 1/3, 0., 0.5], [1/6, 0., 0.5, 0., 1., 0.], [0., 1/6, 0., 0.5, 0., 1.]]])},
                   {'_a': np.array([[[1., 1., 0.5], [0., 1., 1.], [0., 0., 1.]]]), '_b': np.array([[1/12, 0.25, 0.5]]), '_v': np.array([[[0.05, 0.125, 1/6], [0.125, 1/3, 0.5], [1/6, 0.5, 1.]]])},
                   {'_a': np.stack([np.array([[[1., 1., 0.5], [0., 1., 1.], [0., 0., 1.]]])]*100), 
                    '_b': np.zeros((100, 3)), 
                    '_v': np.concatenate([np.array([[[0.05, 0.125, 1/6], [0.125, 1/3, 0.5], [1/6, 0.5, 1.]]])]*50 + [4.*np.array([[[0.05, 0.125, 1/6], [0.125, 1/3, 0.5], [1/6, 0.5, 1.]]])]*50)}, 
                   {'_a': np.stack([np.array([[[1., 1., 0.5], [0., 1., 1.], [0., 0., 1.]]])]*100), 
                    '_b': np.concatenate([np.array([[1/6, 0.5, 1.]])]*50 + [np.array([[1/3, 1., 2.]])]*50), 
                    '_v': np.concatenate([np.array([[[0.05, 0.125, 1/6], [0.125, 1/3, 0.5], [1/6, 0.5, 1.]]])]*50 + [4.*np.array([[[0.05, 0.125, 1/6], [0.125, 1/3, 0.5], [1/6, 0.5, 1.]]])]*50)}, 
                   {'_a': np.concatenate([np.array([[[1., 0., 1., 0., 0.5, 0.], 
                                                    [0., 1., 0., 1., 0., 0.5],
                                                    [0., 0., 1., 0., 1., 0.], 
                                                    [0., 0., 0., 1., 0., 1.], 
                                                    [0., 0., 0., 0., 1., 0.], 
                                                    [0., 0., 0., 0., 0., 1.]]])]*100),
                    '_b': np.concatenate([np.array([[1/6, 1/6, 0.5, 0.5, 1., 1.]])]*50 + [np.array([[1/3, 1/3, 1., 1., 2., 2.]])]*50),
                    '_v': np.concatenate([np.array([[[0.05, 0., 0.125, 0., 1/6, 0.], [0., 0.05, 0., 0.125, 0., 1/6], [0.125, 0., 1/3, 0., 0.5, 0.], [0., 0.125, 0., 1/3, 0., 0.5], [1/6, 0., 0.5, 0., 1., 0.], [0., 1/6, 0., 0.5, 0., 1.]]])] * 50 +
                           [4.* np.array([[[0.05, 0., 0.125, 0., 1/6, 0.], [0., 0.05, 0., 0.125, 0., 1/6], [0.125, 0., 1/3, 0., 0.5, 0.], [0., 0.125, 0., 1/3, 0., 0.5], [1/6, 0., 0.5, 0., 1., 0.], [0., 1/6, 0., 0.5, 0., 1.]]])] * 50)},
                    ]
    
class TestMvOrnsteinUhlenbeck(TestMvLinearSDE):

    LinearSDEClass = sdes.MvOrnsteinUhlenbeck

    @staticmethod
    def f_b(rho, mu):
        # Convenience function for true values of _b
        return mu * (1 - np.exp(-rho))
    
    @staticmethod
    def f_v(rho, phi_sq):
        # Convenience function for true values of _v
        return ((phi_sq)/ (2*rho)) * (1 - np.exp(-2*rho))
    
    @staticmethod
    def g_v(rho_1, rho_2, phi_sq):
        # Convenience function for true values of _v
        return ((phi_sq)/ (rho_1 + rho_2)) * (1 - np.exp(-(rho_1+rho_2)))
    
    test_cases = [{'N': 1, 'dimX': 2, 'rho': np.ones((1, 2)), 'mu': np.zeros((1, 2)), 'phi': np.eye(2)}, # Done
                  {'N': 1, 'dimX': 2, 'rho': np.array([[0.5, 1.0]]), 'mu': np.array([[0.5, 1.0]]), 'phi': 2.*np.eye(2)}, # Done
                  {'N': 1, 'dimX': 2, 'rho': np.array([[0.5, 1.0]]), 'mu': np.array([[0.5, 1.0]]), 'phi': np.array([[1., 0.], [0.9, np.sqrt(1.-0.9 ** 2)]])}, # Done
                  {'N': 1, 'dimX': 5, 'rho': np.array([[0.5, 1.0, 1.5, 2.0, 2.5]]), 'mu': np.array([[0., 1., 2., 3., 4.]]), 'phi': 2.*np.eye(5)}, # Done
                  {'N': 100, 'dimX': 2, 'rho': np.stack([np.array([0.5, 1.0])]*100), 'mu': np.concatenate([np.zeros((50, 2))] + [np.array([[0.5, 1.0]])]*50), 'phi': np.stack([np.eye(2)]*50 + [2.*np.eye(2)]*50)}, # Done
                  {'N': 100, 'dimX': 2, 'rho': np.stack([np.array([0.5, 1.0])]*50 + [np.array([1.0, 2.0])]*50), 'mu': np.concatenate([np.zeros((50, 2))] + [np.array([[0.5, 1.0]])]*50), 'phi': np.stack([2.*np.eye(2)]*50 + [np.array([[1., 0.], [0.9, np.sqrt(1.-0.9 ** 2)]])]*50)} # Done
                  ]

    true_values = [{'_a': np.diag([np.exp(-1.), np.exp(-1.)]).reshape(1, 2, 2), '_b': np.zeros((1, 2)), '_v': np.diag([f_v(1., 1.), f_v(1., 1.)]).reshape(1, 2, 2)},
                   {'_a': np.diag([np.exp(-0.5), np.exp(-1.)]).reshape(1, 2, 2), '_b': np.array([[f_b(0.5, 0.5), f_b(1.0, 1.0)]]), '_v': np.diag([f_v(0.5, 4.), f_v(1., 4.)]).reshape(1, 2, 2)},
                   {'_a': np.diag([np.exp(-0.5), np.exp(-1.)]).reshape(1, 2, 2), '_b': np.array([[f_b(0.5, 0.5), f_b(1.0, 1.0)]]), '_v': np.array([[f_v(0.5, 1.), g_v(0.5, 1., 0.9)], [g_v(0.5, 1., 0.9), f_v(1., 1.)]])},
                   {'_a': np.diag([np.exp(-0.5), np.exp(-1.), np.exp(-1.5), np.exp(-2.0), np.exp(-2.5)]).reshape(1, 5, 5), 
                    '_b': np.array([[f_b(0.5, 0.), f_b(1.0, 1.), f_b(1.5, 2.), f_b(2., 3.), f_b(2.5, 4.)]]),
                    '_v': np.diag([f_v(0.5, 4.), f_v(1., 4.), f_v(1.5, 4.), f_v(2., 4.), f_v(2.5, 4.)]).reshape(1, 5, 5)},
                   {'_a': np.stack([np.diag([np.exp(-0.5), np.exp(-1.)])]*100), 
                    '_b': np.stack([np.array([f_b(0.5, 0.), f_b(1., 0.)])]*50 + [np.array([f_b(0.5, 0.5), f_b(1., 1.)])]*50), 
                    '_v': np.stack([np.diag([f_v(0.5, 1.), f_v(1.0, 1.)])] * 50 + [np.diag([f_v(0.5, 4.), f_v(1.0, 4.)])] * 50)},
                   {'_a': np.stack([np.diag([np.exp(-0.5), np.exp(-1.)])]*50 + [np.diag([np.exp(-1.0), np.exp(-2.0)])]*50), 
                    '_b': np.stack([np.array([f_b(0.5, 0.), f_b(1., 0.)])]*50 + [np.array([f_b(1.0, 0.5), f_b(2.0, 1.)])]*50),
                    '_v': np.stack([np.diag([f_v(0.5, 4.), f_v(1.0, 4.)])] * 50 + [np.array([[f_v(1.0, 1.), g_v(1.0, 2.0, 0.9)], [g_v(1.0, 2.0, 0.9), f_v(2.0, 1.)]])] * 50)}, 
                   ]
    
class TestMvIndepOrnsteinUhlenbeck(TestMvLinearSDE):

    LinearSDEClass = sdes.MvIndepOrnsteinUhlenbeck
    
    @staticmethod
    def f_b(rho, mu):
        # Convenience function for true values of _b
        return mu * (1 - np.exp(-rho))
    
    @staticmethod
    def f_v(rho, phi_sq):
        # Convenience function for true values of _v
        return ((phi_sq)/ (2*rho)) * (1 - np.exp(-2*rho))
    
    test_cases = [{'N': 1, 'dimX': 2, 'rho': np.ones((1, 2)), 'mu': np.zeros((1, 2)), 'phi': np.ones((1, 2))},
                  {'N': 1, 'dimX': 2, 'rho': np.array([[0.5, 1.0]]), 'mu': np.array([[0.5, 1.0]]), 'phi': 2.*np.ones((1, 2))},
                  {'N': 1, 'dimX': 5, 'rho': np.array([[0.5, 1.0, 1.5, 2.0, 2.5]]), 'mu': np.array([[0., 1., 2., 3., 4.]]), 'phi': 2.*np.ones((1, 5))}, 
                  {'N': 100, 'dimX': 2, 'rho': np.stack([np.array([0.5, 1.0])]*100), 'mu': np.concatenate([np.zeros((50, 2))] + [np.array([[0.5, 1.0]])]*50), 'phi': np.concatenate([np.ones((50, 2))] + [2.*np.ones((50, 2))])},
                  ]

    true_values = [{'_a': np.diag([np.exp(-1.), np.exp(-1.)]).reshape(1, 2, 2), '_b': np.zeros((1, 2)), '_v': np.diag([f_v(1., 1.), f_v(1., 1.)]).reshape(1, 2, 2)},
                   {'_a': np.diag([np.exp(-0.5), np.exp(-1.)]).reshape(1, 2, 2), '_b': np.array([[f_b(0.5, 0.5), f_b(1.0, 1.0)]]), '_v': np.diag([f_v(0.5, 4.), f_v(1., 4.)]).reshape(1, 2, 2)},
                   {'_a': np.diag([np.exp(-0.5), np.exp(-1.), np.exp(-1.5), np.exp(-2.0), np.exp(-2.5)]).reshape(1, 5, 5), 
                    '_b': np.array([[f_b(0.5, 0.), f_b(1.0, 1.), f_b(1.5, 2.), f_b(2., 3.), f_b(2.5, 4.)]]),
                    '_v': np.diag([f_v(0.5, 4.), f_v(1., 4.), f_v(1.5, 4.), f_v(2., 4.), f_v(2.5, 4.)]).reshape(1, 5, 5)},
                   {'_a': np.stack([np.diag([np.exp(-0.5), np.exp(-1.)])]*100), 
                    '_b': np.stack([np.array([f_b(0.5, 0.), f_b(1., 0.)])]*50 + [np.array([f_b(0.5, 0.5), f_b(1., 1.)])]*50), 
                    '_v': np.stack([np.diag([f_v(0.5, 1.), f_v(1.0, 1.)])] * 50 + [np.diag([f_v(0.5, 4.), f_v(1.0, 4.)])] * 50)},
                   ]

class TestIntegratedOrnsteinUhlenbeck(TestMvLinearSDE):

    LinearSDEClass = sdes.IntegratedOrnsteinUhlenbeck

    @staticmethod
    def f_a(rho):
        # Convenience function for true values of _a
        return np.exp(-rho)

    @staticmethod
    def f_a_1(rho):
        # Convenience function for true values of _a
        return (1/rho) * (1 - np.exp(-rho))
    
    @staticmethod
    def f_b(rho, mu):
        # Convenience function for true values of _b
        return mu * (1 - np.exp(-rho))

    @staticmethod
    def f_b_1(rho, mu):
        # Convenience function for true values of _b
        return (mu /rho) * (np.exp(-rho) + rho - 1) # t=1
    
    @staticmethod
    def f_v(rho, phi_sq):
        # Convenience function for true values of _v
        return ((phi_sq)/ (2*rho)) * (1 - np.exp(-2*rho))
    
    @staticmethod
    def g_v(rho_1, rho_2, phi_sq):
        # Convenience function for true values of _v
        return ((phi_sq)/ (rho_1 + rho_2)) * (1 - np.exp(-(rho_1+rho_2)))

    @staticmethod
    def f_v_01(rho, phi_sq):
        # Convenience function for true values of _v
        a = ((phi_sq)/ (rho*rho)) * (1 - np.exp(-1.*rho))
        b = ((phi_sq)/ (2.*rho*rho)) * (1 - np.exp(-(2.*rho)))
        return a - b
    
    @staticmethod
    def g_v_01(rho_1, rho_2, phi_sq):
        # Convenience function for true values of _v
        a = ((phi_sq)/ (rho_1 * rho_2)) * (1 - np.exp(-1.*rho_1))
        b = ((phi_sq)/ (rho_2*(rho_1 + rho_2))) * (1 - np.exp(-(rho_1+rho_2)))
        return a - b

    @staticmethod
    def f_v_11(rho, phi_sq):
        # Convenience function for true values of _v
        a = ((phi_sq)/ (rho * rho))
        b_1 = 1.
        b_2 = (1. - np.exp(-1.*rho))/rho
        b_4 = (1. - np.exp(-2.*rho))/(2.*rho)
        return a *(b_1 - 2.*b_2 + b_4)

    @staticmethod
    def g_v_11(rho_1, rho_2, phi_sq):
        # Convenience function for true values of _v
        a = ((phi_sq)/ (rho_1 * rho_2))
        b_1 = 1.
        b_2 = (1. - np.exp(-1.*rho_1))/rho_1
        b_3 = (1. - np.exp(-1.*rho_2))/rho_2
        b_4 = (1. - np.exp(-(rho_1+rho_2)))/(rho_1 + rho_2)
        return a *(b_1 - b_2 - b_3 + b_4)

    test_cases = [{'N': 1, 'dimX': 2, 'rho': np.ones((1, 1)), 'mu': np.zeros((1, 1)), 'phi': np.ones((1, 1))}, # Done
                  {'N': 1, 'dimX': 2, 'rho': np.ones((1, 1)), 'mu': np.ones((1, 1)), 'phi': np.ones((1, 1))}, # Done
                  {'N': 1, 'dimX': 2, 'rho': 2.*np.ones((1, 1)), 'mu': 2.*np.ones((1, 1)), 'phi': 2.*np.ones((1, 1))}, # Done
                  {'N': 1, 'dimX': 4, 'rho': np.ones((1, 2)), 'mu': np.zeros((1, 2)), 'phi': np.eye(2)}, # Done
                  {'N': 1, 'dimX': 4, 'rho': np.ones((1, 2)), 'mu': np.ones((1, 2)), 'phi': np.eye(2)}, # Done
                  {'N': 1, 'dimX': 4, 'rho': 2.*np.ones((1, 2)), 'mu': 2.*np.ones((1, 2)), 'phi': 2.*np.eye(2)}, # Done
                #   {'N': 1, 'dimX': 4, 'rho': np.array([[1., 2.]]), 'mu': np.array([[1., 2.]]), 'phi': np.array([[1., 0.], [0.9, np.sqrt(1.-0.9 ** 2)]])}, # Done
                  {'N': 100, 'dimX': 2, 'rho': np.concatenate([1.*np.ones((50, 1)), 2.*np.ones((50, 1))]), 'mu': np.concatenate([1.*np.ones((50, 1)), 2.*np.ones((50, 1))]),'phi': np.stack([np.ones((1, 1))]*50 + [2.*np.ones((1, 1))]*50, axis=0)}, # Done
                  {'N': 100, 'dimX': 4, 'rho': np.concatenate([1.*np.ones((50, 2)), 2.*np.ones((50, 2))]), 'mu': np.concatenate([1.*np.ones((50, 2)), 2.*np.ones((50, 2))]), 'phi': np.stack([np.eye(2)]*50 + [2.*np.eye(2)]*50, axis=0)}, # Done
                #   {'N': 100, 'dimX': 4, 'rho': np.concatenate([1.*np.ones((50, 2))] + [np.array([[1., 2.]])]*50), 'mu': np.concatenate([1.*np.ones((50, 2))] + [np.array([[1., 2.]])]*50),  'phi': np.stack([np.eye(2)]*50 + [np.array([[1., 0.], [0.9, np.sqrt(1.-0.9 ** 2)]])]*50, axis=0)} # Done
                  ]

    true_values = [{'_a': np.array([[[1., f_a_1(1.)], [0., f_a(1.)]]]), '_b': np.zeros((1, 2)), '_v': np.array([[[f_v_11(1., 1.), f_v_01(1., 1.)], [f_v_01(1., 1.), f_v(1., 1.)]]])},
                   {'_a': np.array([[[1., f_a_1(1.)], [0., f_a(1.)]]]), '_b': np.array([[f_b_1(1., 1.), f_b(1., 1.)]]), '_v': np.array([[[f_v_11(1., 1.), f_v_01(1., 1.)], [f_v_01(1., 1.), f_v(1., 1.)]]])},
                   {'_a': np.array([[[1., f_a_1(2.)], [0., f_a(2.)]]]), '_b': np.array([[f_b_1(2., 2.), f_b(2., 2.)]]), '_v': np.array([[[f_v_11(2., 4.), f_v_01(2., 4.)], [f_v_01(2., 4.), f_v(2., 4.)]]])},
                   {'_a': np.array([[[1., 0., f_a_1(1.), 0.], [0., 1., 0., f_a_1(1.)], [0., 0., f_a(1.), 0.], [0., 0., 0., f_a(1.)]]]),
                    '_b': np.zeros((1, 4)), 
                    '_v': np.array([[[f_v_11(1., 1.), 0., f_v_01(1., 1.,), 0.], [0., f_v_11(1., 1.), 0., f_v_01(1., 1.)], [f_v_01(1., 1.), 0., f_v(1., 1.), 0.], [0., f_v_01(1., 1.), 0., f_v(1., 1.)]]])},
                   {'_a': np.array([[[1., 0., f_a_1(1.), 0.], [0., 1., 0., f_a_1(1.)], [0., 0., f_a(1.), 0.], [0., 0., 0., f_a(1.)]]]),
                    '_b': np.array([[f_b_1(1., 1.), f_b_1(1., 1.), f_b(1., 1.), f_b(1., 1.)]]), 
                    '_v': np.array([[[f_v_11(1., 1.), 0., f_v_01(1., 1.,), 0.], [0., f_v_11(1., 1.), 0., f_v_01(1., 1.)], [f_v_01(1., 1.), 0., f_v(1., 1.), 0.], [0., f_v_01(1., 1.), 0., f_v(1., 1.)]]])},
                   {'_a': np.array([[[1., 0., f_a_1(2.), 0.], [0., 1., 0., f_a_1(2.)], [0., 0., f_a(2.), 0.], [0., 0., 0., f_a(2.)]]]),
                    '_b': np.array([[f_b_1(2., 2.), f_b_1(2., 2.), f_b(2., 2.), f_b(2., 2.)]]), 
                    '_v': np.array([[[f_v_11(2., 4.), 0., f_v_01(2., 4.,), 0.], [0., f_v_11(2., 4.), 0., f_v_01(2., 4.)], [f_v_01(2., 4.), 0., f_v(2., 4.), 0.], [0., f_v_01(2., 4.), 0., f_v(2., 4.)]]])},
                #    {'_a': np.array([[[1., 0., f_a_1(1.), 0.], [0., 1., 0., f_a_1(2.)], [0., 0., f_a(1.), 0.], [0., 0., 0., f_a(2.)]]]),
                #     '_b': np.array([[f_b_1(1., 1.), f_b_1(2., 2.), f_b(1., 1.), f_b(2., 2.)]]), 
                #     '_v': np.array([[[f_v_11(1., 1.), g_v_11(1., 2., 0.9), f_v_01(1., 1.), g_v_01(1., 2., 0.9)], 
                #                      [g_v_11(2., 1., 0.9), f_v_11(2., 1.), g_v_01(2., 1., 0.9), f_v_01(2., 1.)], 
                #                      [f_v_01(1., 1.), g_v_01(2., 1., 0.9), f_v(1., 1.), g_v(1., 2., 0.9)],
                #                      [g_v_01(1., 2., 0.9), f_v_01(2., 1.), g_v(1., 2., 0.9), f_v(2., 1.)]]])},
                   {'_a': np.concatenate([np.array([[[1., f_a_1(1.)], [0., f_a(1.)]]])]*50 + 
                                         [np.array([[[1., f_a_1(2.)], [0., f_a(2.)]]])]*50), 
                    '_b': np.concatenate([np.array([[f_b_1(1., 1.), f_b(1., 1.)]])]*50 + 
                                         [np.array([[f_b_1(2., 2.), f_b(2., 2.)]])]*50), 
                    '_v': np.concatenate([np.array([[[f_v_11(1., 1.), f_v_01(1., 1.)], [f_v_01(1., 1.), f_v(1., 1.)]]])]*50 + 
                                         [np.array([[[f_v_11(2., 4.), f_v_01(2., 4.)], [f_v_01(2., 4.), f_v(2., 4.)]]])]*50)}, 
                   {'_a': np.concatenate([np.array([[[1., 0., f_a_1(1.), 0.], [0., 1., 0., f_a_1(1.)], [0., 0., f_a(1.), 0.], [0., 0., 0., f_a(1.)]]])]*50 + 
                                         [np.array([[[1., 0., f_a_1(2.), 0.], [0., 1., 0., f_a_1(2.)], [0., 0., f_a(2.), 0.], [0., 0., 0., f_a(2.)]]])]*50), 
                    '_b': np.concatenate([np.array([[f_b_1(1., 1.), f_b_1(1., 1.), f_b(1., 1.), f_b(1., 1.)]])]*50 + 
                                         [np.array([[f_b_1(2., 2.), f_b_1(2., 2.), f_b(2., 2.), f_b(2., 2.)]])]*50), 
                    '_v': np.concatenate([np.array([[[f_v_11(1., 1.), 0., f_v_01(1., 1.,), 0.], [0., f_v_11(1., 1.), 0., f_v_01(1., 1.)], [f_v_01(1., 1.), 0., f_v(1., 1.), 0.], [0., f_v_01(1., 1.), 0., f_v(1., 1.)]]])]*50 + 
                                         [np.array([[[f_v_11(2., 4.), 0., f_v_01(2., 4.,), 0.], [0., f_v_11(2., 4.), 0., f_v_01(2., 4.)], [f_v_01(2., 4.), 0., f_v(2., 4.), 0.], [0., f_v_01(2., 4.), 0., f_v(2., 4.)]]])]*50)}, 
                #    {'_a': np.concatenate([np.array([[[1., 0., f_a_1(1.), 0.], [0., 1., 0., f_a_1(1.)], [0., 0., f_a(1.), 0.], [0., 0., 0., f_a(1.)]]])]*50 + 
                #                          [np.array([[[1., 0., f_a_1(1.), 0.], [0., 1., 0., f_a_1(2.)], [0., 0., f_a(1.), 0.], [0., 0., 0., f_a(2.)]]])]*50), 
                #     '_b': np.concatenate([np.array([[f_b_1(1., 1.), f_b_1(1., 1.), f_b(1., 1.), f_b(1., 1.)]])]*50 + 
                #                          [np.array([[f_b_1(1., 1.), f_b_1(2., 2.), f_b(1., 1.), f_b(2., 2.)]])]*50), 
                #     '_v': np.concatenate([np.array([[[f_v_11(1., 1.), 0., f_v_01(1., 1.,), 0.], [0., f_v_11(1., 1.), 0., f_v_01(1., 1.)], [f_v_01(1., 1.), 0., f_v(1., 1.), 0.], [0., f_v_01(1., 1.), 0., f_v(1., 1.)]]])]*50 + 
                #                          [np.array([[[f_v_11(1., 1.), g_v_11(1., 2., 0.9), f_v_01(1., 1.), g_v_01(1., 2., 0.9)], 
                #                                     [g_v_11(2., 1., 0.9), f_v_11(2., 1.), g_v_01(2., 1., 0.9), f_v_01(1., 1.)], 
                #                                     [f_v_01(1., 1.), g_v_01(2., 1., 0.9), f_v(1., 1.), g_v(1., 2., 0.9)],
                #                                     [g_v_01(1., 2., 0.9), f_v_01(2., 2.), g_v(1., 2., 0.9), f_v(2., 2.)]]])]*50)} 
                   ]

class TestIntegratedIndepOrnsteinUhlenbeck(TestMvLinearSDE):

    LinearSDEClass = sdes.IntegratedIndepOrnsteinUhlenbeck

    @staticmethod
    def f_a(rho):
        # Convenience function for true values of _a
        return np.exp(-rho)

    @staticmethod
    def f_a_1(rho):
        # Convenience function for true values of _a
        return (1/rho) * (1 - np.exp(-rho))
    
    @staticmethod
    def f_b(rho, mu):
        # Convenience function for true values of _b
        return mu * (1 - np.exp(-rho))

    @staticmethod
    def f_b_1(rho, mu):
        # Convenience function for true values of _b
        return (mu /rho) * (np.exp(-rho) + rho - 1) # t=1
    
    @staticmethod
    def f_v(rho, phi_sq):
        # Convenience function for true values of _v
        return ((phi_sq)/ (2*rho)) * (1 - np.exp(-2*rho))
    
    @staticmethod
    def f_v_01(rho, phi_sq):
        # Convenience function for true values of _v
        a = ((phi_sq)/ (rho*rho)) * (1 - np.exp(-1.*rho))
        b = ((phi_sq)/ (2.*rho*rho)) * (1 - np.exp(-(2.*rho)))
        return a - b
    
    @staticmethod
    def f_v_11(rho, phi_sq):
        # Convenience function for true values of _v
        a = ((phi_sq)/ (rho * rho))
        b_1 = 1.
        b_2 = (1. - np.exp(-1.*rho))/rho
        b_4 = (1. - np.exp(-2.*rho))/(2.*rho)
        return a *(b_1 - 2.*b_2 + b_4)

    test_cases = [{'N': 1, 'dimX': 2, 'rho': np.ones((1, 1)), 'mu': np.zeros((1, 1)), 'phi': np.ones((1, 1))}, # Done
                  {'N': 1, 'dimX': 2, 'rho': np.ones((1, 1)), 'mu': np.ones((1, 1)), 'phi': np.ones((1, 1))}, # Done
                  {'N': 1, 'dimX': 2, 'rho': 2.*np.ones((1, 1)), 'mu': 2.*np.ones((1, 1)), 'phi': 2.*np.ones((1, 1))}, # Done
                  {'N': 1, 'dimX': 4, 'rho': np.ones((1, 2)), 'mu': np.zeros((1, 2)), 'phi': np.ones((1, 2))}, # Done
                  {'N': 1, 'dimX': 4, 'rho': np.ones((1, 2)), 'mu': np.ones((1, 2)), 'phi': np.ones((1, 2))}, # Done
                  {'N': 1, 'dimX': 4, 'rho': 2.*np.ones((1, 2)), 'mu': 2.*np.ones((1, 2)), 'phi': 2.*np.ones((1, 2))}, # Done
                  {'N': 100, 'dimX': 2, 'rho': np.concatenate([1.*np.ones((50, 1)), 2.*np.ones((50, 1))]), 'mu': np.concatenate([1.*np.ones((50, 1)), 2.*np.ones((50, 1))]),'phi': np.concatenate([np.ones((1, 1))]*50 + [2.*np.ones((1, 1))]*50, axis=0)}, # Done
                  {'N': 100, 'dimX': 4, 'rho': np.concatenate([1.*np.ones((50, 2)), 2.*np.ones((50, 2))]), 'mu': np.concatenate([1.*np.ones((50, 2)), 2.*np.ones((50, 2))]), 'phi': np.concatenate([np.ones((1, 2))]*50 + [2.*np.ones((1, 2))]*50, axis=0)}, # Done
                  ]

    true_values = [{'_a': np.array([[[1., f_a_1(1.)], [0., f_a(1.)]]]), '_b': np.zeros((1, 2)), '_v': np.array([[[f_v_11(1., 1.), f_v_01(1., 1.)], [f_v_01(1., 1.), f_v(1., 1.)]]])},
                   {'_a': np.array([[[1., f_a_1(1.)], [0., f_a(1.)]]]), '_b': np.array([[f_b_1(1., 1.), f_b(1., 1.)]]), '_v': np.array([[[f_v_11(1., 1.), f_v_01(1., 1.)], [f_v_01(1., 1.), f_v(1., 1.)]]])},
                   {'_a': np.array([[[1., f_a_1(2.)], [0., f_a(2.)]]]), '_b': np.array([[f_b_1(2., 2.), f_b(2., 2.)]]), '_v': np.array([[[f_v_11(2., 4.), f_v_01(2., 4.)], [f_v_01(2., 4.), f_v(2., 4.)]]])},
                   {'_a': np.array([[[1., 0., f_a_1(1.), 0.], [0., 1., 0., f_a_1(1.)], [0., 0., f_a(1.), 0.], [0., 0., 0., f_a(1.)]]]),
                    '_b': np.zeros((1, 4)), 
                    '_v': np.array([[[f_v_11(1., 1.), 0., f_v_01(1., 1.,), 0.], [0., f_v_11(1., 1.), 0., f_v_01(1., 1.)], [f_v_01(1., 1.), 0., f_v(1., 1.), 0.], [0., f_v_01(1., 1.), 0., f_v(1., 1.)]]])},
                   {'_a': np.array([[[1., 0., f_a_1(1.), 0.], [0., 1., 0., f_a_1(1.)], [0., 0., f_a(1.), 0.], [0., 0., 0., f_a(1.)]]]),
                    '_b': np.array([[f_b_1(1., 1.), f_b_1(1., 1.), f_b(1., 1.), f_b(1., 1.)]]), 
                    '_v': np.array([[[f_v_11(1., 1.), 0., f_v_01(1., 1.,), 0.], [0., f_v_11(1., 1.), 0., f_v_01(1., 1.)], [f_v_01(1., 1.), 0., f_v(1., 1.), 0.], [0., f_v_01(1., 1.), 0., f_v(1., 1.)]]])},
                   {'_a': np.array([[[1., 0., f_a_1(2.), 0.], [0., 1., 0., f_a_1(2.)], [0., 0., f_a(2.), 0.], [0., 0., 0., f_a(2.)]]]),
                    '_b': np.array([[f_b_1(2., 2.), f_b_1(2., 2.), f_b(2., 2.), f_b(2., 2.)]]), 
                    '_v': np.array([[[f_v_11(2., 4.), 0., f_v_01(2., 4.,), 0.], [0., f_v_11(2., 4.), 0., f_v_01(2., 4.)], [f_v_01(2., 4.), 0., f_v(2., 4.), 0.], [0., f_v_01(2., 4.), 0., f_v(2., 4.)]]])},
                   {'_a': np.concatenate([np.array([[[1., f_a_1(1.)], [0., f_a(1.)]]])]*50 + 
                                         [np.array([[[1., f_a_1(2.)], [0., f_a(2.)]]])]*50), 
                    '_b': np.concatenate([np.array([[f_b_1(1., 1.), f_b(1., 1.)]])]*50 + 
                                         [np.array([[f_b_1(2., 2.), f_b(2., 2.)]])]*50), 
                    '_v': np.concatenate([np.array([[[f_v_11(1., 1.), f_v_01(1., 1.)], [f_v_01(1., 1.), f_v(1., 1.)]]])]*50 + 
                                         [np.array([[[f_v_11(2., 4.), f_v_01(2., 4.)], [f_v_01(2., 4.), f_v(2., 4.)]]])]*50)}, 
                   {'_a': np.concatenate([np.array([[[1., 0., f_a_1(1.), 0.], [0., 1., 0., f_a_1(1.)], [0., 0., f_a(1.), 0.], [0., 0., 0., f_a(1.)]]])]*50 + 
                                         [np.array([[[1., 0., f_a_1(2.), 0.], [0., 1., 0., f_a_1(2.)], [0., 0., f_a(2.), 0.], [0., 0., 0., f_a(2.)]]])]*50), 
                    '_b': np.concatenate([np.array([[f_b_1(1., 1.), f_b_1(1., 1.), f_b(1., 1.), f_b(1., 1.)]])]*50 + 
                                         [np.array([[f_b_1(2., 2.), f_b_1(2., 2.), f_b(2., 2.), f_b(2., 2.)]])]*50), 
                    '_v': np.concatenate([np.array([[[f_v_11(1., 1.), 0., f_v_01(1., 1.,), 0.], [0., f_v_11(1., 1.), 0., f_v_01(1., 1.)], [f_v_01(1., 1.), 0., f_v(1., 1.), 0.], [0., f_v_01(1., 1.), 0., f_v(1., 1.)]]])]*50 + 
                                         [np.array([[[f_v_11(2., 4.), 0., f_v_01(2., 4.,), 0.], [0., f_v_11(2., 4.), 0., f_v_01(2., 4.)], [f_v_01(2., 4.), 0., f_v(2., 4.), 0.], [0., f_v_01(2., 4.), 0., f_v(2., 4.)]]])]*50)}, 
                   ]

# The following code will run all the unit tests manually: you can then use the internal debugger of VSCode 
# to step through and debug:

if __name__ == '__main__':
    # test_classes = [TestMvBrownianMotion, TestMvIndepBrownianMotion, TestIntegratedBrownianMotion, TestIntegratedIndepBrownianMotion, TestMvOrnsteinUhlenbeck, TestMvIndepOrnsteinUhlenbeck, TestTwiceIntegratedBrownianMotion, TestTwiceIntegratedIndepBrownianMotion]
    test_classes = [TestIntegratedOrnsteinUhlenbeck, TestIntegratedIndepOrnsteinUhlenbeck]
    for test_class in test_classes:
        test_class.setUpClass()
        test_instance = test_class()
        methods_to_test = ['_a', '_b', '_v']
        for method_to_test in methods_to_test:
            test_instance.setUp()
            test_method = getattr(test_instance, 'test' + method_to_test)
            test_method()
            # try:
            #     test_method()
            # except AssertionError as e:
            #     print(f"Test failed for {method_to_test} in {test_class.__name__}: {e}")
            test_instance.tearDown()
        test_class.tearDownClass()