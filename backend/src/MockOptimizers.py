import numpy as np
import matplotlib.pyplot as plt

class MockOptimizers:
    def __init__(self, x=None, y=None):
        # We don't need real data for the mock, just a fake "true minimum" target
        self.true_a0 = 5.0
        self.true_a1 = 2.5

    def gradient_descent(self, alpha, a0_init, a1_init, max_iter=100, tol=1e-6):
        history = []
        a0, a1 = a0_init, a1_init
        loss = 5000.0

        for i in range(max_iter):
            history.append({'iter': i, 'a0': a0, 'a1': a1, 'loss': loss})

            # Simulate behavior based on learning rate (alpha)
            if alpha >= 0.5: 
                # Simulate overshooting / divergence
                a0 += (np.random.rand() - 0.5) * alpha * 20
                a1 += (np.random.rand() - 0.5) * alpha * 20
                loss = loss * 1.1 + 100 
            else: 
                # Simulate normal convergence
                a0 = a0 - alpha * (a0 - self.true_a0)
                a1 = a1 - alpha * (a1 - self.true_a1)
                loss = loss * (1 - alpha/2)

            # Stop conditions
            if loss < tol or loss > 1e6:
                break
                
        return history

    def newtons_method(self, alpha, a0_init, a1_init, max_iter=100, tol=1e-6):
        history = []
        a0, a1 = a0_init, a1_init
        loss = 5000.0

        for i in range(max_iter):
            history.append({'iter': i, 'a0': a0, 'a1': a1, 'loss': loss})

            # Check distance from true minimum for stability analysis
            dist = np.sqrt((a0 - self.true_a0)**2 + (a1 - self.true_a1)**2)
            
            if dist > 1000: 
                # Simulate Newton's method diverging if initial guess is too far
                a0 += np.random.randn() * 500
                a1 += np.random.randn() * 500
                loss *= 1.5
            else: 
                # Simulate rapid, aggressive convergence (typical of Newton)
                a0 = self.true_a0 + (a0 - self.true_a0) * 0.1
                a1 = self.true_a1 + (a1 - self.true_a1) * 0.1
                loss *= 0.05

            if loss < tol or loss > 1e6:
                break
                
        return history