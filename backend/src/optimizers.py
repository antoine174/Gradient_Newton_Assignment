import numpy as np

class Optimizers:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def compute_loss(self, a0, a1):
        # TODO (Abd elrahman): Implement the SSE Loss function.
        # L = 0.5 * sum( (y_true - (a0 + a1*x))^2 )
        pass

    def compute_gradients(self, a0, a1):
        # TODO (Abd elrahman): Calculate partial derivative with respect to a0.
        # TODO (Abd elrahman): Calculate partial derivative with respect to a1.
        # Return dL/da0, dL/da1
        pass

    def compute_hessian(self, a0, a1):
        # TODO (Abd elrahman): Calculate the second derivatives to form the 2x2 Hessian matrix.
        # TODO (Abd elrahman): Calculate the Jacobian matrix (the gradients).
        # Return H, J
        pass

    def gradient_descent(self, alpha, a0_init, a1_init, max_iter=1000, tol=1e-6):
        # TODO (Abd elrahman): Implement the GD loop: a_i(k+1) = a_i(k) - alpha * dL/da_i(k)
        # TODO (Abd elrahman): Stop if abs(loss_new - loss_old) < tol or max_iter is reached.
        # MUST RETURN: A list of dictionaries tracking history: [{'iter': i, 'a0': val, 'a1': val, 'loss': val}, ...]
        pass

    def newtons_method(self, alpha, a0_init, a1_init, max_iter=1000, tol=1e-6):
        # TODO (Abd elrahman): Implement the Damped Newton Method loop: X = a - alpha * inv(H) * J
        # TODO (Abd elrahman): Handle matrix inversion using np.linalg.inv.
        # MUST RETURN: The exact same history format as gradient_descent.
        pass