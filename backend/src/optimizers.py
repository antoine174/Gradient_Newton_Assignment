import numpy as np

class Optimizers:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def compute_loss(self, a0, a1):
        return 0.5 * np.sum((self.y - (a0 + a1 * self.x)) ** 2)

    def compute_gradients(self, a0, a1):
        errors = self.y - (a0 + a1 * self.x)
        dL_da0 = -np.sum(errors)
        dL_da1 = -np.sum(errors * self.x)
        return dL_da0, dL_da1

    def compute_hessian(self, a0, a1):
        H = np.array([
            [len(self.x), np.sum(self.x)],
            [np.sum(self.x), np.sum(self.x ** 2)]
        ])
        dL_da0, dL_da1 = self.compute_gradients(a0, a1)
        J = np.array([dL_da0, dL_da1])
        return H, J

    def gradient_descent(self, alpha, a0_init, a1_init, max_iter=1000, tol=1e-6):
        a0, a1 = a0_init, a1_init
        history = []
        loss_old = float('inf')

        for i in range(max_iter):
            loss = self.compute_loss(a0, a1)
            history.append({'iter': i, 'a0': a0, 'a1': a1, 'loss': loss})
            
            if abs(loss_old - loss) < tol:
                break
                
            loss_old = loss
            dL_da0, dL_da1 = self.compute_gradients(a0, a1)
            a0 = a0 - alpha * dL_da0
            a1 = a1 - alpha * dL_da1
            
        return history

    def newtons_method(self, alpha, a0_init, a1_init, max_iter=1000, tol=1e-6):
        a = np.array([a0_init, a1_init])
        history = []
        loss_old = float('inf')

        for i in range(max_iter):
            a0, a1 = a[0], a[1]
            loss = self.compute_loss(a0, a1)
            history.append({'iter': i, 'a0': a0, 'a1': a1, 'loss': loss})
            
            if abs(loss_old - loss) < tol:
                break
                
            loss_old = loss
            H, J = self.compute_hessian(a0, a1)
            
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                break # Stop if Hessian is singular
                
            a = a - alpha * np.dot(H_inv, J)
            
        return history