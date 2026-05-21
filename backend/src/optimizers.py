import numpy as np

class Optimizers:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def compute_loss(self, a0, a1):
        loss = 0.0
        for xi, yi in zip(self.x, self.y):
            loss += (yi - (a0 + a1 * xi)) ** 2
        return 0.5 * loss

    def compute_gradients(self, a0, a1):
        dL_da0 = 0.0
        dL_da1 = 0.0
        for xi, yi in zip(self.x, self.y):
            error = yi - (a0 + a1 * xi)
            dL_da0 -= error
            dL_da1 -= error * xi
        return dL_da0, dL_da1

    def compute_hessian(self, a0, a1):
        n = len(self.x)
        sum_x = sum(self.x)
        sum_x_sq = sum(xi ** 2 for xi in self.x)
        
        H = np.array([
            [n, sum_x],
            [sum_x, sum_x_sq]
        ])
        dL_da0, dL_da1 = self.compute_gradients(a0, a1)
        J = np.array([dL_da0, dL_da1])
        return H, J

    def gradient_descent(self, alpha, a0_init, a1_init, max_iter=1000, tol=1e-6):
        a0, a1 = a0_init, a1_init
        history = []
        loss_old = float('inf')
        status = "Max Iterations Reached (Too Slow)"

        for i in range(max_iter):
            loss = self.compute_loss(a0, a1)
            if np.isnan(loss) or np.isinf(loss) or loss > 1e150:
                status = "Diverged (Overshot Minimum)"
                print(f"Algorithm diverged at iteration {i}. Stopping early.")
                break

            history.append({'iter': i, 'a0': float(a0), 'a1': float(a1), 'loss': float(loss)})   
            if abs(loss_old - loss) < tol:
                status = "Optimal (Converged Successfully)"
                break
                
            loss_old = loss
            dL_da0, dL_da1 = self.compute_gradients(a0, a1)
            a0 = a0 - alpha * dL_da0
            a1 = a1 - alpha * dL_da1
            
        return {"history": history, "status": status}

    def newtons_method(self, alpha, a0_init, a1_init, max_iter=1000, tol=1e-6):
        a = np.array([a0_init, a1_init])
        history = []
        loss_old = float('inf')
        status = "Max Iterations Reached (Too Slow)"

        for i in range(max_iter):
            a0, a1 = a[0], a[1]
            loss = self.compute_loss(a0, a1)
            if np.isnan(loss) or np.isinf(loss) or loss > 1e150:
                status = "Diverged (Overshot Minimum)"
                print(f"Algorithm diverged at iteration {i}. Stopping early.")
                break
            history.append({'iter': i, 'a0': float(a0), 'a1': float(a1), 'loss': float(loss)})
            
            if abs(loss_old - loss) < tol:
                status = "Optimal (Converged Successfully)"
                break
                
            loss_old = loss
            H, J = self.compute_hessian(a0, a1)
            
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                break # Stop if Hessian is singular
                
            a = a - alpha * np.dot(H_inv, J)
            
        return {"history": history, "status": status}