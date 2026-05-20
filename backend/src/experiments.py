# from optimizers import Optimizers
import matplotlib.pyplot as plt

class ExperimentRunner:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # self.opts = Optimizers(x, y)

    def compare_learning_rates(self, rates_to_test: list):
        # TODO (Antonyos): Loop through rates_to_test (e.g., [0.001, 0.01, 0.5, 1.0]).
        # TODO (Antonyos): Run gd for each rate and observe aggressive overshooting vs slow convergence.
        # TODO (Antonyos): Format the results to easily show why a specific rate is optimal.
        pass

    def generate_report_plots(self, gd_history, newton_history):
        # TODO (Antonyos): Path Visualization: Plot the loss function curve and steps taken by each algorithm.
        # TODO (Antonyos): Weight Visualization: Plot (a0, a1) at each step for both GD and Newton.
        # TODO (Antonyos): Save these plots to a 'reports/figures' folder using matplotlib.
        pass

    def analyze_stability(self):
        # TODO (Antonyos): Write a script that initializes the weights very far from the optimum.
        # TODO (Antonyos): Run both GD and Newton's method from this bad initial guess.
        # TODO (Antonyos): Log the results to document why Newton's might diverge while GD takes conservative steps.
        # Note: This output will form the basis of the Stability Analysis section of your report.
        pass