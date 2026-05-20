# from optimizers import Optimizers
import os

import matplotlib.pyplot as plt

from MockOptimizers import MockOptimizers
class ExperimentRunner:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # self.opts = Optimizers(x, y)
        self.opts = MockOptimizers();

    def compare_learning_rates(self, rates_to_test: list):
        print("\n--- Learning Rate Comparison Report ---")
        print(f"{'Learning Rate':<15} | {'Final Loss':<15} | {'Iterations':<12} | {'Status'}")
        print("-" * 75)

        all_histories = {}

        for alpha in rates_to_test:
            history = self.opts.gradient_descent(alpha, a0_init=0.0, a1_init=0.0, max_iter=200)
            final_step = history[-1]
            final_loss = final_step['loss']
            total_steps = len(history)
            if final_loss > 5000:
                status = "Diverged (Overshot minimum)"
            elif total_steps == 200:
                status = "TOO SLOW (Failed to converge)"
            else:  
                status = "OPTIMAL (Converged successfully)"
            print(f"{alpha:<15} | {final_loss:<15.4f} | {total_steps:<12} | {status}")
            all_histories[alpha] = history

        print("-"*75)
        return all_histories
        pass

    def generate_report_plots(self, gd_history, newton_history):
        """
        Generates and saves comparison plots for Gradient Descent and Newton's Method.
        """
        # 1. Ensure the output directory exists so plt.savefig doesn't crash
        os.makedirs("reports/figures", exist_ok=True)

        # 2. Extract data for plotting from the dictionary lists
        gd_iters = [step['iter'] for step in gd_history]
        gd_loss = [step['loss'] for step in gd_history]
        gd_a0 = [step['a0'] for step in gd_history]
        gd_a1 = [step['a1'] for step in gd_history]

        newton_iters = [step['iter'] for step in newton_history]
        newton_loss = [step['loss'] for step in newton_history]
        newton_a0 = [step['a0'] for step in newton_history]
        newton_a1 = [step['a1'] for step in newton_history]

        # ---------------------------------------------------------
        # Plot 1: Loss Curve (Iterations vs Loss)
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        plt.plot(gd_iters, gd_loss, label='Gradient Descent', color='blue', linewidth=2)
        plt.plot(newton_iters, newton_loss, label="Newton's Method", color='red', linewidth=2, linestyle='--')
        
        plt.title('Loss Curve Comparison: GD vs Newton')
        plt.xlabel('Iterations')
        plt.ylabel('Loss (SSE)')
        plt.yscale('log') # Log scale is highly recommended for loss drops!
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        
        # Save and close the figure to free up memory
        plt.savefig("reports/figures/loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Successfully saved: reports/figures/loss_curve.png")

        # ---------------------------------------------------------
        # Plot 2: Weight Trajectory (a0 vs a1)
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        
        # Plot the paths
        plt.plot(gd_a0, gd_a1, label='Gradient Descent', color='blue', marker='o', markersize=4, alpha=0.6)
        plt.plot(newton_a0, newton_a1, label="Newton's Method", color='red', marker='x', markersize=6)
        
        # Highlight the starting coordinate so the path direction is obvious
        if gd_a0 and gd_a1:
            plt.plot(gd_a0[0], gd_a1[0], marker='s', color='black', markersize=8, label='Start Point')
        
        plt.title('Weight Trajectory ($a_0$ vs $a_1$)')
        plt.xlabel('Weight $a_0$')
        plt.ylabel('Weight $a_1$')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)

        # Save and close
        plt.savefig("reports/figures/weight_path.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Successfully saved: reports/figures/weight_path.png")
        pass

    def analyze_stability(self):
        """
        Tests the robustness of GD vs Newton's method by initializing weights
        extremely far from the true minimum.
        """
        print("\n--- Stability Analysis Report ---")
        print("Test: Extreme Initial Guess (a0 = 10000.0, a1 = 10000.0)")
        print("-" * 75)

        a0 = 10000.0
        a1 = 10000.0
        gd_history = self.opts.gradient_descent(alpha=0.1, a0_init=a0, a1_init=a1, max_iter=200)
        newton_history = self.opts.newtons_method(alpha=0.1, a0_init=a0, a1_init=a1, max_iter=200)
        gd_final = gd_history[-1]
        newton_final = newton_history[-1]
        gd_status = "SURVIVED (Moving safely toward minimum)" if gd_final['loss'] < 5000 else "FAILED"
        newton_status = "EXPLODED (Diverged massively)" if newton_final['loss'] > 5000 else "CONVERGED"

        print(f"Gradient Descent Final Loss: {gd_final['loss']:>20.4f} | {gd_status}")
        print(f"Newton's Method Final Loss:  {newton_final['loss']:>20.4f} | {newton_status}")
        
        print("-" * 75)
        print("Conclusion:")
        print("Gradient Descent takes conservative steps based on the gradient and remains stable.")
        print("Newton's method attempts massive leaps based on local curvature (the Hessian),")
        print("which leads to immediate divergence if the initial guess is too far off.")
        print("-" * 75)
        return gd_history, newton_history
        pass

if __name__ == "__main__":
    # Initialize the runner
    runner = ExperimentRunner(x=None, y=None)
    
    # ---------------------------------------------------------
    # Test 1: Compare Learning Rates
    # ---------------------------------------------------------
    rates = [0.001, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.9, 1.0, 2.0, 5.0]
    runner.compare_learning_rates(rates)
    
    # ---------------------------------------------------------
    # Test 2: Generate Visualizations
    # ---------------------------------------------------------
    # Generate ideal runs to plot
    print("\nGenerating Report Plots...")
    ideal_gd = runner.opts.gradient_descent(alpha=0.1, a0_init=0.0, a1_init=0.0)
    ideal_newton = runner.opts.newtons_method(alpha=1.0, a0_init=0.0, a1_init=0.0)
    runner.generate_report_plots(ideal_gd, ideal_newton)
    
    # ---------------------------------------------------------
    # Test 3: Analyze Stability
    # ---------------------------------------------------------
    # Run the bad-initialization test
    runner.analyze_stability()