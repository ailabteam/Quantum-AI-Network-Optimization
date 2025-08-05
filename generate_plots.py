# file: generate_plots.py
# Version 2.0: English labels, larger fonts, output to results/figures/
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Global plot settings for consistency ---
plt.style.use('seaborn-v0_8-whitegrid')
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12

def plot_static_optimizer_convergence():
    """
    Generates and saves the convergence plots for VQE and QAOA (Figure 1).
    Data is hardcoded from the terminal outputs.
    """
    print("Generating Figure 1: VQE and QAOA Convergence Plots...")

    # --- Data from terminal outputs ---
    vqe_steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]
    vqe_energy = [216.75, 184.09, 170.34, 164.46, 162.83, 162.53, 162.503, 162.5003, 162.50005, 162.50000, 162.50000, 162.50000, 162.50000, 162.50000, 162.50000, 162.50000, 162.50000, 162.504, 162.5000, 162.5000]
    
    qaoa_steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
    qaoa_energy = [322.10, 330.77, 324.50, 390.31, 341.56, 416.75, 337.08, 309.91, 326.20, 317.49, 322.82, 275.55, 283.64, 259.38, 355.62]

    # --- Plot setup ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # VQE Plot
    ax1.plot(vqe_steps, vqe_energy, marker='o', linestyle='-', color='royalblue')
    ax1.set_title('VQE Convergence for Shortest Path', fontsize=TITLE_FONTSIZE)
    ax1.set_xlabel('Optimization Steps', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('Energy (Cost Function Value)', fontsize=LABEL_FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax1.text(0.5, 0.6, 'Converged to an\nInvalid Solution State',
             horizontalalignment='center', verticalalignment='center',
             transform=ax1.transAxes, fontsize=14, color='darkred',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # QAOA Plot
    ax2.plot(qaoa_steps, qaoa_energy, marker='x', linestyle='--', color='crimson')
    ax2.set_title('QAOA Convergence for Shortest Path', fontsize=TITLE_FONTSIZE)
    ax2.set_xlabel('Optimization Steps', fontsize=LABEL_FONTSIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax2.text(0.5, 0.6, 'Non-Convergent Behavior\n(Barren Plateau Indication)',
             horizontalalignment='center', verticalalignment='center',
             transform=ax2.transAxes, fontsize=14, color='darkred',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    fig.tight_layout()

    # Save the figure
    output_path = os.path.join("results", "figures", "static_optimizers_convergence.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Figure 1 saved to '{output_path}'")
    plt.close()


def plot_qrl_learning_curve():
    """
    Generates and saves the learning curve plots for the QRL agent (Figure 2).
    """
    print("Generating Figure 2: QRL Agent Learning Curve...")

    # --- Data from terminal outputs ---
    episodes = np.arange(100, 3001, 100)
    avg_rewards = [35.61, 45.70, 30.07, 34.60, 30.13, 29.29, 44.86, 26.99, 39.55, 24.08,
                   21.84, 27.50, 35.70, 31.77, 47.27, 36.81, 12.28, 15.78, 45.06, 20.67,
                   39.05, 44.35, 32.72, 25.79, 43.55, 37.67, 21.94, 30.33, 20.19, 24.73]
    success_rates = [0.81, 0.86, 0.76, 0.79, 0.77, 0.78, 0.87, 0.75, 0.81, 0.76,
                     0.76, 0.74, 0.81, 0.81, 0.84, 0.79, 0.68, 0.71, 0.84, 0.76,
                     0.86, 0.79, 0.81, 0.75, 0.84, 0.80, 0.77, 0.76, 0.70, 0.74]

    # --- Plot setup ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Success Rate on the left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Episode', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('Success Rate (100-ep moving average)', color=color, fontsize=LABEL_FONTSIZE)
    ax1.plot(episodes, success_rates, color=color, marker='o', linestyle='-', label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=TICK_FONTSIZE)
    ax1.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add a horizontal line for the average success rate
    avg_sr = np.mean(success_rates)
    ax1.axhline(y=avg_sr, color=color, linestyle=':', alpha=0.8, label=f'Avg. Success Rate ({avg_sr:.2f})')
    ax1.legend(loc='upper left')

    # Create a second y-axis (right) for the Average Reward
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Reward (100-ep moving average)', color=color, fontsize=LABEL_FONTSIZE)
    ax2.plot(episodes, avg_rewards, color=color, marker='x', linestyle='--', label='Average Reward')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=TICK_FONTSIZE)

    # Title and layout
    plt.title('QRL Agent Performance in Dynamic 8-Node Environment', fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    
    # Save the figure
    output_path = os.path.join("results", "figures", "qrl_learning_curve.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Figure 2 saved to '{output_path}'")
    plt.close()


def main():
    """
    Main function to generate all figures and tables for the paper.
    """
    # Create the output directory structure if it doesn't exist
    output_dir = os.path.join("results", "figures")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory '{output_dir}'")
        
    # Generate the plots
    plot_static_optimizer_convergence()
    plot_qrl_learning_curve()
    
    # Print the LaTeX code for Table 1 for easy copying
    print("\n--- LaTeX Code for Table 1 ---")
    table_latex = r"""
\begin{table}[ht!]
\centering
\caption{Summary of Static Optimization Results}
\label{tab:static_results}
\begin{tabular}{@{}llcc@{}}
\toprule
Problem & Algorithm & Solution Validity & Approx. Ratio \\ \midrule
\textbf{5-Node Max-Cut} & VQE & Invalid & 0.0 \\
(Control Experiment) & QAOA & \textbf{Valid (Optimal)} & \textbf{1.0} \\ \addlinespace
\textbf{4-Node Shortest Path} & VQE & Invalid & N/A \\
(Main Task) & QAOA & Invalid & N/A \\ \bottomrule
\end{tabular}
\end{table}
    """
    print(table_latex)


if __name__ == "__main__":
    main()
