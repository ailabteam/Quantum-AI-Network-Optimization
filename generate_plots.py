# file: generate_plots.py
# Version 3.2: Fixed the text annotation visibility issue by removing the bbox parameter.
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Global plot settings for consistency ---
plt.style.use('seaborn-v0_8-whitegrid')
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12

def generate_static_optimizer_plots():
    """
    Generates and saves SEPARATE convergence plots for VQE and QAOA,
    including visible text annotations to explain the results.
    """
    print("Generating separate plots for VQE and QAOA convergence...")

    # --- Data from terminal outputs ---
    vqe_steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]
    vqe_energy = [216.75, 184.09, 170.34, 164.46, 162.83, 162.53, 162.503, 162.5003, 162.50005, 162.50000, 162.50000, 162.50000, 162.50000, 162.50000, 162.50000, 162.50000, 162.50000, 162.504, 162.5000, 162.5000]
    
    qaoa_steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
    qaoa_energy = [322.10, 330.77, 324.50, 390.31, 341.56, 416.75, 337.08, 309.91, 326.20, 317.49, 322.82, 275.55, 283.64, 259.38, 355.62]
    
    output_dir = os.path.join("results", "figures")

    # --- Plot 1: VQE Convergence ---
    plt.figure(figsize=(7, 6))
    ax1 = plt.gca()
    ax1.plot(vqe_steps, vqe_energy, marker='o', linestyle='-', color='royalblue')
    ax1.set_title('VQE Convergence for Shortest Path', fontsize=TITLE_FONTSIZE)
    ax1.set_xlabel('Optimization Steps', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('Energy (Cost Function Value)', fontsize=LABEL_FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # *** FIXED TEXT ANNOTATION ***
    # Removed bbox parameter for reliability
    ax1.text(0.5, 0.5, 'Converged to an\nInvalid Solution State',
             horizontalalignment='center', verticalalignment='center',
             transform=ax1.transAxes, fontsize=14, color='darkred', weight='bold')

    # Save the VQE figure
    vqe_output_path = os.path.join(output_dir, "vqe_sp_convergence.png")
    plt.savefig(vqe_output_path, dpi=600, bbox_inches='tight')
    print(f"VQE plot saved to '{vqe_output_path}'")
    plt.close()

    # --- Plot 2: QAOA Convergence ---
    plt.figure(figsize=(7, 6))
    ax2 = plt.gca()
    ax2.plot(qaoa_steps, qaoa_energy, marker='x', linestyle='--', color='crimson')
    ax2.set_title('QAOA Convergence for Shortest Path', fontsize=TITLE_FONTSIZE)
    ax2.set_xlabel('Optimization Steps', fontsize=LABEL_FONTSIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # *** FIXED TEXT ANNOTATION ***
    # Removed bbox parameter for reliability
    ax2.text(0.5, 0.5, 'Non-Convergent Behavior\n(Barren Plateau Indication)',
             horizontalalignment='center', verticalalignment='center',
             transform=ax2.transAxes, fontsize=14, color='darkred', weight='bold')

    # Save the QAOA figure
    qaoa_output_path = os.path.join(output_dir, "qaoa_sp_convergence.png")
    plt.savefig(qaoa_output_path, dpi=600, bbox_inches='tight')
    print(f"QAOA plot saved to '{qaoa_output_path}'")
    plt.close()


def plot_qrl_learning_curve():
    # ... (this function remains the same as it was correct)
    print("Generating QRL Agent Learning Curve plot...")
    episodes = np.arange(100, 3001, 100)
    avg_rewards = [35.61, 45.70, 30.07, 34.60, 30.13, 29.29, 44.86, 26.99, 39.55, 24.08,
                   21.84, 27.50, 35.70, 31.77, 47.27, 36.81, 12.28, 15.78, 45.06, 20.67,
                   39.05, 44.35, 32.72, 25.79, 43.55, 37.67, 21.94, 30.33, 20.19, 24.73]
    success_rates = [0.81, 0.86, 0.76, 0.79, 0.77, 0.78, 0.87, 0.75, 0.81, 0.76,
                     0.76, 0.74, 0.81, 0.81, 0.84, 0.79, 0.68, 0.71, 0.84, 0.76,
                     0.86, 0.79, 0.81, 0.75, 0.84, 0.80, 0.77, 0.76, 0.70, 0.74]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Episode', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('Success Rate (100-ep moving average)', color=color, fontsize=LABEL_FONTSIZE)
    ax1.plot(episodes, success_rates, color=color, marker='o', linestyle='-', label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=TICK_FONTSIZE)
    ax1.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    avg_sr = np.mean(success_rates)
    ax1.axhline(y=avg_sr, color=color, linestyle=':', alpha=0.8, label=f'Avg. Success Rate ({avg_sr:.2f})')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Reward (100-ep moving average)', color=color, fontsize=LABEL_FONTSIZE)
    ax2.plot(episodes, avg_rewards, color=color, marker='x', linestyle='--', label='Average Reward')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=TICK_FONTSIZE)
    plt.title('QRL Agent Performance in Dynamic 8-Node Environment', fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    output_path = os.path.join("results", "figures", "qrl_learning_curve.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"QRL plot saved to '{output_path}'")
    plt.close()


def main():
    # ... (this function remains the same)
    output_dir = os.path.join("results", "figures")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory '{output_dir}'")
    generate_static_optimizer_plots()
    plot_qrl_learning_curve()
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
