# file: generate_plots.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_static_optimizer_convergence():
    """
    Vẽ và lưu biểu đồ hội tụ cho VQE và QAOA (Figure 1).
    Dữ liệu được lấy trực tiếp từ output terminal bạn đã gửi.
    """
    print("Đang tạo Figure 1: Biểu đồ hội tụ của VQE và QAOA...")

    # --- Dữ liệu từ output ---
    # VQE cho Shortest Path (4-node)
    vqe_steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]
    vqe_energy = [216.75, 184.09, 170.34, 164.46, 162.83, 162.53, 162.503, 162.5003, 162.50005, 162.50000, 162.50000, 162.50000, 162.50000, 162.50000, 162.50000, 162.50000, 162.50000, 162.504, 162.5000, 162.5000]
    
    # QAOA cho Shortest Path (4-node)
    qaoa_steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
    qaoa_energy = [322.10, 330.77, 324.50, 390.31, 341.56, 416.75, 337.08, 309.91, 326.20, 317.49, 322.82, 275.55, 283.64, 259.38, 355.62]

    # --- Cài đặt biểu đồ ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Biểu đồ VQE
    ax1.plot(vqe_steps, vqe_energy, marker='o', linestyle='-', color='b')
    ax1.set_title('VQE Convergence for Shortest Path', fontsize=14)
    ax1.set_xlabel('Optimization Steps', fontsize=12)
    ax1.set_ylabel('Energy (Cost)', fontsize=12)
    ax1.text(0.5, 0.5, 'Converged to Invalid Solution',
             horizontalalignment='center', verticalalignment='center',
             transform=ax1.transAxes, fontsize=12, color='red', alpha=0.7)

    # Biểu đồ QAOA
    ax2.plot(qaoa_steps, qaoa_energy, marker='x', linestyle='--', color='r')
    ax2.set_title('QAOA Convergence for Shortest Path', fontsize=14)
    ax2.set_xlabel('Optimization Steps', fontsize=12)
    # ax2.set_ylabel('Energy (Cost)', fontsize=12) # Trục y giống nhau
    ax2.text(0.5, 0.5, 'Non-Convergent (Barren Plateau)',
             horizontalalignment='center', verticalalignment='center',
             transform=ax2.transAxes, fontsize=12, color='red', alpha=0.7)

    fig.suptitle('Figure 1: Static Optimizer Performance on 4-Node (16-Qubit) Routing Task', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Điều chỉnh để title không bị đè

    # Lưu file
    output_path = "figures/static_optimizers_convergence.png"
    plt.savefig(output_path, dpi=600)
    print(f"Đã lưu biểu đồ vào '{output_path}'")
    plt.close()


def plot_qrl_learning_curve():
    """
    Vẽ và lưu biểu đồ learning curve cho QRL (Figure 2).
    Dữ liệu được lấy từ output terminal bạn đã gửi.
    """
    print("Đang tạo Figure 2: Biểu đồ học tập của QRL...")

    # --- Dữ liệu từ output ---
    episodes = np.arange(100, 3001, 100)
    avg_rewards = [35.61, 45.70, 30.07, 34.60, 30.13, 29.29, 44.86, 26.99, 39.55, 24.08,
                   21.84, 27.50, 35.70, 31.77, 47.27, 36.81, 12.28, 15.78, 45.06, 20.67,
                   39.05, 44.35, 32.72, 25.79, 43.55, 37.67, 21.94, 30.33, 20.19, 24.73]
    success_rates = [0.81, 0.86, 0.76, 0.79, 0.77, 0.78, 0.87, 0.75, 0.81, 0.76,
                     0.76, 0.74, 0.81, 0.81, 0.84, 0.79, 0.68, 0.71, 0.84, 0.76,
                     0.86, 0.79, 0.81, 0.75, 0.84, 0.80, 0.77, 0.76, 0.70, 0.74]

    # --- Cài đặt biểu đồ ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Vẽ Success Rate trên trục y chính (bên trái)
    color = 'tab:blue'
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Success Rate (100-ep window)', color=color, fontsize=12)
    ax1.plot(episodes, success_rates, color=color, marker='o', label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=np.mean(success_rates), color=color, linestyle='--', alpha=0.7, label=f'Avg Success Rate ({np.mean(success_rates):.2f})')

    # Tạo trục y thứ hai (bên phải) cho Average Reward
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Reward (100-ep window)', color=color, fontsize=12)
    ax2.plot(episodes, avg_rewards, color=color, marker='x', linestyle='--', label='Average Reward')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Thêm chú thích và title
    fig.suptitle('Figure 2: QRL Agent Learning Curve in Dynamic 8-Node Environment', fontsize=16)
    fig.tight_layout()
    
    # Lưu file
    output_path = "figures/qrl_learning_curve.png"
    plt.savefig(output_path, dpi=600)
    print(f"Đã lưu biểu đồ vào '{output_path}'")
    plt.close()


def main():
    """
    Hàm chính để tạo tất cả các figures và tables.
    """
    # Tạo thư mục figures nếu chưa có
    if not os.path.exists('figures'):
        os.makedirs('figures')
        print("Đã tạo thư mục 'figures/'.")
        
    # Tạo các biểu đồ
    plot_static_optimizer_convergence()
    plot_qrl_learning_curve()
    
    # In ra mã LaTeX cho Table 1 để dễ dàng sao chép
    #print("\n--- Mã LaTeX cho Table 1 ---")
    #table_latex = r"""
