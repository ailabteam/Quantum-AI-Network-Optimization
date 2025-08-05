# file: run_qrl_training.py (Phiên bản Cải tiến)
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque # Dùng để tính running average hiệu quả hơn

from environments.dynamic_network_env import DynamicNetworkEnv
from algorithms.qrl_agent import QRLAgent

def train():
    # --- 1. Thiết lập các siêu tham số ĐÃ ĐƯỢC CẢI TIẾN ---
    num_episodes = 3000       # Tăng số episodes để có thêm thời gian học
    learning_rate = 0.001     # GIẢM learning rate để học ổn định hơn
    n_layers = 6              # TĂNG độ phức tạp của PQC
    
    num_nodes = 8
    state_size = num_nodes + num_nodes + (num_nodes * num_nodes)
    action_size = num_nodes
    
    # Tạo môi trường và agent với cấu hình mới
    env = DynamicNetworkEnv(num_nodes=num_nodes, change_interval=10, max_steps_per_episode=25) # Tăng max_steps một chút
    agent = QRLAgent(state_size, action_size, n_layers=n_layers, learning_rate=learning_rate)
    
    print("--- Bắt đầu quá trình huấn luyện QRL Agent (Cải tiến) ---")
    print(f"Cấu hình: episodes={num_episodes}, lr={learning_rate}, n_layers={n_layers}")
    
    # --- 2. Vòng lặp huấn luyện ---
    all_rewards = []
    # Sử dụng deque để lưu 100 kết quả gần nhất một cách hiệu quả
    recent_outcomes = deque(maxlen=100) 
    
    # List để vẽ biểu đồ
    avg_rewards_history = []
    success_rate_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0 # Dùng float
        
        # Chạy một episode
        while not done:
            possible_actions = list(env.graph.neighbors(env.current_node))
            if not possible_actions:
                break
            
            action = agent.choose_action(state, possible_actions)
            if action is None: # Xử lý trường hợp không có hành động
                break
            
            next_state, reward, done, info = env.step(action)
            
            agent.rewards.append(reward)
            state = next_state
            episode_reward += reward
        
        # Agent học
        agent.learn()
        
        # --- 3. Ghi lại kết quả và theo dõi ---
        all_rewards.append(episode_reward)
        # Lưu lại kết quả của episode này (1.0 là thành công, 0.0 là thất bại)
        recent_outcomes.append(1.0 if episode_reward > 0 else 0.0)
        
        if (episode + 1) % 100 == 0:
            # Tính toán trên 100 episodes gần nhất
            avg_reward = np.mean(all_rewards[-100:])
            success_rate = np.mean(list(recent_outcomes))
            
            # Lưu lại để vẽ biểu đồ
            avg_rewards_history.append(avg_reward)
            success_rate_history.append(success_rate)
            
            print(f"Episode {episode+1:4d}/{num_episodes} | Phần thưởng TB: {avg_reward:7.2f} | Tỷ lệ thành công: {success_rate:.2f}")

    print("\nHuấn luyện hoàn tất!")
    
    # --- 4. Trực quan hóa kết quả huấn luyện ---
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    # Làm mượt đường cong phần thưởng tổng thể
    running_avg = np.convolve(all_rewards, np.ones(100)/100, mode='valid')
    plt.plot(running_avg)
    plt.title("Phần thưởng trung bình qua các Episode (làm mượt)")
    plt.xlabel("Episode")
    plt.ylabel("Phần thưởng")

    plt.subplot(1, 2, 2)
    # Vẽ các chỉ số tính trên cửa sổ 100 episodes
    x_axis = np.arange(100, num_episodes + 1, 100)
    plt.plot(x_axis, success_rate_history, label='Tỷ lệ thành công')
    plt.title("Tỷ lệ thành công (cửa sổ 100 episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Tỷ lệ thành công")
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    # Thêm biểu đồ phần thưởng trung bình vào cùng một plot
    ax2 = plt.gca().twinx()
    ax2.plot(x_axis, avg_rewards_history, 'r-', label='Phần thưởng TB')
    ax2.set_ylabel('Phần thưởng TB', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.tight_layout()
    plt.savefig("qrl_training_results_improved.png")
    print("Đã lưu biểu đồ kết quả huấn luyện vào 'qrl_training_results_improved.png'")
    
    torch.save(agent.params, "qrl_agent_params_improved.pth")
    print("Đã lưu các tham số của agent đã huấn luyện vào 'qrl_agent_params_improved.pth'")

if __name__ == "__main__":
    train()
