# file: run_qrl_training.py
import torch
import numpy as np
import matplotlib.pyplot as plt

from environments.dynamic_network_env import DynamicNetworkEnv
from algorithms.qrl_agent import QRLAgent

def train():
    # --- 1. Thiết lập các siêu tham số ---
    num_episodes = 2000
    num_nodes = 8
    state_size = num_nodes + num_nodes + (num_nodes * num_nodes) # current_one_hot + dest_one_hot + adj_matrix
    action_size = num_nodes
    
    # Tạo môi trường và agent
    env = DynamicNetworkEnv(num_nodes=num_nodes, change_interval=10, max_steps_per_episode=20)
    agent = QRLAgent(state_size, action_size, n_layers=4, learning_rate=0.01)
    
    print("Bắt đầu quá trình huấn luyện QRL Agent...")
    
    # --- 2. Vòng lặp huấn luyện ---
    all_rewards = []
    success_rates = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Agent chọn hành động
            possible_actions = list(env.graph.neighbors(env.current_node))
            if not possible_actions:
                break # Bị kẹt
            
            action = agent.choose_action(state, possible_actions)
            
            # Môi trường phản hồi
            next_state, reward, done, info = env.step(action)
            
            # Lưu lại kinh nghiệm
            agent.rewards.append(reward)
            
            state = next_state
            episode_reward += reward
        
        # Agent học từ kinh nghiệm của episode vừa rồi
        agent.learn()
        
        # --- 3. Ghi lại kết quả để theo dõi ---
        all_rewards.append(episode_reward)
        
        # Tính tỷ lệ thành công trong 100 episodes gần nhất
        if (episode + 1) % 100 == 0:
            last_100_rewards = all_rewards[-100:]
            # Đếm xem có bao nhiêu episode thành công (reward > 0)
            success_count = sum(1 for r in last_100_rewards if r > 0)
            success_rate = success_count / 100.0
            success_rates.append(success_rate)
            avg_reward = sum(last_100_rewards) / 100.0
            print(f"Episode {episode+1}/{num_episodes} | Phần thưởng TB (100 ep): {avg_reward:.2f} | Tỷ lệ thành công (100 ep): {success_rate:.2f}")

    print("\nHuấn luyện hoàn tất!")
    
    # --- 4. Trực quan hóa kết quả huấn luyện ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # Làm mượt đường cong phần thưởng
    running_avg = np.convolve(all_rewards, np.ones(100)/100, mode='valid')
    plt.plot(running_avg)
    plt.title("Phần thưởng trung bình qua các Episode")
    plt.xlabel("Episode")
    plt.ylabel("Phần thưởng TB (100 ep)")

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(100, num_episodes + 1, 100), success_rates)
    plt.title("Tỷ lệ thành công qua các Episode")
    plt.xlabel("Episode")
    plt.ylabel("Tỷ lệ thành công (100 ep)")
    plt.ylim(0, 1.1) # Giới hạn trục y từ 0 đến 1.1 (110%)
    
    plt.tight_layout()
    plt.savefig("qrl_training_results.png")
    print("Đã lưu biểu đồ kết quả huấn luyện vào 'qrl_training_results.png'")
    
    # Lưu lại các tham số đã huấn luyện
    torch.save(agent.params, "qrl_agent_params.pth")
    print("Đã lưu các tham số của agent đã huấn luyện vào 'qrl_agent_params.pth'")

if __name__ == "__main__":
    train()
