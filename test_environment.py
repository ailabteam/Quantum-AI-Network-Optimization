# file: test_environment.py
from environments.dynamic_network_env import DynamicNetworkEnv
import random
import networkx as nx

def main():
    print("--- Bắt đầu kiểm tra môi trường mạng động ---")
    
    # Khởi tạo môi trường
    env = DynamicNetworkEnv(num_nodes=8, change_interval=5, max_steps_per_episode=15)
    
    # Chạy thử 10 episodes
    total_rewards = []
    success_count = 0
    
    for episode in range(10):
        print(f"\n--- Episode {episode + 1} ---")
        state = env.reset()
        done = False
        episode_reward = 0
        
        print(f"Nhiệm vụ mới: Đi từ {env.source} đến {env.destination}")
        
        while not done:
            # Agent "ngu ngốc": Chọn một hành động ngẫu nhiên từ các node kề
            current_neighbors = list(env.graph.neighbors(env.current_node))
            if not current_neighbors:
                print("Không có đường đi tiếp! Kết thúc.")
                break # Bị kẹt
            
            action = random.choice(current_neighbors)
            
            print(f"  Tại node {env.current_node}, chọn đi đến {action}")
            
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            
            if done:
                print(f"Episode kết thúc. Lý do: {info.get('status', 'unknown')}")
                print(f"Tổng thưởng episode này: {episode_reward}")
                if info.get('status') == 'success':
                    success_count += 1
        
        total_rewards.append(episode_reward)

    print("\n--- Tổng kết ---")
    print(f"Tỷ lệ thành công của agent ngẫu nhiên: {success_count / 10 * 100:.2f}%")
    print(f"Thưởng trung bình mỗi episode: {sum(total_rewards)/len(total_rewards):.2f}")


if __name__ == "__main__":
    main()
