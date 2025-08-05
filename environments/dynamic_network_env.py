# file: environments/dynamic_network_env.py
import networkx as nx
import numpy as np
import random

class DynamicNetworkEnv:
    """
    Một môi trường mô phỏng mạng động cho Học tăng cường.
    Tuân thủ một phần giao diện của Gymnasium (thư viện RL phổ biến).
    """
    def __init__(self, num_nodes=8, change_interval=10, max_steps_per_episode=20):
        self.num_nodes = num_nodes
        self.change_interval = change_interval # Thay đổi mạng sau mỗi X bước
        self.max_steps_per_episode = max_steps_per_episode
        
        # Khởi tạo đồ thị ban đầu (ví dụ: Barabasi-Albert)
        self.graph = nx.barabasi_albert_graph(self.num_nodes, m=2, seed=42)
        for (u, v) in self.graph.edges():
            self.graph.edges[u, v]['weight'] = random.randint(1, 10)
        
        self.source = 0
        self.destination = 0
        self.current_node = 0
        self.steps_taken = 0
        self.topology_timer = 0
        
        print(f"Môi trường mạng động đã được khởi tạo với {self.num_nodes} đỉnh.")

    def _get_state(self):
        """
        Trả về trạng thái hiện tại của môi trường.
        Trạng thái bao gồm: vị trí hiện tại, đích, và ma trận kề.
        """
        # Ma trận kề, 1.0 nếu có cạnh, 0.0 nếu không
        adj_matrix = nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes()))
        
        # One-hot encoding cho vị trí hiện tại và đích
        current_loc_one_hot = np.zeros(self.num_nodes)
        current_loc_one_hot[self.current_node] = 1.0
        
        dest_loc_one_hot = np.zeros(self.num_nodes)
        dest_loc_one_hot[self.destination] = 1.0
        
        # Ghép tất cả lại thành một vector trạng thái duy nhất
        state_vector = np.concatenate([current_loc_one_hot, dest_loc_one_hot, adj_matrix.flatten()])
        return state_vector

    def _change_topology(self):
        """Thay đổi cấu trúc mạng một cách ngẫu nhiên."""
        if len(self.graph.edges) == 0 or self.num_nodes < 2:
            return

        # 1. Xóa một cạnh ngẫu nhiên
        edge_to_remove = random.choice(list(self.graph.edges()))
        self.graph.remove_edge(*edge_to_remove)

        # 2. Thêm một cạnh mới ngẫu nhiên (giữa 2 node chưa có kết nối)
        non_edges = list(nx.non_edges(self.graph))
        if non_edges:
            u, v = random.choice(non_edges)
            self.graph.add_edge(u, v, weight=random.randint(1, 10))
        
        # print("--- Cấu trúc mạng đã thay đổi! ---")

    def reset(self):
        """
        Bắt đầu một episode mới.
        Chọn ngẫu nhiên điểm đầu và điểm cuối mới.
        """
        # Chọn ngẫu nhiên source và destination, đảm bảo chúng khác nhau
        self.source, self.destination = random.sample(range(self.num_nodes), 2)
        self.current_node = self.source
        self.steps_taken = 0
        self.topology_timer = 0
        
        return self._get_state()

    def step(self, action_node):
        """
        Thực hiện một bước trong môi trường.
        - action_node: Đỉnh tiếp theo mà agent chọn.
        """
        reward = 0
        done = False
        info = {}

        # Kiểm tra xem hành động có hợp lệ không
        if not self.graph.has_edge(self.current_node, action_node):
            # Phạt nặng cho hành động không hợp lệ (chọn một node không kề)
            reward = -50 
            done = True # Kết thúc episode vì agent đã "phá luật"
            info = {'status': 'invalid_action'}
        else:
            # Hành động hợp lệ, di chuyển đến node tiếp theo
            latency = self.graph.edges[self.current_node, action_node]['weight']
            reward = -latency # Phần thưởng là âm của latency
            self.current_node = action_node

            if self.current_node == self.destination:
                # Đã đến đích!
                reward += 100 # Thưởng lớn
                done = True
                info = {'status': 'success'}
            
        self.steps_taken += 1
        self.topology_timer += 1

        # Kiểm tra xem đã đến lúc thay đổi mạng chưa
        if self.topology_timer % self.change_interval == 0 and not done:
            self._change_topology()

        # Kiểm tra xem có hết thời gian không
        if self.steps_taken >= self.max_steps_per_episode and not done:
            done = True
            reward -= 20 # Phạt vì không đến được đích
            info = {'status': 'timeout'}
        
        next_state = self._get_state()
        
        return next_state, reward, done, info
