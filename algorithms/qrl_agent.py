# file: algorithms/qrl_agent.py (Sửa lỗi grad_fn)
import pennylane as qml
from pennylane import numpy as pnp
import torch
from torch.optim import Adam
import numpy as np

class QRLAgent:
    def __init__(self, state_size, action_space_size, n_layers=4, learning_rate=0.01, gamma=0.99):
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.n_qubits = int(np.ceil(np.log2(action_space_size)))
        self.n_layers = n_layers
        self.gamma = gamma

        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # --- SỬA LỖI Ở ĐÂY ---
        # 1. Định nghĩa QNode một lần duy nhất trong __init__
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

        # 2. Khởi tạo tham số và đảm bảo nó được QNode "nhìn thấy"
        #    Chúng ta sẽ truyền tham số vào mạch một cách tường minh
        param_shape = (self.n_layers, self.state_size)
        self.params = torch.tensor(np.random.uniform(0, 2 * np.pi, param_shape), requires_grad=True)

        self.optimizer = Adam([self.params], lr=learning_rate)
        
        self.rewards = []
        self.log_probs = []

    def _circuit(self, state, params):
        """
        PQC bây giờ nhận cả state và params làm đối số.
        """
        for i in range(self.n_qubits):
            qml.RY(np.pi * state[i % self.state_size], wires=i)

        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RX(params[layer, i % self.state_size], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        
        return qml.probs(wires=range(self.n_qubits))

    def get_policy(self, state):
        """
        Chạy QNode đã được định nghĩa sẵn.
        """
        # Chạy QNode và truyền cả state và self.params
        full_probs = self.qnode(state, self.params)
        return full_probs[:self.action_space_size]

    def choose_action(self, state, possible_actions):
        # ... (Hàm này giữ nguyên, không cần thay đổi)
        probs = self.get_policy(state)
        mask = torch.zeros_like(probs)
        if possible_actions:
             mask[possible_actions] = 1.0
        
        masked_probs = probs * mask
        if torch.sum(masked_probs) > 1e-8:
            masked_probs /= torch.sum(masked_probs)
        else:
            if torch.sum(mask) > 0:
                masked_probs = mask / torch.sum(mask)
            else: # Trường hợp không có hành động nào
                return None 
        
        action_dist = torch.distributions.Categorical(masked_probs)
        action = action_dist.sample()
        self.log_probs.append(action_dist.log_prob(action))
        return action.item()

    def learn(self):
        # ... (Hàm này giữ nguyên, không cần thay đổi)
        if not self.log_probs:
            return

        discounted_rewards = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        rewards_tensor = torch.tensor(discounted_rewards)
        # Sửa UserWarning: chỉ chuẩn hóa nếu có nhiều hơn 1 phần tử
        if len(rewards_tensor) > 1:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-9)
        
        policy_loss = []
        for log_prob, R in zip(self.log_probs, rewards_tensor):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        self.rewards = []
        self.log_probs = []
