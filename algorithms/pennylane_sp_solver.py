# file: algorithms/pennylane_sp_solver.py (Phiên bản cuối, tự chuyển đổi QUBO->Ising)
import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np

def qubo_to_ising_hamiltonian(Q):
    """
    Tự viết hàm chuyển đổi ma trận QUBO sang PennyLane Hamiltonian.
    Đây là cách làm ổn định và không phụ thuộc vào phiên bản thư viện.
    """
    n_vars = Q.shape[0]
    coeffs = []
    obs = []

    # Xây dựng các thành phần của Hamiltonian
    for i in range(n_vars):
        # Hệ số cục bộ (h_i)
        h_i = 0.5 * Q[i, i] + 0.25 * np.sum(Q[i, :] + Q[:, i]) - 0.25 * (Q[i,i] + Q[i,i])
        if not np.isclose(h_i, 0):
            coeffs.append(h_i)
            obs.append(qml.PauliZ(i))
        
        # Hệ số tương tác (J_ij)
        for j in range(i + 1, n_vars):
            J_ij = 0.25 * (Q[i, j] + Q[j, i])
            if not np.isclose(J_ij, 0):
                coeffs.append(J_ij)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    
    # Hằng số offset cũng có thể được tính, nhưng nó không ảnh hưởng đến việc tìm kiếm
    # trạng thái cơ bản, nên chúng ta có thể bỏ qua để đơn giản hóa.
    
    return qml.Hamiltonian(coeffs, obs)


def build_shortest_path_hamiltonian(graph, start_node, penalty=30.0):
    n_nodes = len(graph.nodes())
    n_vars = n_nodes * n_nodes
    
    Q = np.zeros((n_vars, n_vars))

    def var_index(node_idx, pos_idx):
        return node_idx * n_nodes + pos_idx

    # 1. Hàm mục tiêu
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)
        for k in range(n_nodes - 1):
            idx1 = var_index(u, k)
            idx2 = var_index(v, k + 1)
            Q[idx1, idx2] += weight
            Q[idx2, idx1] += weight

    # 2. Ràng buộc
    # Ràng buộc 1: Bắt đầu ở start_node tại vị trí 0
    idx_start = var_index(start_node, 0)
    Q[idx_start, idx_start] += penalty * -2
    # Cộng hằng số P vào ma trận, sẽ được xử lý sau

    # Ràng buộc 2: Mỗi vị trí k chỉ có 1 node
    for k in range(n_nodes):
        for i in range(n_nodes):
            idx_i = var_index(i, k)
            Q[idx_i, idx_i] += penalty * -2
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                idx_i = var_index(i, k)
                idx_j = var_index(j, k)
                Q[idx_i, idx_j] += penalty * 2
    
    # Ràng buộc 3: Mỗi node i chỉ xuất hiện 1 lần
    for i in range(n_nodes):
        for k in range(n_nodes):
            idx_k = var_index(i, k)
            Q[idx_k, idx_k] += penalty * -2
        for k in range(n_nodes):
            for l in range(k + 1, n_nodes):
                idx_l = var_index(i, l)
                idx_k = var_index(i, k)
                Q[idx_k, idx_l] += penalty * 2
    
    # 3. Chuyển ma trận QUBO (Q) sang Ising Hamiltonian
    hamiltonian = qubo_to_ising_hamiltonian(Q)
    return hamiltonian

# --- Các hàm solve_... và diễn giải kết quả giữ nguyên như phiên bản trước ---
# (Dán phần còn lại của file từ phiên bản trước vào đây)
def solve_shortest_path_pennylane_vqe(graph, start_node, end_node, n_steps=300, learning_rate=0.05, n_layers=4):
    n_nodes = graph.number_of_nodes()
    n_qubits = n_nodes * n_nodes

    max_weight = max(d['weight'] for _, _, d in graph.edges(data=True)) if graph.edges else 1
    penalty_val = float(n_nodes * max_weight)
    cost_h = build_shortest_path_hamiltonian(graph, start_node, penalty=penalty_val)
    print(f"Đã xây dựng Hamiltonian với {n_qubits} qubits.")

    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def cost_function(params):
        qml.templates.BasicEntanglerLayers(weights=params, wires=range(n_qubits))
        return qml.expval(cost_h)

    optimizer = torch.optim.Adam
    params_shape = qml.templates.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    params = torch.tensor(pnp.random.uniform(0, 2 * np.pi, size=params_shape), requires_grad=True)
    opt = optimizer([params], lr=learning_rate)

    print(f"Bắt đầu tối ưu hóa VQE cho {n_qubits}-qubit Shortest Path...")
    for step in range(n_steps):
        opt.zero_grad()
        loss = cost_function(params)
        loss_real = torch.real(loss)
        loss_real.backward()
        opt.step()
        if (step + 1) % 20 == 0:
            print(f"  VQE Step {step+1:3d}: Cost (Energy) = {loss_real.item():.7f}")
    
    @qml.qnode(dev)
    def get_probabilities(params):
        qml.templates.BasicEntanglerLayers(weights=params.detach(), wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))
    
    probs = get_probabilities(params)
    bit_string = format(pnp.argmax(probs), f'0{n_qubits}b')
    
    node_positions = {}
    for k in range(n_nodes):
        for i in range(n_nodes):
            idx = i * n_nodes + k
            if bit_string[idx] == '1':
                node_positions[k] = i

    path = []
    if node_positions:
        sorted_positions = sorted(node_positions.keys())
        path = [node_positions[k] for k in sorted_positions]
    
    final_path = list(dict.fromkeys(path))
    path_cost = 0
    if final_path and final_path[0] == start_node and end_node in final_path:
         try:
            end_idx = final_path.index(end_node)
            final_path = final_path[:end_idx+1]
            for i in range(len(final_path) - 1):
                u, v = final_path[i], final_path[i+1]
                if graph.has_edge(u, v):
                    path_cost += graph[u][v]['weight']
                else:
                    path_cost = float('inf'); break
         except (ValueError, IndexError):
            path_cost = float('inf')
    else:
         path_cost = float('inf')

    return final_path, path_cost
