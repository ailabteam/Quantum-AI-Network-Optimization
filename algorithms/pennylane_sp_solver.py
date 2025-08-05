# file: algorithms/pennylane_sp_solver.py (Penalty Lớn)
import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np

# ... (hàm qubo_to_ising_hamiltonian và build_shortest_path_hamiltonian giữ nguyên)
def qubo_to_ising_hamiltonian(Q):
    n_vars = Q.shape[0]
    coeffs = []
    obs = []
    offset = 0.
    for i in range(n_vars):
        offset += Q[i,i] / 2.
        for j in range(i+1, n_vars):
            offset += (Q[i,j] + Q[j,i]) / 4.
    for i in range(n_vars):
        h_i = Q[i, i] / 2. + np.sum(Q[i, i+1:] + Q[i+1:, i]) / 4.
        if not np.isclose(h_i, 0):
            coeffs.append(h_i)
            obs.append(qml.PauliZ(i))
        for j in range(i + 1, n_vars):
            J_ij = (Q[i, j] + Q[j, i]) / 4.
            if not np.isclose(J_ij, 0):
                coeffs.append(J_ij)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    hamiltonian = qml.Hamiltonian(coeffs, obs)
    if not np.isclose(offset, 0):
        hamiltonian += offset * qml.Identity(0)
    return hamiltonian

def build_shortest_path_hamiltonian(graph, start_node, end_node, penalty):
    n_nodes = len(graph.nodes())
    n_vars = n_nodes * n_nodes
    Q = np.zeros((n_vars, n_vars))
    def var_index(node_idx, pos_idx):
        return node_idx * n_nodes + pos_idx
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)
        for k in range(n_nodes - 1):
            idx1 = var_index(u, k)
            idx2 = var_index(v, k + 1)
            Q[idx1, idx2] += weight
            Q[idx2, idx1] += weight
    idx_start = var_index(start_node, 0)
    Q[idx_start, idx_start] -= penalty
    for k in range(n_nodes):
        idx_k = var_index(end_node, k)
        Q[idx_k, idx_k] -= penalty
    for k in range(n_nodes):
        for l in range(k + 1, n_nodes):
            idx_k = var_index(end_node, k)
            idx_l = var_index(end_node, l)
            Q[idx_k, idx_l] += 2 * penalty
    for k in range(1, n_nodes):
        for i in range(n_nodes):
            idx_i = var_index(i, k)
            Q[idx_i, idx_i] -= penalty
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                idx_i = var_index(i, k)
                idx_j = var_index(j, k)
                Q[idx_i, idx_j] += 2 * penalty
    for i in range(n_nodes):
        if i == start_node or i == end_node:
            continue
        for k in range(n_nodes):
            idx_k = var_index(i, k)
            Q[idx_k, idx_k] -= penalty
        for k in range(n_nodes):
            for l in range(k + 1, n_nodes):
                idx_l = var_index(i, l)
                idx_k = var_index(i, k)
                Q[idx_k, idx_l] += 2 * penalty
    return qubo_to_ising_hamiltonian(Q)

def solve_shortest_path_pennylane_vqe(graph, start_node, end_node, n_steps=300, learning_rate=0.05, n_layers=4):
    n_nodes = graph.number_of_nodes()
    n_qubits = n_nodes * n_nodes

    # SỬA LỖI Ở ĐÂY: Tăng Penalty lên đáng kể
    total_weight = sum(d['weight'] for _, _, d in graph.edges(data=True))
    penalty_val = 2 * total_weight
    print(f"INFO: Sử dụng hệ số phạt (penalty) = {penalty_val:.2f}")

    cost_h = build_shortest_path_hamiltonian(graph, start_node, end_node, penalty=penalty_val)
    print(f"Đã xây dựng Hamiltonian với {n_qubits} qubits.")

    # ... Phần còn lại của hàm giữ nguyên ...
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
    path_cost = 0.0
    is_valid = True
    if not final_path or final_path[0] != start_node or end_node not in final_path:
        is_valid = False
    else:
        try:
            end_idx = final_path.index(end_node)
            final_path = final_path[:end_idx+1]
            for i in range(len(final_path) - 1):
                u, v = final_path[i], final_path[i+1]
                if graph.has_edge(u, v):
                    path_cost += graph[u][v]['weight']
                else:
                    is_valid = False; break
        except (ValueError, IndexError):
            is_valid = False
    if not is_valid:
        path_cost = float('inf')
    return final_path, path_cost
