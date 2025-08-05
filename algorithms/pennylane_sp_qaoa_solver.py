# file: algorithms/pennylane_sp_qaoa_solver.py
import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np

# Tái sử dụng hàm xây dựng Hamiltonian từ VQE solver
# Đảm bảo import từ đúng file
from algorithms.pennylane_sp_solver import build_shortest_path_hamiltonian

def solve_shortest_path_pennylane_qaoa(graph, start_node, end_node, n_layers=3, n_steps=200, learning_rate=0.1):
    n_nodes = graph.number_of_nodes()
    n_qubits = n_nodes * n_nodes
    
    # 1. Xây dựng Hamiltonian chi phí (Cost Hamiltonian)
    # Chúng ta tái sử dụng hàm đã viết, nó chính là H_cost của chúng ta
    total_weight = sum(d['weight'] for _, _, d in graph.edges(data=True))
    penalty_val = 2 * total_weight
    print(f"INFO: [QAOA] Sử dụng hệ số phạt (penalty) = {penalty_val:.2f}")
    cost_h = build_shortest_path_hamiltonian(graph, start_node, end_node, penalty=penalty_val)
    print(f"Đã xây dựng Hamiltonian chi phí với {n_qubits} qubits.")
    
    # 2. Xây dựng Hamiltonian bộ trộn (Mixer Hamiltonian)
    # Mixer chuẩn cho QAOA là tổng của các toán tử PauliX
    mixer_h = qml.Hamiltonian([1] * n_qubits, [qml.PauliX(i) for i in range(n_qubits)])
    
    # 3. Thiết lập và chạy QAOA
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def circuit(params):
        # params có shape (2, n_layers): gammas và betas
        for i in range(n_qubits):
            qml.Hadamard(wires=i) # Trạng thái ban đầu
        
        for i in range(n_layers):
            qml.ApproxTimeEvolution(cost_h, params[0, i], 1)
            qml.ApproxTimeEvolution(mixer_h, params[1, i], 1)
        
        return qml.expval(cost_h)

    optimizer = torch.optim.Adam
    params = torch.tensor(pnp.random.uniform(0, np.pi, (2, n_layers)), requires_grad=True)
    opt = optimizer([params], lr=learning_rate)

    print(f"Bắt đầu tối ưu hóa QAOA cho {n_qubits}-qubit Shortest Path...")
    for step in range(n_steps):
        opt.zero_grad()
        loss = circuit(params)
        loss_real = torch.real(loss)
        loss_real.backward()
        opt.step()
        if (step + 1) % 20 == 0:
            print(f"  QAOA Step {step+1:3d}: Cost (Energy) = {loss_real.item():.7f}")
            
    # 4. Diễn giải kết quả
    @qml.qnode(dev)
    def get_probabilities(params):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        
        detached_params = params.detach()
        for i in range(n_layers):
            qml.ApproxTimeEvolution(cost_h, detached_params[0, i], 1)
            qml.ApproxTimeEvolution(mixer_h, detached_params[1, i], 1)
            
        return qml.probs(wires=range(n_qubits))

    probs = get_probabilities(params)
    bit_string = format(pnp.argmax(probs), f'0{n_qubits}b')
    
    # Tái sử dụng logic diễn giải từ VQE solver
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
