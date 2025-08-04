# file: algorithms/shortest_path_vqe_solver.py (Sửa lỗi Complex Grad)
import pennylane as qml
from pennylane import numpy as pnp
import torch
import networkx as nx
import numpy as np

# Import các công cụ từ Qiskit để xây dựng bài toán
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

def qp_to_pennylane_hamiltonian(qp: QuadraticProgram):
    """
    Hàm tiện ích để chuyển đổi một QuadraticProgram của Qiskit
    sang một Hamiltonian của PennyLane. Đảm bảo các hệ số là số thực.
    """
    qubo = QuadraticProgramToQubo().convert(qp)
    offset = qubo.objective.constant
    linear_terms = qubo.objective.linear.to_dict()
    quadratic_terms = qubo.objective.quadratic.to_dict()

    coeffs = []
    obs = []

    # Xây dựng Hamiltonian theo công thức chuyển đổi từ QUBO sang Ising
    # Đảm bảo tất cả các hệ số đều là số thực
    for (i, j), coeff in quadratic_terms.items():
        if np.imag(coeff) != 0:
            print(f"Warning: Complex quadratic coefficient found for ({i},{j}): {coeff}")
        
        real_coeff = np.real(coeff)
        coeffs.append(real_coeff / 4)
        obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
        
        coeffs.append(-real_coeff / 4)
        obs.append(qml.PauliZ(i))
        
        coeffs.append(-real_coeff / 4)
        obs.append(qml.PauliZ(j))
        
        offset += real_coeff / 4

    for i, coeff in linear_terms.items():
        if np.imag(coeff) != 0:
            print(f"Warning: Complex linear coefficient found for ({i}): {coeff}")
            
        real_coeff = np.real(coeff)
        coeffs.append(-real_coeff / 2)
        obs.append(qml.PauliZ(i))
        offset += real_coeff / 2

    if not np.isclose(np.real(offset), 0):
        coeffs.append(np.real(offset))
        obs.append(qml.Identity(0))
    
    return qml.Hamiltonian(coeffs, obs)


def solve_shortest_path_vqe(graph, start_node, end_node, penalty=20, n_steps=300, learning_rate=0.1, n_layers=4):
    n_nodes = graph.number_of_nodes()

    # 1. Xây dựng Quadratic Program (QP)
    # (Giữ nguyên, không thay đổi)
    qp = QuadraticProgram(name="Shortest Path")
    for i in range(n_nodes):
        for k in range(n_nodes):
            qp.binary_var(f'x_{i}_{k}')

    objective = {}
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if graph.has_edge(i, j):
                for k in range(n_nodes - 1):
                    objective[(f'x_{i}_{k}', f'x_{j}_{k+1}')] = graph[i][j]['weight']
                    objective[(f'x_{j}_{k}', f'x_{i}_{k+1}')] = graph[i][j]['weight']
    qp.minimize(quadratic=objective)

    qp.linear_constraint({'x_{}_{}'.format(start_node, 0): 1}, '==', 1, name="start_constraint")
    qp.linear_constraint({'x_{}_{}'.format(end_node, n_nodes - 1): 1}, '==', 1, name="end_constraint")
    for k in range(n_nodes):
        qp.linear_constraint({f'x_{i}_{k}': 1 for i in range(n_nodes)}, '==', 1, name=f"pos_constraint_{k}")
    for i in range(n_nodes):
        qp.linear_constraint({f'x_{i}_{k}': 1 for k in range(n_nodes)}, '<=', 1, name=f"node_constraint_{i}")
    print("Đã xây dựng Quadratic Program cho Shortest Path.")

    # 2. Chuyển QP sang Hamiltonian
    cost_h = qp_to_pennylane_hamiltonian(qp)
    print(f"Đã chuyển QP sang Hamiltonian với {cost_h.num_wires} qubits.")

    # 3. Thiết lập và chạy VQE
    dev = qml.device("default.qubit", wires=cost_h.num_wires)
    
    @qml.qnode(dev, interface="torch")
    def circuit(params):
        qml.templates.BasicEntanglerLayers(weights=params, wires=range(cost_h.num_wires))
        return qml.expval(cost_h)

    optimizer = torch.optim.Adam
    params_shape = qml.templates.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=cost_h.num_wires)
    params = torch.tensor(pnp.random.uniform(0, 2 * np.pi, size=params_shape), requires_grad=True)
    opt = optimizer([params], lr=learning_rate)

    print("Bắt đầu tối ưu hóa VQE cho Shortest Path...")
    for step in range(n_steps):
        opt.zero_grad()
        # Tính toán giá trị loss (có thể là số phức)
        loss_complex = circuit(params)
        # SỬA LỖI Ở ĐÂY: Chỉ lấy phần thực để tính đạo hàm
        loss_real = torch.real(loss_complex)
        loss_real.backward()
        
        opt.step()
        if (step + 1) % 20 == 0:
            print(f"  VQE Step {step+1:3d}: Cost (Energy) = {loss_real.item():.7f}")

    # 4. Diễn giải kết quả (giữ nguyên)
    @qml.qnode(dev)
    def get_probabilities(params):
        qml.templates.BasicEntanglerLayers(weights=params.detach(), wires=range(cost_h.num_wires))
        return qml.probs(wires=range(cost_h.num_wires))

    probs = get_probabilities(params)
    bit_string = format(pnp.argmax(probs), f'0{cost_h.num_wires}b')[::-1]

    path = []
    solution_matrix = np.zeros((n_nodes, n_nodes))
    for k in range(n_nodes):
        for i in range(n_nodes):
            var_index = qp.get_variable(f'x_{i}_{k}').index
            if bit_string[var_index] == '1':
                solution_matrix[i, k] = 1
    
    current_node = start_node
    path.append(current_node)
    for k in range(n_nodes - 1):
        found_next = False
        # Tìm node ở vị trí tiếp theo trong đường đi
        for next_node in range(n_nodes):
            if solution_matrix[next_node, k+1] == 1:
                # Không cần kiểm tra graph.has_edge ở đây
                path.append(next_node)
                current_node = next_node
                found_next = True
                break
        if not found_next:
            break
    
    # Lọc các node trùng lặp có thể xuất hiện do encoding
    final_path = list(dict.fromkeys(path))

    path_cost = 0
    valid_path = True
    if not final_path or final_path[0] != start_node or final_path[-1] != end_node:
        valid_path = False
    
    for i in range(len(final_path) - 1):
        u, v = final_path[i], final_path[i+1]
        if graph.has_edge(u, v):
            path_cost += graph[u][v]['weight']
        else: 
            valid_path = False
            break
            
    if not valid_path:
        path_cost = float('inf')

    return final_path, path_cost
