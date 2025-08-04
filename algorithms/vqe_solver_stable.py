# file: algorithms/vqe_solver.py (Phiên bản ổn định 2.0)
import pennylane as qml
from pennylane import numpy as pnp
import torch

def build_maxcut_hamiltonian(graph):
    """
    Tự xây dựng Hamiltonian cho bài toán Max-Cut một cách tường minh.
    """
    n_nodes = len(graph.nodes)
    
    obs = []
    coeffs = []
    
    # Hằng số: 0.5 * tổng số cạnh
    # Trong các phiên bản mới, không cần thêm hằng số vào Hamiltonian
    # mà có thể cộng trực tiếp vào cost function.
    # Nhưng để đơn giản, chúng ta giữ nguyên cách này.
    coeffs.append(0.5 * len(graph.edges))
    obs.append(qml.Identity(0))
    
    # Phần tương tác: -0.5 * Z_i * Z_j
    for i, j in graph.edges:
        obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
        coeffs.append(-0.5)

    hamiltonian = qml.Hamiltonian(coeffs, obs)
    return hamiltonian

def solve_maxcut_vqe(graph, n_steps=100, learning_rate=0.1, n_layers=2):
    """
    Giải bài toán Max-Cut trên một đồ thị cho trước bằng VQE.
    """
    n_nodes = len(graph.nodes)
    
    # 1. Tự xây dựng Hamiltonian cho Max-Cut
    cost_h = build_maxcut_hamiltonian(graph)
    print("Đã tự xây dựng Hamiltonian cho Max-Cut thành công.")
    
    # 2. Thiết lập VQE
    dev = qml.device("default.qubit", wires=n_nodes)
    
    @qml.qnode(dev, interface="torch")
    def cost_function(params):
        # SỬA LỖI Ở ĐÂY: Dùng một ansatz ổn định hơn
        qml.templates.BasicEntanglerLayers(weights=params, wires=range(n_nodes))
        return qml.expval(cost_h)

    # 3. Vòng lặp tối ưu hóa
    optimizer = torch.optim.Adam
    # SỬA LỖI Ở ĐÂY: Lấy shape từ ansatz mới
    params_shape = qml.templates.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_nodes)
    params = torch.tensor(pnp.random.uniform(0, 2 * pnp.pi, size=params_shape), requires_grad=True)
    opt = optimizer([params], lr=learning_rate)

    print("Bắt đầu tối ưu hóa VQE cho Max-Cut...")
    for step in range(n_steps):
        opt.zero_grad()
        loss = cost_function(params)
        loss.backward()
        opt.step()
        if (step + 1) % 20 == 0:
            print(f"  VQE Step {step+1:3d}: Cost (Energy) = {loss.item():.7f}")
    
    # 4. Diễn giải kết quả
    @qml.qnode(dev, interface="torch")
    def get_probabilities(params):
        # SỬA LỖI Ở ĐÂY: Dùng ansatz mới
        qml.templates.BasicEntanglerLayers(weights=params, wires=range(n_nodes))
        return qml.probs(wires=range(n_nodes))

    probs = get_probabilities(params).detach().numpy()
    most_likely_outcome = pnp.argmax(probs)
    bit_string = format(most_likely_outcome, f'0{n_nodes}b')
    
    solution = [int(bit) for bit in bit_string]
    
    # Tính số cạnh cắt được từ chuỗi bit kết quả
    cut_size = 0
    for i, j in graph.edges:
        if solution[i] != solution[j]:
            cut_size += 1
    
    return solution, cut_size
