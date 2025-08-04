# file: algorithms/qaoa_solver.py (Phiên bản 4.0 - Dùng ApproxTimeEvolution)
import pennylane as qml
from pennylane import numpy as pnp
import torch

def solve_maxcut_qaoa(graph, n_layers=1, n_steps=100, learning_rate=0.1):
    """
    Giải bài toán Max-Cut trên một đồ thị cho trước bằng QAOA.
    Sử dụng qml.ApproxTimeEvolution để tương thích với các phiên bản PennyLane mới.
    """
    n_nodes = len(graph.nodes)
    
    # 1. Xây dựng Hamiltonian chi phí (Cost) và bộ trộn (Mixer)
    cost_h, mixer_h = qml.qaoa.maxcut(graph)
    
    # 2. Định nghĩa mạch QAOA (Ansatz tường minh)
    dev = qml.device("default.qubit", wires=n_nodes)
    
    @qml.qnode(dev, interface="torch")
    def circuit(params):
        # params sẽ có shape (2, n_layers).
        # params[0] là các góc gamma, params[1] là các góc beta.
        
        # Bắt đầu từ trạng thái chồng chập đều
        for i in range(n_nodes):
            qml.Hadamard(wires=i)
        
        # Áp dụng các lớp QAOA một cách tường minh
        for i in range(n_layers):
            # SỬA LỖI Ở ĐÂY: Dùng ApproxTimeEvolution thay cho QAOALayer
            qml.ApproxTimeEvolution(cost_h, params[0, i], 1) # Áp dụng e^(-i*gamma*H_cost)
            qml.ApproxTimeEvolution(mixer_h, params[1, i], 1) # Áp dụng e^(-i*beta*H_mixer)
        
        return qml.expval(cost_h)

    # 3. Vòng lặp tối ưu hóa
    optimizer = torch.optim.Adam
    params = torch.tensor(pnp.random.uniform(0, 2 * pnp.pi, (2, n_layers)), requires_grad=True)
    opt = optimizer([params], lr=learning_rate)

    print("Bắt đầu tối ưu hóa QAOA cho Max-Cut...")
    for step in range(n_steps):
        opt.zero_grad()
        loss = circuit(params)
        loss.backward()
        opt.step()
        if (step + 1) % 20 == 0:
            print(f"  QAOA Step {step+1:3d}: Cost (Energy) = {loss.item():.7f}")

    # 4. Diễn giải kết quả
    @qml.qnode(dev)
    def get_probabilities(params):
        for i in range(n_nodes):
            qml.Hadamard(wires=i)
        
        # Dùng detach() để PyTorch không theo dõi các phép tính này nữa
        detached_params = params.detach()
        for i in range(n_layers):
             # SỬA LỖI Ở ĐÂY: Dùng ApproxTimeEvolution thay cho QAOALayer
            qml.ApproxTimeEvolution(cost_h, detached_params[0, i], 1)
            qml.ApproxTimeEvolution(mixer_h, detached_params[1, i], 1)
            
        return qml.probs(wires=range(n_nodes))

    probs = get_probabilities(params)
    most_likely_outcome = pnp.argmax(probs)
    bit_string = format(most_likely_outcome, f'0{n_nodes}b')
    
    solution = [int(bit) for bit in bit_string]
    
    # Tính số cạnh cắt được
    cut_size = 0
    for i, j in graph.edges:
        if solution[i] != solution[j]:
            cut_size += 1
            
    return solution, cut_size
