# file: algorithms/qiskit_vqe_solver.py (Sử dụng hoàn toàn hệ sinh thái Qiskit)
import networkx as nx
import numpy as np

# Import các công cụ từ Qiskit
from qiskit_optimization.applications import Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler

# Callback để theo dõi quá trình tối ưu hóa
def vqe_callback(eval_count, parameters, mean, std):
    if eval_count % 20 == 0:
        print(f"  VQE Eval Cnt {eval_count:4d}: Cost (Energy) = {mean:.7f}")


def solve_shortest_path_qiskit_vqe(graph, start_node, end_node, maxiter=300):
    """
    Giải bài toán Shortest Path bằng VQE của Qiskit.
    """
    n_nodes = graph.number_of_nodes()

    # 1. Tạo ma trận khoảng cách cho TSP
    dist_matrix = nx.to_numpy_array(graph, nonedge=9999, weight='weight')
    # Mẹo: Ép TSP tìm đường đi s->d
    dist_matrix[end_node, start_node] = -n_nodes * np.max(dist_matrix[dist_matrix != 9999])
    
    # 2. Tạo đối tượng TSP và Quadratic Program
    tsp_problem = Tsp(dist_matrix)
    qp = tsp_problem.to_quadratic_program()
    num_vars = qp.get_num_vars()
    print(f"Đã xây dựng QP cho TSP với {num_vars} biến (qubits).")

    # 3. Chuyển QP sang Ising Hamiltonian
    qubo = QuadraticProgramToQubo().convert(qp)
    hamiltonian, offset = qubo.to_ising()
    print(f"Đã chuyển QP sang Hamiltonian với {hamiltonian.num_qubits} qubits.")

    # 4. Thiết lập và chạy VQE của Qiskit
    # Do số lượng qubit lớn, chúng ta cần một ansatz và optimizer hiệu quả
    optimizer = SPSA(maxiter=maxiter)
    ansatz = TwoLocal(hamiltonian.num_qubits, "ry", "cz", reps=2, entanglement="linear")
    sampler = Sampler() # Dùng simulator mặc định

    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=optimizer, callback=vqe_callback)
    
    print("Bắt đầu tối ưu hóa VQE của Qiskit...")
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    
    # 5. Diễn giải kết quả
    # Lấy chuỗi bit từ kết quả
    eigenstate_result = result.optimal_parameters
    # Qiskit's VQE không trực tiếp trả về chuỗi bit, chúng ta cần chạy lại mạch để lấy
    # Để đơn giản, chúng ta sẽ dựa vào hàm interpret của TSP
    # LƯU Ý: Đây là một cách diễn giải gần đúng
    interpreted_result = tsp_problem.interpret(result.eigenstate)
    
    path = interpreted_result
    
    # Tính chi phí
    path_cost = 0
    if path and path[0] == start_node and path[-1] == end_node:
         # Logic tính cost...
         # Tạm thời để đơn giản
         path_cost = tsp_problem.get_graph_solution_cost(path)
         # Trừ đi trọng số âm của cạnh ảo
         path_cost -= dist_matrix[end_node, start_node]
    else:
         path_cost = float('inf')

    return path, path_cost
