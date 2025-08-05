# file: run_scenario_a.py (So sánh VQE và QAOA cho Shortest Path)
import networkx as nx

# Import cả hai solver
from algorithms.pennylane_sp_solver import solve_shortest_path_pennylane_vqe
from algorithms.pennylane_sp_qaoa_solver import solve_shortest_path_pennylane_qaoa

def get_classical_shortest_path(graph, start, end):
    try:
        path = nx.shortest_path(graph, source=start, target=end, weight='weight')
        length = nx.shortest_path_length(graph, source=start, target=end, weight='weight')
        return path, length
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, float('inf')

def main():
    print("--- Bắt đầu thực nghiệm trên đồ thị 4-node (16 qubits) ---")
    
    graph = nx.read_gml("scenarios/scenario_4node.gml", label='id')
    graph = nx.relabel_nodes(graph, {node: int(node) for node in graph.nodes()})
    print(f"Đã tải thành công đồ thị 4-node với {graph.number_of_nodes()} đỉnh.")

    start_node, end_node = 0, 3
    
    classical_path, classical_cost = get_classical_shortest_path(graph, start_node, end_node)
    print("\n--- Lời giải tối ưu cổ điển (Shortest Path) ---")
    if classical_path:
        print(f"Đường đi: {' -> '.join(map(str, classical_path))}")
        print(f"Tổng chi phí: {classical_cost}")
    print("-" * 45)

    # --- Thử nghiệm 1: VQE ---
    print("\n[Thử nghiệm VQE cho Shortest Path]")
    vqe_path, vqe_cost = solve_shortest_path_pennylane_vqe(
        graph, start_node, end_node,
        n_steps=400, learning_rate=0.08, n_layers=4
    )
    print("\n--- Kết quả VQE ---")
    print(f"Đường đi tìm thấy: {' -> '.join(map(str, vqe_path))}")
    print(f"Tổng chi phí: {vqe_cost:.4f}")
    if classical_cost > 0 and vqe_cost != float('inf'):
        print(f"Tỷ lệ xấp xỉ: {vqe_cost / classical_cost:.4f}")
    print("-" * 45)

    # --- Thử nghiệm 2: QAOA ---
    print("\n[Thử nghiệm QAOA cho Shortest Path]")
    qaoa_path, qaoa_cost = solve_shortest_path_pennylane_qaoa(
        graph, start_node, end_node,
        n_layers=4, n_steps=300, learning_rate=0.1
    )
    print("\n--- Kết quả QAOA ---")
    print(f"Đường đi tìm thấy: {' -> '.join(map(str, qaoa_path))}")
    print(f"Tổng chi phí: {qaoa_cost:.4f}")
    if classical_cost > 0 and qaoa_cost != float('inf'):
        print(f"Tỷ lệ xấp xỉ: {qaoa_cost / classical_cost:.4f}")
    print("-" * 45)


    print("\n--- Thực nghiệm hoàn tất ---")

if __name__ == "__main__":
    main()
