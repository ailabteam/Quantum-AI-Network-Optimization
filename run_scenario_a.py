# file: run_scenario_a.py (Phiên bản 8.0 - PennyLane-only cho 4-node)
import networkx as nx

# Import solver mới hoàn toàn bằng PennyLane
from algorithms.pennylane_sp_solver import solve_shortest_path_pennylane_vqe

def get_classical_shortest_path(graph, start, end):
    """Tính lời giải tối ưu cổ điển cho Shortest Path."""
    try:
        path = nx.shortest_path(graph, source=start, target=end, weight='weight')
        length = nx.shortest_path_length(graph, source=start, target=end, weight='weight')
        return path, length
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, float('inf')

def main():
    print("--- Bắt đầu thực nghiệm trên đồ thị 4-node (16 qubits) ---")
    
    # 1. Tải "dataset" 4-node
    try:
        graph = nx.read_gml("scenarios/scenario_4node.gml", label='id')
        # Đảm bảo các node là số nguyên để làm index
        graph = nx.relabel_nodes(graph, {node: int(node) for node in graph.nodes()})
        print(f"Đã tải thành công đồ thị 4-node với {graph.number_of_nodes()} đỉnh.")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'scenarios/scenario_4node.gml'.")
        print("Vui lòng chạy 'python data_generation/create_scenarios.py' trước.")
        return

    # --- Định nghĩa bài toán Shortest Path ---
    start_node, end_node = 0, 3
    
    # --- Lời giải tham chiếu ---
    classical_path, classical_cost = get_classical_shortest_path(graph, start_node, end_node)
    print("\n--- Lời giải tối ưu cổ điển (Shortest Path) ---")
    if classical_path:
        print(f"Đường đi: {' -> '.join(map(str, classical_path))}")
        print(f"Tổng chi phí: {classical_cost}")
    else:
        print("Không có đường đi.")
    print("-" * 45)

    # --- Thử nghiệm VQE với PennyLane ---
    print("\n[Thử nghiệm VQE cho Shortest Path (PennyLane-only)]")
    # Điều chỉnh tham số để chạy nhanh hơn và có cơ hội hội tụ tốt
    vqe_path, vqe_cost = solve_shortest_path_pennylane_vqe(
        graph, 
        start_node, 
        end_node,
        n_steps=200, 
        learning_rate=0.1,
        n_layers=3
    )
    print("\n--- Kết quả VQE ---")
    print(f"Đường đi tìm thấy: {' -> '.join(map(str, vqe_path))}")
    print(f"Tổng chi phí: {vqe_cost:.4f}")
    
    if classical_cost > 0 and vqe_cost != float('inf'):
        approximation_ratio = vqe_cost / classical_cost
        print(f"Tỷ lệ xấp xỉ: {approximation_ratio:.4f}")

    print("-" * 45)
    print("\n--- Thực nghiệm hoàn tất ---")

if __name__ == "__main__":
    main()
