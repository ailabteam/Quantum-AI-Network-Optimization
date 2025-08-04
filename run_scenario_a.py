# file: run_scenario_a.py (Phiên bản 4.0 - Thêm Shortest Path)
import networkx as nx

from algorithms.shortest_path_vqe_solver import solve_shortest_path_vqe

def get_classical_shortest_path(graph, start, end):
    """Tính lời giải tối ưu cổ điển cho Shortest Path."""
    try:
        path = nx.shortest_path(graph, source=start, target=end, weight='weight')
        length = nx.shortest_path_length(graph, source=start, target=end, weight='weight')
        return path, length
    except nx.NetworkXNoPath:
        return None, float('inf')

def main():
    print("--- Bắt đầu thực nghiệm cho Kịch bản A ---")
    
    # 1. Tải "dataset"
    try:
        graph = nx.read_gml("scenarios/scenario_a.gml", label='id')
        graph = nx.relabel_nodes(graph, {node: int(node) for node in graph.nodes()})
        print(f"Đã tải thành công đồ thị Kịch bản A với {graph.number_of_nodes()} đỉnh.")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'scenarios/scenario_a.gml'.")
        return

    # --- Định nghĩa bài toán Shortest Path ---
    start_node, end_node = 0, 4
    
    # --- Lời giải tham chiếu ---
    classical_path, classical_cost = get_classical_shortest_path(graph, start_node, end_node)
    print("\n--- Lời giải tối ưu cổ điển (Shortest Path) ---")
    print(f"Đường đi: {' -> '.join(map(str, classical_path))}")
    print(f"Tổng chi phí: {classical_cost}")
    print("-" * 45)

    # --- Thử nghiệm 3: VQE cho bài toán Shortest Path ---
    print("\n[Thử nghiệm 3: VQE cho bài toán Shortest Path]")
    vqe_path, vqe_cost = solve_shortest_path_vqe(
        graph, 
        start_node, 
        end_node,
        n_steps=300, 
        learning_rate=0.05,
        n_layers=4
    )
    print("\n--- Kết quả VQE (Shortest Path) ---")
    print(f"Đường đi tìm thấy: {' -> '.join(map(str, vqe_path))}")
    print(f"Tổng chi phí: {vqe_cost:.4f}")
    
    if classical_cost > 0:
        approximation_ratio = vqe_cost / classical_cost
        print(f"Tỷ lệ xấp xỉ: {approximation_ratio:.4f}")

    print("-" * 45)
    print("\n--- Thực nghiệm Kịch bản A hoàn tất ---")

if __name__ == "__main__":
    main()
