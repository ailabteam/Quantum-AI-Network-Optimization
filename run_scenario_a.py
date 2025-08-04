# file: run_scenario_a.py (Phiên bản 3.0 - So sánh VQE và QAOA)
import networkx as nx
from itertools import product

# Sửa lại VQE solver để nó ổn định hơn
from algorithms.vqe_solver_stable import solve_maxcut_vqe 
from algorithms.qaoa_solver import solve_maxcut_qaoa

def get_classical_maxcut(graph):
    """Tính lời giải tối ưu cổ điển bằng brute-force."""
    max_cut = 0
    best_solution = None
    n_nodes = graph.number_of_nodes()
    for p in product([0, 1], repeat=n_nodes):
        current_cut = 0
        for i, j in graph.edges:
            if p[i] != p[j]:
                current_cut += 1
        if current_cut > max_cut:
            max_cut = current_cut
            best_solution = p
    return ''.join(map(str, best_solution)), max_cut

def main():
    print("--- Bắt đầu thực nghiệm cho Kịch bản A ---")
    
    # 1. Tải "dataset"
    try:
        graph = nx.read_gml("scenarios/scenario_a.gml", label='id')
        graph = nx.relabel_nodes(graph, {node: int(node) for node in graph.nodes()})
        print(f"Đã tải thành công đồ thị Kịch bản A với {graph.number_of_nodes()} đỉnh và {len(graph.edges)} cạnh.")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'scenarios/scenario_a.gml'.")
        return

    # --- Tính lời giải tối ưu để tham chiếu ---
    classical_sol, classical_max_cut = get_classical_maxcut(graph)
    print("\n--- Lời giải tối ưu cổ điển ---")
    print(f"Phân chia tối ưu: {classical_sol}")
    print(f"Số cạnh cắt được: {classical_max_cut}")
    print("-" * 35)

    # --- Thử nghiệm 1: VQE ---
    print("\n[Thử nghiệm 1: VQE cho bài toán Max-Cut]")
    vqe_solution, vqe_cost = solve_maxcut_vqe(graph, n_steps=200, n_layers=4, learning_rate=0.02)
    print("\n--- Kết quả VQE ---")
    print(f"Cách phân chia: {''.join(map(str, vqe_solution))}")
    print(f"Số cạnh cắt được: {vqe_cost:.4f}")
    print(f"Tỷ lệ xấp xỉ: {vqe_cost / classical_max_cut:.4f}")
    print("-" * 35)

    # --- Thử nghiệm 2: QAOA ---
    print("\n[Thử nghiệm 2: QAOA cho bài toán Max-Cut]")
    # QAOA thường cần ít lớp hơn nhưng nhiều bước hơn để hội tụ các góc
    qaoa_solution, qaoa_cost = solve_maxcut_qaoa(graph, n_steps=150, n_layers=3, learning_rate=0.05)
    print("\n--- Kết quả QAOA ---")
    print(f"Cách phân chia: {''.join(map(str, qaoa_solution))}")
    print(f"Số cạnh cắt được: {qaoa_cost:.4f}")
    print(f"Tỷ lệ xấp xỉ: {qaoa_cost / classical_max_cut:.4f}")
    print("-" * 35)

    print("\n--- Thực nghiệm Kịch bản A hoàn tất ---")

if __name__ == "__main__":
    main()
