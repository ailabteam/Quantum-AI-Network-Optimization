# file: data_generation/create_scenarios.py
import networkx as nx
import os

def create_and_save_scenario_4node():
    """
    Tạo và lưu đồ thị 4-node để test.
    Đường đi tối ưu từ 0 -> 3 là 0 -> 1 -> 2 -> 3 với chi phí 5+2+4=11.
    """
    G = nx.Graph()
    # Thêm các đỉnh
    G.add_nodes_from([0, 1, 2, 3])
    
    # Thêm các cạnh với trọng số
    G.add_edge(0, 1, weight=5)
    G.add_edge(0, 2, weight=8)
    G.add_edge(1, 2, weight=2)
    G.add_edge(1, 3, weight=7)
    G.add_edge(2, 3, weight=4)
    
    # Tạo thư mục scenarios nếu chưa có
    output_dir = '../scenarios'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Lưu đồ thị dưới định dạng GML (Graph Markup Language)
    output_path = os.path.join(output_dir, "scenario_4node.gml")
    nx.write_gml(G, output_path)
    print(f"Đã tạo và lưu đồ thị 4-node vào '{output_path}'")

if __name__ == "__main__":
    create_and_save_scenario_4node()
