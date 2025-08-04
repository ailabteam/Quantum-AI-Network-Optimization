# file: data_generation/create_scenarios.py
import networkx as nx
import os

def create_and_save_scenario_a():
    """Tạo và lưu đồ thị cho Kịch bản A."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=4)
    G.add_edge(0, 2, weight=10)
    G.add_edge(1, 2, weight=5)
    G.add_edge(1, 3, weight=12)
    G.add_edge(2, 3, weight=3)
    G.add_edge(2, 4, weight=15)
    G.add_edge(3, 4, weight=4)
    
    # Tạo thư mục scenarios nếu chưa có
    if not os.path.exists('../scenarios'):
        os.makedirs('../scenarios')
        
    # Lưu đồ thị dưới định dạng GML (Graph Markup Language)
    nx.write_gml(G, "../scenarios/scenario_a.gml")
    print("Đã tạo và lưu đồ thị Kịch bản A vào 'scenarios/scenario_a.gml'")

if __name__ == "__main__":
    create_and_save_scenario_a()
