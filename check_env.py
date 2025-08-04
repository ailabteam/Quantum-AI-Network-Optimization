# file: check_env.py
# Mục đích: Kiểm tra toàn bộ môi trường cài đặt cho paper.

# --- Import các thư viện ---
try:
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import pennylane as qml
    from pennylane import numpy as pnp
    import torch
    
    print("SUCCESS: Tất cả các thư viện đã được import thành công.\n")
except ImportError as e:
    print(f"ERROR: Không thể import thư viện. Lỗi: {e}")
    print("Vui lòng kiểm tra lại quá trình cài đặt.")
    exit()

# --- Các hàm kiểm tra ---

def test_classical_libs():
    """Kiểm tra các thư viện cổ điển: numpy, networkx, matplotlib."""
    print("--- 1. Bắt đầu kiểm tra các thư viện Cổ điển ---")
    try:
        # Tạo đồ thị đơn giản
        G = nx.Graph()
        G.add_edge('A', 'B', weight=4)
        G.add_edge('B', 'C', weight=2)
        G.add_edge('A', 'C', weight=7)
        print("NetworkX: Đã tạo đồ thị thành công.")
        
        # Vẽ và lưu đồ thị ra file
        plt.figure()
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='skyblue')
        plt.savefig("test_classical.png")
        plt.close() # Đóng plot để script không bị treo
        print("Matplotlib: Đã vẽ và lưu đồ thị ra file 'test_classical.png'.")
        
        print("=> SUCCESS: Các thư viện Cổ điển hoạt động tốt.\n")
        return True
    
    except Exception as e:
        print(f"ERROR trong khi test thư viện Cổ điển: {e}")
        return False

def test_quantum_libs():
    """Kiểm tra các thư viện lượng tử: pennylane, torch."""
    print("--- 2. Bắt đầu kiểm tra các thư viện Lượng tử (PennyLane + PyTorch) ---")
    try:
        # Chọn thiết bị
        dev = qml.device('default.qubit', wires=2)
        print("PennyLane: Đã chọn thiết bị 'default.qubit'.")

        # Định nghĩa một mạch lượng tử (QNode)
        @qml.qnode(dev, interface='torch')
        def simple_quantum_circuit(phi, theta):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(phi, wires=0)
            qml.RY(theta, wires=1)
            return qml.expval(qml.PauliZ(1))
        
        print("PennyLane: Đã định nghĩa QNode thành công.")

        # Tạo tham số bằng PyTorch
        phi = torch.tensor(0.5, requires_grad=True)
        theta = torch.tensor(0.1, requires_grad=True)
        print("PyTorch: Đã tạo tham số (tensors) thành công.")

        # Chạy mạch
        result = simple_quantum_circuit(phi, theta)
        print(f"PennyLane: Chạy mạch thành công, kết quả: {result.item():.4f}")

        # Tính đạo hàm
        result.backward()
        print("PyTorch Autograd: Đã tính toán đạo hàm thành công.")
        print(f"  - Gradient của phi: {phi.grad.item():.4f}")
        
        print("=> SUCCESS: PennyLane và PyTorch được tích hợp và hoạt động tốt.\n")
        return True

    except Exception as e:
        print(f"ERROR trong khi test thư viện Lượng tử: {e}")
        return False

# --- Hàm chính để chạy tất cả ---

def main():
    """Hàm chính điều phối các bài test."""
    print("=============================================")
    print(" BẮT ĐẦU KIỂM TRA MÔI TRƯỜNG NGHIÊN CỨU ")
    print("=============================================\n")

    classical_ok = test_classical_libs()
    quantum_ok = test_quantum_libs()
    
    print("--- TỔNG KẾT ---")
    if classical_ok and quantum_ok:
        print("=> CHÚC MỪNG! Môi trường của bạn đã sẵn sàng cho dự án.")
        print("   Bạn có thể xem file 'test_classical.png' để kiểm tra hình ảnh.")
    else:
        print("=> CÓ LỖI XẢY RA. Vui lòng xem lại các thông báo lỗi ở trên.")
    
    print("=============================================")

if __name__ == "__main__":
    main()
