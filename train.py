import os
import torch
from ultralytics import YOLO

def check_dataset_structure():
    """Verify that the dataset folders exist before training."""
    required_paths = [
        "dataset/images/train/images",
        "dataset/images/train/labels"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
            
    if missing_paths:
        print("❌ CẢNH BÁO: Không tìm thấy các thư mục dữ liệu sau:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\n👉 Vui lòng tải tập dữ liệu (Dataset) từ Roboflow hoặc gán nhãn tự tạo")
        print("và sắp xếp vào thư mục 'dataset' theo cấu trúc trên trước khi chạy huấn luyện.")
        return False
    return True

def train_model():
    print("="*60)
    print("🔥 KHỞI ĐỘNG TIẾN TRÌNH HUẤN LUYỆN YOLOv8 NHẬN DIỆN CHÁY 🔥")
    print("="*60)
    
    # 1. Check dataset
    if not check_dataset_structure():
        print("\nĐể chạy thử nghiệm giao diện trước khi train, bạn có thể chạy file 'app.py' ngay lập tức.")
        print("Hệ thống sẽ tự động dùng mô hình mặc định 'yolov8n.pt' ở chế độ mô phỏng.")
        return

    # 2. Check hardware (GPU CUDA / CPU)
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"🍀 Thiết bị sử dụng huấn luyện: {device.upper()}")
    if device == '0':
        print(f"  -> Phát hiện GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  -> Chạy trên CPU (Sẽ chậm hơn rất nhiều so với chạy trên GPU Nvidia)")

    # 3. Load YOLO model
    # We use yolov8n.pt (Nano) as the base pretrained weight - it's fast, lightweight, and accurate
    print("\n⏳ Đang tải mô hình YOLOv8n nền...")
    model = YOLO("yolov8n.pt")

    # 4. Run training
    # Parameters:
    #   - data: path to dataset config file (data.yaml)
    #   - epochs: number of training epochs (default 50 for personal project testing)
    #   - imgsz: image size (640 is standard)
    #   - device: device to train on ('0' for GPU, 'cpu' for CPU)
    #   - workers: dataloader workers (set to 2 or 4 for faster loading)
    print("\n🚀 Bắt đầu huấn luyện...")
    try:
        model.train(
            data="data.yaml",
            epochs=50,
            imgsz=640,
            device=device,
            workers=2,
            project="runs/detect",
            name="train_fire"
        )
        
        print("\n" + "="*60)
        print("🎉 HUẤN LUYỆN THÀNH CÔNG! 🎉")
        print("="*60)
        
        # Guide user to use the trained weights
        best_weights = os.path.join("runs", "detect", "train_fire", "weights", "best.pt")
        if os.path.exists(best_weights):
            print(f"Trọng số mô hình tốt nhất của bạn được lưu tại: {best_weights}")
            print("\n👉 CÁCH SỬ DỤNG:")
            print(f"1. Copy file 'best.pt' trong '{best_weights}' ra ngoài thư mục gốc dự án.")
            print("2. Chạy 'python app.py' để ứng dụng tự động nhận diện bằng mô hình mới của bạn!")
        else:
            print("Đã hoàn thành huấn luyện. Hãy tìm file 'best.pt' trong thư mục 'runs/detect/' của bạn.")
            
    except Exception as e:
        print(f"\n❌ Đã xảy ra lỗi trong quá trình huấn luyện: {e}")
        print("Vui lòng kiểm tra lại cấu hình bộ nhớ RAM/VRAM hoặc định dạng nhãn của dataset.")

if __name__ == "__main__":
    train_model()
