# 🎨 Data Augmentation Web Application

> Công cụ web-based mạnh mẽ để tăng cường dữ liệu cho các tác vụ Object Detection, hỗ trợ 18+ kỹ thuật augmentation với giao diện trực quan.

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)

## ✨ Tính năng chính

### 🚀 Giao diện Web thân thiện
- **Upload dễ dàng**: Kéo thả hoặc chọn file ảnh và labels
- **Preview trực tiếp**: Xem trước kết quả augmentation với random sample
- **Lịch sử tác vụ**: Lưu trữ và quản lý tất cả các phiên augmentation
- **Download nhanh**: Tải về kết quả dưới dạng ZIP

### 🎯 18+ Kỹ thuật Augmentation
Hỗ trợ đầy đủ các kỹ thuật augmentation phổ biến:

#### Geometric Transformations
- **Horizontal Flip** - Lật ngang ảnh
- **Rotate** - Xoay ảnh với góc tùy chỉnh
- **Scale** - Thay đổi kích thước đối tượng
- **Translate** - Dịch chuyển vị trí
- **Shear** - Biến dạng nghiêng
- **Resize** - Thay đổi độ phân giải

#### Color Adjustments
- **Brightness** - Điều chỉnh độ sáng
- **Contrast** - Tăng/giảm độ tương phản
- **Saturation** - Thay đổi độ bão hòa màu
- **HSV** - Điều chỉnh Hue, Saturation, Value

#### Advanced Augmentations
- **Cutout** - Che ngẫu nhiên vùng ảnh
- **GridMask** - Tạo lưới che phủ
- **Mixup** - Trộn nhiều ảnh
- **Lighting Noise** - Thêm nhiễu ánh sáng
- **Noisy** - Thêm nhiễu Gaussian/Salt & Pepper
- **Filters** - Blur, Sharpen, Edge detection
- **Small Object Augmentation** - Tăng cường cho vật thể nhỏ
- **Rotate Only BBoxes** - Xoay chỉ bounding boxes

### 📦 Hỗ trợ nhiều định dạng
- **YOLO format** - `.txt` files với normalized coordinates
- **Pascal VOC format** - `.xml` files
- Tự động chuyển đổi giữa các định dạng
- Bảo toàn chính xác annotations sau augmentation

### 🔄 Workflow linh hoạt
1. Upload ảnh và labels
2. Chọn các augmentation muốn áp dụng
3. Preview kết quả với random sample
4. Apply cho toàn bộ dataset
5. Download hoặc tái sử dụng từ lịch sử

## 📋 Yêu cầu hệ thống

- **Docker & Docker Compose** (khuyến nghị)
- Hoặc **Python 3.8+** để chạy local

## 🚀 Cài đặt & Chạy

### Cách 1: Sử dụng Docker (Khuyến nghị)

```bash
# Clone repository
git clone <repository-url>
cd data-augmentation

# Khởi động container
docker-compose up -d

# Truy cập: http://localhost:2222
```

### Cách 2: Chạy Local

```bash
# Cài đặt dependencies
cd webapp
pip install -r requirements.txt

# Chạy ứng dụng
python app.py

# Truy cập: http://localhost:2222
```

## 🎮 Hướng dẫn sử dụng

### 1️⃣ Upload dữ liệu

**Cấu trúc thư mục input:**
```
dataset/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── labels/
    ├── img1.txt  (YOLO format)
    └── img2.txt
```

- Chọn **Images folder** và **Labels folder**
- Click **Upload** để tải lên
- Hệ thống tự động validate và tạo task

### 2️⃣ Chọn Augmentations

Tích chọn các augmentation muốn áp dụng từ danh sách:
- Mỗi augmentation có thể bật/tắt độc lập
- Hover để xem mô tả chi tiết
- Click **Preview** để xem mẫu trước khi apply

### 3️⃣ Preview & Apply

**Preview:**
- Chọn random 1 ảnh từ dataset
- Hiển thị ảnh gốc và ảnh đã augment
- Bounding boxes được vẽ trên cả hai

**Apply to All:**
- Click **Apply Augmentation** để xử lý toàn bộ
- Progress bar hiển thị tiến trình
- Kết quả lưu tự động

### 4️⃣ Download kết quả

**Cấu trúc output:**
```
output_{task_id}/
├── images/
│   ├── brightness_abc123_img1.jpg
│   ├── contrast_def456_img1.jpg
│   └── ...
└── labels/
    ├── brightness_abc123_img1.txt
    ├── contrast_def456_img1.txt
    └── ...
```

- Mỗi augmentation tạo 1 file mới với prefix
- Format labels giữ nguyên như input
- Click **Download** để tải ZIP file

### 5️⃣ Quản lý lịch sử

**History Tab:**
- Xem tất cả các task đã tạo
- Thông tin: Thời gian, số ảnh, augmentations
- Re-augment với cấu hình khác
- Delete task không cần thiết

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────┐
│          Web Browser (Port 2222)            │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Upload  │  │ Preview  │  │ History  │   │
│  └─────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│          Flask Application (app.py)         │
│  • REST API endpoints                       │
│  • File upload/download                     │
│  • Task management                          │
└─────────────────────────────────────────────┘
         ▼                    ▼
┌──────────────────┐  ┌──────────────────────┐
│ AugmentationSvc  │  │    Database (SQLite) │
│ • 18+ techniques │  │  • Tasks metadata    │
│ • Bbox transform │  │  • Config history    │
└──────────────────┘  └──────────────────────┘
```

## 📚 API Documentation

### Endpoints chính

#### `GET /api/augmentations`
Lấy danh sách các augmentation khả dụng
```json
[
  {
    "id": "brightness",
    "name": "Brightness Adjustment",
    "description": "Điều chỉnh độ sáng ảnh"
  },
  ...
]
```

#### `POST /api/upload`
Upload images và labels
```bash
POST /api/upload
Content-Type: multipart/form-data

images: [file1, file2, ...]
labels: [label1.txt, label2.txt, ...]
```

#### `POST /api/preview/{task_id}`
Tạo preview với random sample
```bash
POST /api/preview/{task_id}
Content-Type: application/json

{
  "augmentations": ["brightness", "horizontal_flip"]
}
```

#### `POST /api/augment/{task_id}`
Apply augmentation cho toàn bộ dataset
```bash
POST /api/augment/{task_id}
Content-Type: application/json

{
  "augmentations": ["brightness", "contrast", "cutout"]
}
```

#### `GET /api/download/{output_id}`
Download kết quả dưới dạng ZIP

#### `GET /api/tasks`
Lấy danh sách tất cả tasks

#### `DELETE /api/tasks/{task_id}`
Xóa task và dữ liệu liên quan

## ⚙️ Configuration

Mỗi augmentation có thể config trong file `augmentations/*.py`:

**Ví dụ Brightness:**
```python
def apply(image, bboxes, **params):
    factor = params.get('factor', 1.2)  # Default brightness factor
    adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return adjusted, bboxes
```

Tùy chỉnh parameters trong code hoặc qua API request.

## 📁 Cấu trúc Project

```
data-augmentation/
├── webapp/                    # Web application
│   ├── app.py                # Flask main app
│   ├── augmentation_service.py  # Core augmentation logic
│   ├── database.py           # SQLite database manager
│   ├── static/               # CSS, JS
│   ├── templates/            # HTML templates
│   ├── uploads/              # Uploaded files
│   └── outputs/              # Augmented results
├── augmentations/            # Augmentation modules
│   ├── brightness.py
│   ├── contrast.py
│   ├── horizontal_flip.py
│   └── ...                   # 18+ techniques
├── utils/                    # Utility functions
│   └── utils.py
├── assets/                   # Sample data
│   ├── yolo/
│   └── voc/
├── docker-compose.yml        # Docker configuration
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🔧 Development

### Thêm augmentation mới

1. Tạo file trong `augmentations/new_augmentation.py`:
```python
def apply(image, bboxes, **params):
    """
    Args:
        image: numpy array (H, W, C)
        bboxes: list of [class_id, x_center, y_center, width, height]
        params: custom parameters
    Returns:
        augmented_image, transformed_bboxes
    """
    # Your augmentation logic here
    return augmented_image, bboxes
```

2. Augmentation tự động được detect và thêm vào danh sách

### Testing

```bash
# Run tests
python -m pytest tests/

# Test specific augmentation
python -m pytest tests/test_brightness.py
```

## 🐛 Troubleshooting

**Lỗi upload file:**
- Kiểm tra file size < 100MB
- Đảm bảo format đúng (.jpg, .png, .txt, .xml)

**Preview không hiển thị:**
- Kiểm tra labels có match với images
- Format annotations đúng (YOLO/VOC)

**Bounding boxes bị sai:**
- Verify input labels format
- Check coordinates trong range [0, 1] cho YOLO

**Container không start:**
```bash
# Check logs
docker logs data-augmentation-webapp

# Rebuild
docker-compose down
docker-compose up -d --build
```

## 📊 Performance

- **Tốc độ**: ~10-50 images/second (tùy augmentation)
- **Memory**: ~500MB-2GB RAM
- **Storage**: Tăng ~2-20x tùy số augmentations
- **Batch processing**: Hỗ trợ dataset lớn với progress tracking

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - xem file [LICENSE](LICENSE) để biết chi tiết.

## 👨‍💻 Author

**quanglv1996**
- GitHub: [@quanglv1996](https://github.com/quanglv1996)

## 🙏 Acknowledgments

- OpenCV for image processing
- Flask for web framework
- YOLOv8/v11 format support
- Community contributions

## 📮 Support

Nếu gặp vấn đề hoặc có câu hỏi:
- 🐛 Issue Tracker
- 💬 Thảo luận hoặc góp ý

---

⭐ **Nếu thấy hữu ích, hãy star repo này!**

