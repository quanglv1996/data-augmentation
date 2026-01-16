# Data Augmentation Web Application

Ứng dụng web để tăng cường dữ liệu hình ảnh cho Object Detection với giao diện đồ họa thân thiện.

## Tính năng

✨ **Các tính năng chính:**

1. **Tải ảnh và nhãn lên**
   - Hỗ trợ nhiều định dạng: PNG, JPG, JPEG, BMP
   - Hỗ trợ nhãn YOLO (.txt) và Pascal VOC (.xml)
   - Upload nhiều file cùng lúc

2. **Chọn phương pháp Augmentation**
   - 12+ phương pháp augmentation có sẵn
   - Brightness, Contrast, Saturation
   - Horizontal Flip, Rotate, Scale, Shear, Translate
   - Cutout, Grid Mask, Noise, HSV
   - Chọn nhiều augmentation để áp dụng cùng lúc

3. **Xem trước kết quả**
   - Lấy ngẫu nhiên 1 mẫu từ dataset
   - Hiển thị ảnh gốc và ảnh sau augmentation
   - Hiển thị bounding boxes trên cả 2 ảnh

4. **Áp dụng Augmentation**
   - Xử lý toàn bộ dataset
   - Lưu ảnh và nhãn sau augmentation
   - Tải về kết quả dưới dạng file ZIP

5. **Quản lý lịch sử**
   - Xem lại các task đã upload
   - Lịch sử các lần augmentation
   - Re-augment với các tùy chọn khác trên cùng dataset
   - Xóa task không cần thiết

## Cài đặt và chạy

### Sử dụng Docker (Khuyến nghị)

1. **Build và chạy container:**
```bash
docker-compose up -d --build
```

2. **Truy cập ứng dụng:**
Mở trình duyệt và truy cập: `http://localhost:222`

3. **Dừng ứng dụng:**
```bash
docker-compose down
```

### Chạy trực tiếp (Không dùng Docker)

1. **Cài đặt dependencies:**
```bash
cd webapp
pip install -r requirements.txt
```

2. **Chạy ứng dụng:**
```bash
python app.py
```

3. **Truy cập ứng dụng:**
Mở trình duyệt và truy cập: `http://localhost:222`

## Hướng dẫn sử dụng

### 1. Tải dữ liệu lên

1. Nhập tên task (tùy chọn)
2. Chọn định dạng nhãn (YOLO hoặc Pascal VOC)
3. Chọn các file ảnh
4. Chọn các file nhãn tương ứng
5. Click "Tải lên"

### 2. Chọn Augmentation

Sau khi tải lên thành công, chọn các phương pháp augmentation muốn áp dụng bằng cách tích vào checkbox.

### 3. Xem trước

Click "Xem trước" để xem kết quả trên 1 mẫu ngẫu nhiên. Điều này giúp bạn đánh giá xem các augmentation đã chọn có phù hợp không.

### 4. Áp dụng Augmentation

Nếu hài lòng với kết quả preview, click "Áp dụng Augmentation" để xử lý toàn bộ dataset.

### 5. Tải về kết quả

Sau khi xử lý xong, click "Tải về kết quả" để download file ZIP chứa ảnh và nhãn đã được augment.

### 6. Quản lý lịch sử

Chuyển sang tab "Lịch sử" để:
- Xem tất cả các task đã upload
- Xem lịch sử augmentation của mỗi task
- Download lại kết quả các lần augmentation trước
- Re-augment với các tùy chọn khác
- Xóa task không cần thiết

## Cấu trúc thư mục

```
data-augmentation/
├── webapp/
│   ├── app.py                      # Flask application
│   ├── augmentation_service.py     # Augmentation logic
│   ├── database.py                 # Database management
│   ├── requirements.txt            # Python dependencies
│   ├── templates/
│   │   └── index.html             # Main HTML page
│   ├── static/
│   │   ├── style.css              # Styles
│   │   └── script.js              # JavaScript
│   ├── uploads/                    # Uploaded files
│   ├── outputs/                    # Generated results
│   └── tasks.db                    # SQLite database
├── augmentations/                  # Augmentation modules
├── utils/                          # Utility functions
├── Dockerfile                      # Docker configuration
└── docker-compose.yml             # Docker Compose configuration
```

## API Endpoints

- `GET /` - Trang chủ
- `GET /api/augmentations` - Lấy danh sách augmentations
- `POST /api/upload` - Upload ảnh và nhãn
- `POST /api/preview/<task_id>` - Tạo preview
- `POST /api/augment/<task_id>` - Áp dụng augmentation
- `GET /api/download/<output_id>` - Download kết quả
- `GET /api/tasks` - Lấy danh sách tasks
- `DELETE /api/tasks/<task_id>` - Xóa task

## Yêu cầu hệ thống

- Python 3.9+
- Docker và Docker Compose (nếu sử dụng Docker)
- 4GB RAM tối thiểu
- 10GB dung lượng ổ cứng trống

## Lưu ý

- Port mặc định: **222**
- Kích thước file upload tối đa: 100MB
- Database: SQLite
- Dữ liệu được lưu tại thư mục `uploads` và `outputs`

## Troubleshooting

### Lỗi khi build Docker
```bash
docker-compose down
docker system prune -a
docker-compose up -d --build
```

### Port 222 đã được sử dụng
Sửa file `docker-compose.yml`, thay đổi port mapping:
```yaml
ports:
  - "8080:222"  # Thay 222 thành port khác
```

## License

MIT License - Xem file LICENSE để biết thêm chi tiết.
