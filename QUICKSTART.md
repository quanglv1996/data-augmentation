# HƯỚNG DẪN NHANH - Data Augmentation Web App

## Khởi động ứng dụng

### Cách 1: Sử dụng Docker (Khuyến nghị)

1. **Khởi động:**
   - Windows: Double-click file `start.bat`
   - Hoặc chạy lệnh: `docker-compose up -d --build`

2. **Truy cập:**
   - Mở trình duyệt: http://localhost:222

3. **Dừng:**
   ```bash
   docker-compose down
   ```

### Cách 2: Chạy trực tiếp Python

```bash
cd webapp
pip install -r requirements.txt
python app.py
```

## Sử dụng nhanh

### Bước 1: Upload dữ liệu
1. Nhập tên task (hoặc để trống)
2. Chọn định dạng nhãn: YOLO hoặc VOC
3. Chọn file ảnh (có thể chọn nhiều)
4. Chọn file nhãn tương ứng
5. Click "📤 Tải lên"

### Bước 2: Chọn Augmentation
Tích chọn các phương pháp augmentation:
- ✅ Brightness - Điều chỉnh độ sáng
- ✅ Contrast - Điều chỉnh độ tương phản
- ✅ Horizontal Flip - Lật ngang
- ✅ Rotate - Xoay ảnh
- ✅ Scale - Thay đổi tỷ lệ
- ... và nhiều hơn nữa

### Bước 3: Xem trước (Tùy chọn)
Click "👁️ Xem trước" để xem kết quả trên 1 ảnh mẫu

### Bước 4: Áp dụng
Click "✨ Áp dụng Augmentation" để xử lý toàn bộ dataset

### Bước 5: Tải về
Click "⬇️ Tải về kết quả" để download file ZIP

## Quản lý lịch sử

Chuyển sang tab "📜 Lịch sử" để:
- ✅ Xem tất cả các task đã upload
- ✅ Download lại kết quả cũ
- ✅ Re-augment với các tùy chọn khác
- ✅ Xóa task không cần

## Port

Ứng dụng chạy trên port: **222**

Truy cập: http://localhost:222

## Troubleshooting

### Port 222 đã được sử dụng?
Sửa file `docker-compose.yml`:
```yaml
ports:
  - "8080:222"  # Thay 222 thành port khác
```

### Lỗi Docker?
```bash
docker-compose down
docker system prune -a
docker-compose up -d --build
```

### Xem logs?
```bash
docker-compose logs -f
```

## Hỗ trợ

- Xem hướng dẫn chi tiết: [webapp/README.md](webapp/README.md)
- Kiểm tra logs nếu có lỗi
- Đảm bảo Docker đang chạy (nếu dùng Docker)
