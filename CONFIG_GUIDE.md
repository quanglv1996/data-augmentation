# Configuration Guide - Data Augmentation Web App

## Augmentation Parameters

Bạn có thể tùy chỉnh các tham số của augmentation trong file `webapp/augmentation_service.py`

### Brightness
```python
'brightness': {
    'class': AdjustBrightness, 
    'params': {
        'brightness_min': 0.8,    # Minimum brightness factor (0.5-2.0)
        'brightness_max': 1.2     # Maximum brightness factor (0.5-2.0)
    }
}
```

### Contrast
```python
'contrast': {
    'class': AdjustContrast, 
    'params': {
        'contrast_min': 0.8,      # Minimum contrast factor
        'contrast_max': 1.2       # Maximum contrast factor
    }
}
```

### Saturation
```python
'saturation': {
    'class': AdjustSaturation, 
    'params': {
        'saturation_min': 0.8,    # Minimum saturation factor
        'saturation_max': 1.2     # Maximum saturation factor
    }
}
```

### Rotate
```python
'rotate': {
    'class': Rotate, 
    'params': {
        'angle': 15               # Maximum rotation angle in degrees
    }
}
```

### Scale
```python
'scale': {
    'class': Scale, 
    'params': {
        'scale_x': 0.2,           # Scaling factor for width
        'scale_y': 0.2            # Scaling factor for height
    }
}
```

### Shear
```python
'shear': {
    'class': Shear, 
    'params': {
        'shear_factor': 0.2       # Shear transformation factor
    }
}
```

### Translate
```python
'translate': {
    'class': Translate, 
    'params': {
        'translate_x': 0.2,       # Translation ratio for x-axis
        'translate_y': 0.2        # Translation ratio for y-axis
    }
}
```

### Cutout
```python
'cutout': {
    'class': Cutout, 
    'params': {
        'n_holes': 3,             # Number of cutout regions
        'length': 50              # Size of each cutout square
    }
}
```

## Server Configuration

### Change Port

**File: `webapp/app.py`**
```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=222, debug=True)  # Change 222 to your port
```

**File: `docker-compose.yml`**
```yaml
ports:
  - "222:222"  # Change first 222 to your host port
```

### Upload Size Limit

**File: `webapp/app.py`**
```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB (change as needed)
```

### Database Path

**File: `webapp/app.py`**
```python
app.config['DATABASE'] = 'tasks.db'  # Change database file name
```

### Folders

**File: `webapp/app.py`**
```python
app.config['UPLOAD_FOLDER'] = 'uploads'   # Upload directory
app.config['OUTPUT_FOLDER'] = 'outputs'   # Output directory
```

## Label Mapping (VOC Format)

Nếu sử dụng Pascal VOC format, cập nhật label mapping trong `webapp/augmentation_service.py`:

```python
def read_image_and_label(self, img_path, label_path, label_format):
    ...
    if label_format == 'voc':
        # Customize your class mapping here
        label_mapping = {
            'person': 0,
            'car': 1,
            'dog': 2,
            'cat': 3,
            'bicycle': 4,
            'motorbike': 5,
            # Add more classes as needed
        }
        bboxes = get_info_bbox_pascalvoc(label_path, label_mapping)
```

## Docker Configuration

### Memory Limits

**File: `docker-compose.yml`**
```yaml
services:
  webapp:
    build: .
    mem_limit: 4g          # Memory limit
    cpus: 2                # CPU cores
```

### Environment Variables

**File: `docker-compose.yml`**
```yaml
environment:
  - FLASK_ENV=production
  - MAX_UPLOAD_SIZE=100000000
  - DEBUG=False
```

### Network Configuration

**File: `docker-compose.yml`**
```yaml
networks:
  augmentation-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

## Advanced Features

### Enable Debug Mode

**Development only:**
```python
# webapp/app.py
app.run(host='0.0.0.0', port=222, debug=True)  # Set to False in production
```

### Custom Augmentation Pipeline

Thêm augmentation mới trong `webapp/augmentation_service.py`:

```python
self.augmentation_classes = {
    # ... existing augmentations ...
    
    'your_custom_aug': {
        'class': YourCustomClass,
        'params': {'param1': value1},
        'name': 'Your Augmentation Name',
        'description': 'Mô tả'
    }
}
```

### Probability Control

Trong class `Sequence`, bạn có thể kiểm soát xác suất áp dụng từng augmentation:

```python
# Current: All augmentations applied with 100% probability
aug_pipeline = Sequence(augmentations, probs=1)

# Custom: Different probabilities for each augmentation
aug_pipeline = Sequence(augmentations, probs=[0.5, 0.8, 1.0, 0.3, ...])
```

## Security Recommendations

### Production Deployment

1. **Disable Debug Mode**
```python
app.run(host='0.0.0.0', port=222, debug=False)
```

2. **Add Authentication** (example):
```python
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    # Implement your authentication logic
    pass

@app.route('/api/upload', methods=['POST'])
@auth.login_required
def upload_files():
    # Your code
```

3. **HTTPS Configuration**
```python
# Use a reverse proxy like Nginx with SSL
# Or use Flask-SSLify
```

4. **File Upload Validation**
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
# Add more strict validation as needed
```

## Performance Tuning

### Batch Processing

Để xử lý nhiều ảnh nhanh hơn, có thể implement batch processing:

```python
# In augmentation_service.py
def apply_augmentations_batch(self, task_folder, output_folder, 
                              selected_augmentations, label_format, 
                              batch_size=10):
    # Process images in batches
    pass
```

### Caching

Thêm caching cho preview:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def generate_preview_cached(task_id, augmentations_tuple):
    # Your preview logic
    pass
```

## Monitoring

### Logging

Thêm logging để theo dõi:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Metrics

Track số lượng augmentation requests, success rate, etc.

## Backup & Recovery

### Database Backup

```bash
# Backup
cp webapp/tasks.db webapp/tasks.db.backup

# Restore
cp webapp/tasks.db.backup webapp/tasks.db
```

### Full Backup

```bash
# Backup everything
tar -czf backup.tar.gz webapp/uploads webapp/outputs webapp/tasks.db

# Restore
tar -xzf backup.tar.gz
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   - Change port in app.py and docker-compose.yml

2. **Out of memory**
   - Reduce image size or batch size
   - Increase Docker memory limit

3. **File permission errors**
   - Check folder permissions (uploads/, outputs/)
   - Use volumes in Docker

4. **Augmentation not working**
   - Check augmentation module imports
   - Verify parameters are valid

## Support

Nếu cần hỗ trợ, check:
- Application logs
- Docker logs: `docker-compose logs -f`
- Browser console for frontend errors
- Network tab for API errors
