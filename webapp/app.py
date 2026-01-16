import os
import json
import uuid
import shutil
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentation_service import AugmentationService
from database import Database

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['DATABASE'] = 'tasks.db'

# Initialize services
db = Database(app.config['DATABASE'])
aug_service = AugmentationService()

# Create necessary folders
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/augmentations', methods=['GET'])
def get_augmentations():
    """Get list of available augmentations"""
    augmentations = aug_service.get_available_augmentations()
    return jsonify(augmentations)

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload images and labels"""
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    labels = request.files.getlist('labels')
    label_format = request.form.get('label_format', 'yolo')  # yolo or voc
    
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    # Create task
    task_id = str(uuid.uuid4())
    task_folder = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    images_folder = os.path.join(task_folder, 'images')
    labels_folder = os.path.join(task_folder, 'labels')
    
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
    
    uploaded_images = []
    uploaded_labels = []
    
    # Save images
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(images_folder, filename)
            file.save(filepath)
            uploaded_images.append(filename)
    
    # Save labels
    for file in labels:
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(labels_folder, filename)
            file.save(filepath)
            uploaded_labels.append(filename)
    
    if not uploaded_images:
        shutil.rmtree(task_folder)
        return jsonify({'error': 'No valid images uploaded'}), 400
    
    # Save task to database
    task_data = {
        'task_id': task_id,
        'name': request.form.get('task_name', f'Task {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'),
        'label_format': label_format,
        'image_count': len(uploaded_images),
        'created_at': datetime.now().isoformat()
    }
    
    db.create_task(task_data)
    
    return jsonify({
        'task_id': task_id,
        'uploaded_images': uploaded_images,
        'uploaded_labels': uploaded_labels,
        'message': 'Files uploaded successfully'
    })

@app.route('/api/preview/<task_id>', methods=['POST'])
def preview_augmentation(task_id):
    """Generate preview with random sample"""
    data = request.json
    selected_augmentations = data.get('augmentations', [])
    
    if not selected_augmentations:
        return jsonify({'error': 'No augmentations selected'}), 400
    
    task_folder = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    if not os.path.exists(task_folder):
        return jsonify({'error': 'Task not found'}), 404
    
    # Get task info
    task = db.get_task(task_id)
    if not task:
        return jsonify({'error': 'Task not found in database'}), 404
    
    # Generate preview
    result = aug_service.generate_preview(
        task_folder,
        selected_augmentations,
        task['label_format']
    )
    
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result)

@app.route('/api/augment/<task_id>', methods=['POST'])
def augment_task(task_id):
    """Apply augmentation to all images in task"""
    data = request.json
    selected_augmentations = data.get('augmentations', [])
    
    if not selected_augmentations:
        return jsonify({'error': 'No augmentations selected'}), 400
    
    task_folder = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    if not os.path.exists(task_folder):
        return jsonify({'error': 'Task not found'}), 404
    
    task = db.get_task(task_id)
    if not task:
        return jsonify({'error': 'Task not found in database'}), 404
    
    # Create output folder
    output_id = str(uuid.uuid4())
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], output_id)
    
    # Apply augmentation
    result = aug_service.apply_augmentations(
        task_folder,
        output_folder,
        selected_augmentations,
        task['label_format']
    )
    
    if 'error' in result:
        return jsonify(result), 400
    
    # Update task with augmentation info
    augmentation_data = {
        'output_id': output_id,
        'augmentations': selected_augmentations,
        'output_count': result['processed_count'],
        'created_at': datetime.now().isoformat()
    }
    
    db.add_augmentation(task_id, augmentation_data)
    
    return jsonify({
        'output_id': output_id,
        'message': 'Augmentation completed',
        **result
    })

@app.route('/api/download/<output_id>')
def download_results(output_id):
    """Download augmented results as zip"""
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], output_id)
    
    if not os.path.exists(output_folder):
        return jsonify({'error': 'Output not found'}), 404
    
    # Create zip file
    zip_path = f"{output_folder}.zip"
    shutil.make_archive(output_folder, 'zip', output_folder)
    
    return send_file(zip_path, as_attachment=True, download_name=f'augmented_{output_id}.zip')

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get all tasks with their augmentation history"""
    tasks = db.get_all_tasks()
    return jsonify(tasks)

@app.route('/api/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    """Delete a task and its data"""
    # Delete from database
    db.delete_task(task_id)
    
    # Delete files
    task_folder = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    if os.path.exists(task_folder):
        shutil.rmtree(task_folder)
    
    # Delete associated outputs
    task = db.get_task(task_id)
    if task and 'augmentations' in task:
        for aug in task['augmentations']:
            output_folder = os.path.join(app.config['OUTPUT_FOLDER'], aug['output_id'])
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
    
    return jsonify({'message': 'Task deleted successfully'})

@app.route('/api/preview-image/<path:filename>')
def serve_preview_image(filename):
    """Serve preview images"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=222, debug=True)
