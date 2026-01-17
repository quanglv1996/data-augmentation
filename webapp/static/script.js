// Global variables
let currentTaskId = null;
let currentReaugmentTaskId = null;
let availableAugmentations = [];
let currentOutputId = null;

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    initTabs();
    loadAugmentations();
    loadTaskHistory();
    
    // Event listeners
    document.getElementById('upload-btn').addEventListener('click', uploadFiles);
    document.getElementById('preview-btn').addEventListener('click', previewAugmentation);
    document.getElementById('apply-btn').addEventListener('click', applyAugmentation);
    document.getElementById('reaugment-apply-btn').addEventListener('click', applyReaugmentation);
    
    // Modal close
    document.querySelector('.close').addEventListener('click', closeModal);
    window.addEventListener('click', function(event) {
        const modal = document.getElementById('reaugment-modal');
        if (event.target == modal) {
            closeModal();
        }
    });
});

// Tab management
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked tab
            this.classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
            
            // Reload history when switching to history tab
            if (tabName === 'history') {
                loadTaskHistory();
            }
        });
    });
}

// Load available augmentations
async function loadAugmentations() {
    try {
        const response = await fetch('/api/augmentations');
        availableAugmentations = await response.json();
        renderAugmentations('augmentation-list');
        renderAugmentations('reaugment-list');
    } catch (error) {
        showError('Không thể tải danh sách augmentations');
    }
}

// Render augmentation checkboxes
function renderAugmentations(containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    availableAugmentations.forEach(aug => {
        const item = document.createElement('div');
        item.className = 'augmentation-item';
        item.innerHTML = `
            <label>
                <input type="checkbox" value="${aug.id}" class="aug-checkbox">
                ${aug.name}
            </label>
            <p>${aug.description}</p>
        `;
        
        // Add click handler to toggle selection
        item.addEventListener('click', function(e) {
            if (e.target.tagName !== 'INPUT') {
                const checkbox = this.querySelector('input[type="checkbox"]');
                checkbox.checked = !checkbox.checked;
            }
            this.classList.toggle('selected', this.querySelector('input[type="checkbox"]').checked);
        });
        
        container.appendChild(item);
    });
}

// Get selected augmentations
function getSelectedAugmentations(containerId) {
    const checkboxes = document.querySelectorAll(`#${containerId} .aug-checkbox:checked`);
    return Array.from(checkboxes).map(cb => cb.value);
}

// Upload files
async function uploadFiles() {
    const taskName = document.getElementById('task-name').value;
    const labelFormat = document.getElementById('label-format').value;
    const images = document.getElementById('images-input').files;
    const labels = document.getElementById('labels-input').files;
    
    if (images.length === 0) {
        showError('Vui lòng chọn ảnh');
        return;
    }
    
    const formData = new FormData();
    formData.append('task_name', taskName || `Task ${new Date().toLocaleString()}`);
    formData.append('label_format', labelFormat);
    
    for (let i = 0; i < images.length; i++) {
        formData.append('images', images[i]);
    }
    
    for (let i = 0; i < labels.length; i++) {
        formData.append('labels', labels[i]);
    }
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentTaskId = result.task_id;
            showSuccess(`Đã tải lên ${result.uploaded_images.length} ảnh thành công`);
            
            // Show augmentation selection
            document.getElementById('augmentation-section').style.display = 'block';
            document.getElementById('preview-section').style.display = 'none';
            document.getElementById('results-section').style.display = 'none';
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Lỗi khi tải file lên');
    } finally {
        showLoading(false);
    }
}

// Preview augmentation
async function previewAugmentation() {
    const selectedAugmentations = getSelectedAugmentations('augmentation-list');
    
    if (selectedAugmentations.length === 0) {
        showError('Vui lòng chọn ít nhất một phương pháp augmentation');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`/api/preview/${currentTaskId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                augmentations: selectedAugmentations
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Show preview section
            document.getElementById('preview-section').style.display = 'block';
            
            // Create preview container HTML
            const previewContainer = document.querySelector('.preview-container');
            previewContainer.innerHTML = `
                <div class="preview-item">
                    <h3>Ảnh gốc</h3>
                    <img src="/api/preview-image/${result.original_image}" alt="Original Image">
                    <p>Số bbox: ${result.original_bbox_count}</p>
                </div>
            `;
            
            // Add each augmented image
            result.augmented_images.forEach(aug => {
                const augItem = document.createElement('div');
                augItem.className = 'preview-item';
                augItem.innerHTML = `
                    <h3>${aug.augmentation_name}</h3>
                    <img src="/api/preview-image/${aug.image_path}" alt="${aug.augmentation_name}">
                    <p>Số bbox: ${aug.bbox_count}</p>
                `;
                previewContainer.appendChild(augItem);
            });
            
            showSuccess('Xem trước thành công');
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Lỗi khi tạo preview');
    } finally {
        showLoading(false);
    }
}

// Apply augmentation
async function applyAugmentation() {
    const selectedAugmentations = getSelectedAugmentations('augmentation-list');
    
    if (selectedAugmentations.length === 0) {
        showError('Vui lòng chọn ít nhất một phương pháp augmentation');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`/api/augment/${currentTaskId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                augmentations: selectedAugmentations
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentOutputId = result.output_id;
            
            // Show results
            document.getElementById('results-section').style.display = 'block';
            const resultsInfo = document.querySelector('.results-info');
            resultsInfo.innerHTML = `
                <p>✅ Hoàn thành augmentation!</p>
                <div style="margin: 20px 0; text-align: left; max-width: 400px; margin-left: auto; margin-right: auto;">
                    <p><strong>📊 Thống kê:</strong></p>
                    <p>• Ảnh gốc: <span style="color: #667eea;">${result.original_count}</span></p>
                    <p>• Ảnh augmented: <span style="color: #667eea;">${result.augmented_count}</span></p>
                    <p>• Tổng cộng: <span style="color: #667eea;">${result.total_count}</span> ảnh</p>
                </div>
                <button id="download-btn" class="btn btn-primary">⬇️ Tải về kết quả</button>
            `;
            
            // Add download button handler
            document.getElementById('download-btn').onclick = () => downloadResults(currentOutputId);
            
            showSuccess(`Đã xử lý thành công! Tổng ${result.total_count} ảnh`);
            
            // Reload task history
            loadTaskHistory();
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Lỗi khi áp dụng augmentation');
    } finally {
        showLoading(false);
    }
}

// Download results
function downloadResults(outputId) {
    window.location.href = `/api/download/${outputId}`;
}

// Load task history
async function loadTaskHistory() {
    try {
        const response = await fetch('/api/tasks');
        const tasks = await response.json();
        
        renderTaskHistory(tasks);
    } catch (error) {
        showError('Không thể tải lịch sử tasks');
    }
}

// Render task history
function renderTaskHistory(tasks) {
    const container = document.getElementById('task-list');
    container.innerHTML = '';
    
    if (tasks.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #666;">Chưa có task nào</p>';
        return;
    }
    
    tasks.forEach(task => {
        const taskCard = document.createElement('div');
        taskCard.className = 'task-card';
        
        const augmentationsHtml = task.augmentations.map(aug => `
            <div class="aug-record">
                <div>
                    <div class="aug-tags">
                        ${aug.augmentations.map(a => `<span class="aug-tag">${a}</span>`).join('')}
                    </div>
                    <small style="color: #666; margin-top: 5px; display: block;">
                        ${new Date(aug.created_at).toLocaleString()} - ${aug.output_count} ảnh
                    </small>
                </div>
                <div class="aug-actions">
                    <button class="btn btn-primary btn-small" onclick="downloadResults('${aug.output_id}')">
                        ⬇️ Tải về
                    </button>
                </div>
            </div>
        `).join('');
        
        taskCard.innerHTML = `
            <div class="task-header">
                <h3>${task.name}</h3>
                <div>
                    <button class="btn btn-secondary btn-small" onclick="reaugmentTask('${task.task_id}')">
                        🔄 Augment lại
                    </button>
                    <button class="btn btn-danger btn-small" onclick="deleteTask('${task.task_id}')">
                        🗑️ Xóa
                    </button>
                </div>
            </div>
            <div class="task-info">
                <div class="task-info-item">
                    <strong>Số ảnh:</strong>
                    ${task.image_count}
                </div>
                <div class="task-info-item">
                    <strong>Định dạng nhãn:</strong>
                    ${task.label_format.toUpperCase()}
                </div>
                <div class="task-info-item">
                    <strong>Ngày tạo:</strong>
                    ${new Date(task.created_at).toLocaleString()}
                </div>
            </div>
            ${task.augmentations.length > 0 ? `
                <div class="augmentation-history">
                    <h4>Lịch sử Augmentation (${task.augmentations.length})</h4>
                    ${augmentationsHtml}
                </div>
            ` : '<p style="color: #666; margin-top: 10px;">Chưa có augmentation nào</p>'}
        `;
        
        container.appendChild(taskCard);
    });
}

// Re-augment task
function reaugmentTask(taskId) {
    currentReaugmentTaskId = taskId;
    
    // Clear selections
    document.querySelectorAll('#reaugment-list .aug-checkbox').forEach(cb => {
        cb.checked = false;
        cb.closest('.augmentation-item').classList.remove('selected');
    });
    
    // Show modal
    document.getElementById('reaugment-modal').style.display = 'block';
}

// Apply re-augmentation
async function applyReaugmentation() {
    const selectedAugmentations = getSelectedAugmentations('reaugment-list');
    
    if (selectedAugmentations.length === 0) {
        showError('Vui lòng chọn ít nhất một phương pháp augmentation');
        return;
    }
    
    closeModal();
    showLoading(true);
    
    try {
        const response = await fetch(`/api/augment/${currentReaugmentTaskId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                augmentations: selectedAugmentations
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showSuccess(`Đã xử lý thành công ${result.processed_count} ảnh`);
            loadTaskHistory();
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Lỗi khi áp dụng augmentation');
    } finally {
        showLoading(false);
    }
}

// Delete task
async function deleteTask(taskId) {
    if (!confirm('Bạn có chắc chắn muốn xóa task này?')) {
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`/api/tasks/${taskId}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showSuccess('Đã xóa task thành công');
            loadTaskHistory();
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError('Lỗi khi xóa task');
    } finally {
        showLoading(false);
    }
}

// Modal functions
function closeModal() {
    document.getElementById('reaugment-modal').style.display = 'none';
}

// UI Helper functions
function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'flex' : 'none';
}

function showSuccess(message) {
    alert('✅ ' + message);
}

function showError(message) {
    alert('❌ ' + message);
}
