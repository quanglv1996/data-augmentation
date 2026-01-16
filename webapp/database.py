import sqlite3
import json
from datetime import datetime


class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_db()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Tasks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                label_format TEXT NOT NULL,
                image_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Augmentations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS augmentations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                output_id TEXT NOT NULL,
                augmentations TEXT NOT NULL,
                output_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES tasks (task_id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_task(self, task_data):
        """Create a new task"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tasks (task_id, name, label_format, image_count, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            task_data['task_id'],
            task_data['name'],
            task_data['label_format'],
            task_data['image_count'],
            task_data['created_at']
        ))
        
        conn.commit()
        conn.close()
    
    def get_task(self, task_id):
        """Get a task by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()
        
        if row:
            task = dict(row)
            # Get augmentations for this task
            cursor.execute('SELECT * FROM augmentations WHERE task_id = ? ORDER BY created_at DESC', (task_id,))
            aug_rows = cursor.fetchall()
            task['augmentations'] = [
                {
                    'output_id': aug['output_id'],
                    'augmentations': json.loads(aug['augmentations']),
                    'output_count': aug['output_count'],
                    'created_at': aug['created_at']
                }
                for aug in aug_rows
            ]
            conn.close()
            return task
        
        conn.close()
        return None
    
    def get_all_tasks(self):
        """Get all tasks with their augmentations"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tasks ORDER BY created_at DESC')
        rows = cursor.fetchall()
        
        tasks = []
        for row in rows:
            task = dict(row)
            # Get augmentations for this task
            cursor.execute('SELECT * FROM augmentations WHERE task_id = ? ORDER BY created_at DESC', (task['task_id'],))
            aug_rows = cursor.fetchall()
            task['augmentations'] = [
                {
                    'id': aug['id'],
                    'output_id': aug['output_id'],
                    'augmentations': json.loads(aug['augmentations']),
                    'output_count': aug['output_count'],
                    'created_at': aug['created_at']
                }
                for aug in aug_rows
            ]
            tasks.append(task)
        
        conn.close()
        return tasks
    
    def add_augmentation(self, task_id, augmentation_data):
        """Add an augmentation record for a task"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO augmentations (task_id, output_id, augmentations, output_count, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            task_id,
            augmentation_data['output_id'],
            json.dumps(augmentation_data['augmentations']),
            augmentation_data['output_count'],
            augmentation_data['created_at']
        ))
        
        conn.commit()
        conn.close()
    
    def delete_task(self, task_id):
        """Delete a task and its augmentations"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM augmentations WHERE task_id = ?', (task_id,))
        cursor.execute('DELETE FROM tasks WHERE task_id = ?', (task_id,))
        
        conn.commit()
        conn.close()
    
    def delete_augmentation(self, augmentation_id):
        """Delete an augmentation record"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM augmentations WHERE id = ?', (augmentation_id,))
        
        conn.commit()
        conn.close()
