#!/usr/bin/env python3
"""
Training Data Collection System for Card Scanner
Collects user feedback on card detection and recognition accuracy.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import base64

# Optional imports for image processing
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

@dataclass
class TrainingExample:
    """Represents a single training example with user feedback."""
    id: Optional[int] = None
    timestamp: str = ""
    image_path: str = ""
    image_base64: str = ""  # For storing small images directly
    detection_correct: bool = False
    recognition_correct: bool = False
    detected_name: str = ""
    actual_name: str = ""
    detected_set: str = ""
    actual_set: str = ""
    detected_type: str = ""
    actual_type: str = ""
    confidence_score: float = 0.0
    user_notes: str = ""
    card_region_coords: str = ""  # JSON string of bounding box coordinates
    preprocessing_params: str = ""  # JSON string of preprocessing parameters used
    ocr_results: str = ""  # JSON string of raw OCR results
    session_id: str = ""  # To group related training examples

class TrainingDataManager:
    """Manages collection and storage of training data."""
    
    def __init__(self, db_path: str = "training_data.db"):
        self.db_path = db_path
        self.training_dir = "training_data"
        self.images_dir = os.path.join(self.training_dir, "images")
        self.exports_dir = os.path.join(self.training_dir, "exports")
        
        # Create directories
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)
        
        self.init_database()
    
    def init_database(self):
        """Initialize the training data database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_path TEXT,
                    image_base64 TEXT,
                    detection_correct BOOLEAN NOT NULL,
                    recognition_correct BOOLEAN NOT NULL,
                    detected_name TEXT,
                    actual_name TEXT,
                    detected_set TEXT,
                    actual_set TEXT,
                    detected_type TEXT,
                    actual_type TEXT,
                    confidence_score REAL,
                    user_notes TEXT,
                    card_region_coords TEXT,
                    preprocessing_params TEXT,
                    ocr_results TEXT,
                    session_id TEXT
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON training_examples(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON training_examples(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_detection_correct ON training_examples(detection_correct)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_recognition_correct ON training_examples(recognition_correct)")
    
    def save_training_example(self, example: TrainingExample) -> int:
        """Save a training example to the database."""
        if not example.timestamp:
            example.timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO training_examples (
                    timestamp, image_path, image_base64, detection_correct, recognition_correct,
                    detected_name, actual_name, detected_set, actual_set, detected_type, actual_type,
                    confidence_score, user_notes, card_region_coords, preprocessing_params,
                    ocr_results, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                example.timestamp, example.image_path, example.image_base64,
                example.detection_correct, example.recognition_correct,
                example.detected_name, example.actual_name,
                example.detected_set, example.actual_set,
                example.detected_type, example.actual_type,
                example.confidence_score, example.user_notes,
                example.card_region_coords, example.preprocessing_params,
                example.ocr_results, example.session_id
            ))
            return cursor.lastrowid
    
    def save_card_image(self, image, session_id: str, example_id: Optional[int] = None) -> str:
        """Save a card image to disk and return the path."""
        if not CV2_AVAILABLE:
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if example_id:
            filename = f"card_{session_id}_{example_id}_{timestamp}.jpg"
        else:
            filename = f"card_{session_id}_{timestamp}.jpg"
        
        image_path = os.path.join(self.images_dir, filename)
        cv2.imwrite(image_path, image)
        return image_path
    
    def encode_image_base64(self, image, max_size: Tuple[int, int] = (200, 280)) -> str:
        """Encode image as base64 string for database storage (for small thumbnails)."""
        if not CV2_AVAILABLE:
            return ""
        
        # Resize image to thumbnail size
        h, w = image.shape[:2]
        max_w, max_h = max_size
        
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    def get_training_examples(self, limit: int = 100, offset: int = 0) -> List[TrainingExample]:
        """Retrieve training examples from the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM training_examples 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            examples = []
            for row in cursor.fetchall():
                example = TrainingExample(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    image_path=row['image_path'],
                    image_base64=row['image_base64'],
                    detection_correct=bool(row['detection_correct']),
                    recognition_correct=bool(row['recognition_correct']),
                    detected_name=row['detected_name'] or "",
                    actual_name=row['actual_name'] or "",
                    detected_set=row['detected_set'] or "",
                    actual_set=row['actual_set'] or "",
                    detected_type=row['detected_type'] or "",
                    actual_type=row['actual_type'] or "",
                    confidence_score=row['confidence_score'] or 0.0,
                    user_notes=row['user_notes'] or "",
                    card_region_coords=row['card_region_coords'] or "",
                    preprocessing_params=row['preprocessing_params'] or "",
                    ocr_results=row['ocr_results'] or "",
                    session_id=row['session_id'] or ""
                )
                examples.append(example)
            
            return examples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training data statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM training_examples")
            total_examples = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM training_examples WHERE detection_correct = 1")
            correct_detections = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM training_examples WHERE recognition_correct = 1")
            correct_recognitions = cursor.fetchone()[0]
            
            # Count examples that are fully correct (both detection and recognition)
            cursor = conn.execute("SELECT COUNT(*) FROM training_examples WHERE detection_correct = 1 AND recognition_correct = 1")
            fully_correct_examples = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT session_id) FROM training_examples")
            total_sessions = cursor.fetchone()[0]
            
            # Get accuracy rates
            detection_accuracy = (correct_detections / total_examples * 100) if total_examples > 0 else 0
            recognition_accuracy = (correct_recognitions / total_examples * 100) if total_examples > 0 else 0
            
            return {
                'total_examples': total_examples,
                'correct_detections': correct_detections,
                'correct_recognitions': correct_recognitions,
                'fully_correct_examples': fully_correct_examples,
                'total_sessions': total_sessions,
                'detection_accuracy': detection_accuracy,
                'recognition_accuracy': recognition_accuracy,
                'usable_training_data': fully_correct_examples  # Only fully correct examples are usable for ML training
            }
    
    def export_training_data(self, format: str = 'json', output_path: Optional[str] = None) -> str:
        """Export training data in various formats."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.exports_dir, f"training_data_{timestamp}.{format}")
        
        examples = self.get_training_examples(limit=10000)  # Get all examples
        
        if format.lower() == 'json':
            data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_examples': len(examples),
                    'statistics': self.get_statistics()
                },
                'examples': [asdict(example) for example in examples]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == 'csv':
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if examples:
                    writer = csv.DictWriter(f, fieldnames=asdict(examples[0]).keys())
                    writer.writeheader()
                    for example in examples:
                        writer.writerow(asdict(example))
        
        return output_path
    
    def delete_training_example(self, example_id: int) -> bool:
        """Delete a training example."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM training_examples WHERE id = ?", (example_id,))
            return cursor.rowcount > 0
    
    def clear_all_training_data(self) -> bool:
        """Clear all training data (use with caution)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM training_examples")
            return True
    
    def get_examples_by_session(self, session_id: str) -> List[TrainingExample]:
        """Get all training examples from a specific session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM training_examples 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            """, (session_id,))
            
            examples = []
            for row in cursor.fetchall():
                example = TrainingExample(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    image_path=row['image_path'],
                    image_base64=row['image_base64'],
                    detection_correct=bool(row['detection_correct']),
                    recognition_correct=bool(row['recognition_correct']),
                    detected_name=row['detected_name'] or "",
                    actual_name=row['actual_name'] or "",
                    detected_set=row['detected_set'] or "",
                    actual_set=row['actual_set'] or "",
                    detected_type=row['detected_type'] or "",
                    actual_type=row['actual_type'] or "",
                    confidence_score=row['confidence_score'] or 0.0,
                    user_notes=row['user_notes'] or "",
                    card_region_coords=row['card_region_coords'] or "",
                    preprocessing_params=row['preprocessing_params'] or "",
                    ocr_results=row['ocr_results'] or "",
                    session_id=row['session_id'] or ""
                )
                examples.append(example)
            
            return examples
    
    def get_correct_training_examples(self, limit: int = 1000, offset: int = 0) -> List[TrainingExample]:
        """Get only the training examples that are marked as fully correct (for ML training)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM training_examples 
                WHERE detection_correct = 1 AND recognition_correct = 1
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            examples = []
            for row in cursor.fetchall():
                example = TrainingExample(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    image_path=row['image_path'],
                    image_base64=row['image_base64'],
                    detection_correct=bool(row['detection_correct']),
                    recognition_correct=bool(row['recognition_correct']),
                    detected_name=row['detected_name'] or "",
                    actual_name=row['actual_name'] or "",
                    detected_set=row['detected_set'] or "",
                    actual_set=row['actual_set'] or "",
                    detected_type=row['detected_type'] or "",
                    actual_type=row['actual_type'] or "",
                    confidence_score=row['confidence_score'] or 0.0,
                    user_notes=row['user_notes'] or "",
                    card_region_coords=row['card_region_coords'] or "",
                    preprocessing_params=row['preprocessing_params'] or "",
                    ocr_results=row['ocr_results'] or "",
                    session_id=row['session_id'] or ""
                )
                examples.append(example)
            
            return examples
    
    def export_ml_training_data(self, format: str = 'json', output_path: Optional[str] = None) -> str:
        """Export only the correct training examples for machine learning purposes."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.exports_dir, f"ml_training_data_{timestamp}.{format}")
        
        # Get only correct examples
        correct_examples = self.get_correct_training_examples(limit=10000)
        
        if format.lower() == 'json':
            data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'export_type': 'ml_training_data',
                    'description': 'Only examples marked as fully correct (detection and recognition)',
                    'total_correct_examples': len(correct_examples),
                    'statistics': self.get_statistics()
                },
                'training_examples': [asdict(example) for example in correct_examples]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == 'csv':
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if correct_examples:
                    writer = csv.DictWriter(f, fieldnames=asdict(correct_examples[0]).keys())
                    writer.writeheader()
                    for example in correct_examples:
                        writer.writerow(asdict(example))
        
        return output_path