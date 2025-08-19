#!/usr/bin/env python3
"""
Training Dialog for Card Scanner
Provides UI for collecting user feedback on card detection and recognition.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

# Optional imports for image processing
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QCheckBox, QGroupBox, QScrollArea, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QProgressBar,
    QComboBox, QSpinBox, QTabWidget, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont

from training_data import TrainingDataManager, TrainingExample
from training_coordinator import TrainingCoordinator, DetectionFeedbackRequest

class TrainingDialog(QDialog):
    """Dialog for collecting training feedback on card detection and recognition."""
    
    # Signal for feedback completion
    feedback_completed = pyqtSignal(str, dict)  # feedback_request_id, feedback_data
    
    def __init__(self, parent=None, training_coordinator=None):
        super().__init__(parent)
        self.training_manager = TrainingDataManager()
        self.training_coordinator = training_coordinator
        self.current_session_id = str(uuid.uuid4())
        self.current_card_image = None
        self.current_detection_result = None
        self.current_card_region = None
        self.current_feedback_request = None  # Current DetectionFeedbackRequest
        
        self.setWindowTitle("Card Scanner Training Mode - Automated Feedback Loop")
        self.setGeometry(100, 100, 1000, 700)
        self.setModal(False)  # Allow interaction with main window
        
        # Connect to training coordinator if provided
        if self.training_coordinator:
            self.training_coordinator.feedback_requested.connect(self.on_feedback_requested)
            self.feedback_completed.connect(self.training_coordinator.on_feedback_received)
        
        self.init_ui()
        self.load_statistics()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Training tab
        self.training_tab = self.create_training_tab()
        self.tab_widget.addTab(self.training_tab, "Training Feedback")
        
        # History tab
        self.history_tab = self.create_history_tab()
        self.tab_widget.addTab(self.history_tab, "Training History")
        
        # Statistics tab
        self.stats_tab = self.create_statistics_tab()
        self.tab_widget.addTab(self.stats_tab, "Statistics")
        
        layout.addWidget(self.tab_widget)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Export Training Data")
        self.export_btn.clicked.connect(self.export_training_data)
        
        self.clear_btn = QPushButton("Clear All Data")
        self.clear_btn.clicked.connect(self.clear_training_data)
        self.clear_btn.setStyleSheet("QPushButton { background-color: #ff6b6b; }")
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def create_training_tab(self) -> QWidget:
        """Create the training feedback tab."""
        widget = QWidget()
        layout = QHBoxLayout()
        
        # Left side - Image display
        left_group = QGroupBox("Card Image")
        left_layout = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(300, 420)
        self.image_label.setStyleSheet("border: 2px solid gray; background-color: white;")
        self.image_label.setText("No card image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        
        left_layout.addWidget(self.image_label)
        left_group.setLayout(left_layout)
        
        # Right side - Feedback form
        right_group = QGroupBox("Training Feedback")
        right_layout = QVBoxLayout()
        
        # Detection feedback
        detection_group = QGroupBox("Card Detection")
        detection_layout = QVBoxLayout()
        
        self.detection_correct_cb = QCheckBox("Card was detected correctly")
        self.detection_correct_cb.setChecked(True)
        detection_layout.addWidget(self.detection_correct_cb)
        
        detection_group.setLayout(detection_layout)
        
        # Recognition feedback
        recognition_group = QGroupBox("Card Recognition")
        recognition_layout = QGridLayout()
        
        self.recognition_correct_cb = QCheckBox("Card was recognized correctly")
        self.recognition_correct_cb.setChecked(True)
        recognition_layout.addWidget(self.recognition_correct_cb, 0, 0, 1, 2)
        
        # Add help text
        help_text = QLabel("💡 Only examples marked as BOTH detection and recognition correct will be used for ML training")
        help_text.setWordWrap(True)
        help_text.setStyleSheet("QLabel { color: #4CAF50; font-size: 11px; font-style: italic; margin: 5px; }")
        recognition_layout.addWidget(help_text, 1, 0, 1, 2)
        
        # Detection source indicator
        self.detection_source_label = QLabel("📷 Source: Manual Capture")
        self.detection_source_label.setWordWrap(True)
        self.detection_source_label.setStyleSheet("QLabel { color: #2196F3; font-size: 11px; font-weight: bold; margin: 5px; }")
        recognition_layout.addWidget(self.detection_source_label, 2, 0, 1, 2)
        
        # OCR Results section
        ocr_help = QLabel("Raw OCR Results (what the system actually read):")
        ocr_help.setStyleSheet("QLabel { color: #666; font-size: 11px; font-weight: bold; }")
        recognition_layout.addWidget(ocr_help, 3, 0, 1, 2)
        
        self.ocr_results_text = QTextEdit()
        self.ocr_results_text.setMaximumHeight(80)
        self.ocr_results_text.setReadOnly(True)
        self.ocr_results_text.setStyleSheet("QTextEdit { background-color: #f5f5f5; font-family: monospace; font-size: 10px; }")
        recognition_layout.addWidget(self.ocr_results_text, 4, 0, 1, 2)
        
        # Detected vs Actual fields
        recognition_layout.addWidget(QLabel("Detected Name:"), 5, 0)
        self.detected_name_edit = QLineEdit()
        self.detected_name_edit.setReadOnly(True)
        recognition_layout.addWidget(self.detected_name_edit, 5, 1)
        
        recognition_layout.addWidget(QLabel("Actual Name:"), 6, 0)
        self.actual_name_edit = QLineEdit()
        recognition_layout.addWidget(self.actual_name_edit, 6, 1)
        
        recognition_layout.addWidget(QLabel("Detected Set:"), 7, 0)
        self.detected_set_edit = QLineEdit()
        self.detected_set_edit.setReadOnly(True)
        recognition_layout.addWidget(self.detected_set_edit, 7, 1)
        
        recognition_layout.addWidget(QLabel("Actual Set:"), 8, 0)
        self.actual_set_edit = QLineEdit()
        recognition_layout.addWidget(self.actual_set_edit, 8, 1)
        
        recognition_layout.addWidget(QLabel("Detected Type:"), 9, 0)
        self.detected_type_edit = QLineEdit()
        self.detected_type_edit.setReadOnly(True)
        recognition_layout.addWidget(self.detected_type_edit, 9, 1)
        
        recognition_layout.addWidget(QLabel("Actual Type:"), 10, 0)
        self.actual_type_edit = QLineEdit()
        recognition_layout.addWidget(self.actual_type_edit, 10, 1)
        
        recognition_group.setLayout(recognition_layout)
        
        # Notes
        notes_group = QGroupBox("Additional Notes")
        notes_layout = QVBoxLayout()
        
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(100)
        self.notes_edit.setPlaceholderText("Enter any additional notes about this detection/recognition...")
        notes_layout.addWidget(self.notes_edit)
        
        notes_group.setLayout(notes_layout)
        
        # Quick action buttons
        quick_actions_group = QGroupBox("Quick Actions")
        quick_actions_layout = QVBoxLayout()
        
        quick_help = QLabel("For fast feedback when the result is obviously right or wrong:")
        quick_help.setStyleSheet("QLabel { color: #666; font-size: 11px; font-style: italic; }")
        quick_help.setWordWrap(True)
        
        quick_buttons_layout = QHBoxLayout()
        
        self.correct_btn = QPushButton("✓ This is Correct")
        self.correct_btn.clicked.connect(self.mark_as_correct)
        self.correct_btn.setMinimumHeight(40)
        self.correct_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.correct_btn.setEnabled(False)
        self.correct_btn.setToolTip("Mark as fully correct and add to ML training data immediately")
        
        self.incorrect_btn = QPushButton("✗ This is Wrong")
        self.incorrect_btn.clicked.connect(self.mark_as_incorrect)
        self.incorrect_btn.setMinimumHeight(40)
        self.incorrect_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        self.incorrect_btn.setEnabled(False)
        self.incorrect_btn.setToolTip("Mark as incorrect and allow manual correction")
        
        quick_buttons_layout.addWidget(self.correct_btn)
        quick_buttons_layout.addWidget(self.incorrect_btn)
        
        quick_actions_layout.addWidget(quick_help)
        quick_actions_layout.addLayout(quick_buttons_layout)
        quick_actions_group.setLayout(quick_actions_layout)
        
        # Submit button
        self.submit_btn = QPushButton("Submit Custom Feedback")
        self.submit_btn.clicked.connect(self.submit_feedback)
        self.submit_btn.setMinimumHeight(40)
        self.submit_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        self.submit_btn.setEnabled(False)
        
        right_layout.addWidget(detection_group)
        right_layout.addWidget(recognition_group)
        right_layout.addWidget(notes_group)
        right_layout.addWidget(quick_actions_group)
        right_layout.addWidget(self.submit_btn)
        right_layout.addStretch()
        
        right_group.setLayout(right_layout)
        
        layout.addWidget(left_group, 1)
        layout.addWidget(right_group, 1)
        
        widget.setLayout(layout)
        return widget
    
    def create_history_tab(self) -> QWidget:
        """Create the training history tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_training_history)
        
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.delete_selected_example)
        self.delete_btn.setStyleSheet("QPushButton { background-color: #ff6b6b; }")
        
        controls_layout.addWidget(self.refresh_btn)
        controls_layout.addWidget(self.delete_btn)
        controls_layout.addStretch()
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(8)
        self.history_table.setHorizontalHeaderLabels([
            "Timestamp", "Detection OK", "Recognition OK", "Detected Name", 
            "Actual Name", "Detected Set", "Actual Set", "Notes"
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setAlternatingRowColors(True)
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.history_table)
        
        widget.setLayout(layout)
        return widget
    
    def create_statistics_tab(self) -> QWidget:
        """Create the statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Statistics display
        self.stats_layout = QGridLayout()
        
        # Placeholder labels (will be populated by load_statistics)
        self.total_examples_label = QLabel("0")
        self.detection_accuracy_label = QLabel("0%")
        self.recognition_accuracy_label = QLabel("0%")
        self.total_sessions_label = QLabel("0")
        
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        
        for label in [self.total_examples_label, self.detection_accuracy_label, 
                     self.recognition_accuracy_label, self.total_sessions_label]:
            label.setFont(font)
            label.setAlignment(Qt.AlignCenter)
        
        self.stats_layout.addWidget(QLabel("Total Examples:"), 0, 0)
        self.stats_layout.addWidget(self.total_examples_label, 0, 1)
        
        self.stats_layout.addWidget(QLabel("Detection Accuracy:"), 1, 0)
        self.stats_layout.addWidget(self.detection_accuracy_label, 1, 1)
        
        self.stats_layout.addWidget(QLabel("Recognition Accuracy:"), 2, 0)
        self.stats_layout.addWidget(self.recognition_accuracy_label, 2, 1)
        
        self.stats_layout.addWidget(QLabel("Training Sessions:"), 3, 0)
        self.stats_layout.addWidget(self.total_sessions_label, 3, 1)
        
        # Add ML-ready data count
        self.ml_ready_label = QLabel("0")
        self.ml_ready_label.setFont(font)
        self.ml_ready_label.setAlignment(Qt.AlignCenter)
        self.ml_ready_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")
        
        self.stats_layout.addWidget(QLabel("ML-Ready Examples:"), 4, 0)
        self.stats_layout.addWidget(self.ml_ready_label, 4, 1)
        
        stats_group = QGroupBox("Training Statistics")
        stats_group.setLayout(self.stats_layout)
        
        # ML Export section
        ml_group = QGroupBox("Machine Learning Export")
        ml_layout = QVBoxLayout()
        
        ml_info = QLabel("Export only the examples you've marked as fully correct for training ML models:")
        ml_info.setWordWrap(True)
        ml_info.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        
        self.export_ml_btn = QPushButton("Export ML Training Data")
        self.export_ml_btn.clicked.connect(self.export_ml_training_data)
        self.export_ml_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        
        ml_layout.addWidget(ml_info)
        ml_layout.addWidget(self.export_ml_btn)
        ml_group.setLayout(ml_layout)
        
        layout.addWidget(stats_group)
        layout.addWidget(ml_group)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def set_card_data(self, card_image, detection_result: Dict[str, Any], 
                     card_region=None):
        """Set the current card data for training feedback."""
        if CV2_AVAILABLE and hasattr(card_image, 'copy'):
            self.current_card_image = card_image.copy()
            self.current_card_region = card_region.copy() if card_region is not None else None
        else:
            self.current_card_image = card_image
            self.current_card_region = card_region
            
        self.current_detection_result = detection_result.copy() if hasattr(detection_result, 'copy') else detection_result
        
        # Display the card image
        self.display_card_image(card_image)
        
        # Populate the form with detection results
        self.populate_detection_results(detection_result)
        
        # Enable buttons
        self.submit_btn.setEnabled(True)
        self.correct_btn.setEnabled(True)
        self.incorrect_btn.setEnabled(True)
        
        # Update window title and source indicator based on detection source
        source = detection_result.get('preprocessing_params', {}).get('source', 'manual')
        if source == 'real_time_detection':
            confidence = detection_result.get('confidence', 0)
            self.setWindowTitle(f"Card Scanner Training Mode - Live Detection (Confidence: {confidence:.2f})")
            self.detection_source_label.setText(f"🔴 LIVE: Real-time Detection (Confidence: {confidence:.2f})")
            self.detection_source_label.setStyleSheet("QLabel { color: #ff5722; font-size: 11px; font-weight: bold; margin: 5px; }")
        else:
            self.setWindowTitle("Card Scanner Training Mode - Manual Capture")
            self.detection_source_label.setText("📷 Source: Manual Capture")
            self.detection_source_label.setStyleSheet("QLabel { color: #2196F3; font-size: 11px; font-weight: bold; margin: 5px; }")
        
        # Switch to training tab
        self.tab_widget.setCurrentIndex(0)
    
    def on_feedback_requested(self, feedback_request: DetectionFeedbackRequest):
        """Handle a feedback request from the training coordinator."""
        self.current_feedback_request = feedback_request
        
        # Extract data from the feedback request
        image = feedback_request.image
        detection_results = feedback_request.detection_results
        processing_results = feedback_request.processing_results
        
        # Set the card data using the existing method
        self.set_card_data(image, processing_results)
        
        # Update the form with additional detection information
        if detection_results.get('detections'):
            detection_info = f"Detected {len(detection_results['detections'])} card(s)\n"
            for i, det in enumerate(detection_results['detections']):
                confidence = det.get('confidence', 0)
                detection_info += f"Card {i+1}: {confidence:.2f} confidence\n"
            self.ocr_results_text.setText(detection_info)
        
        # Show the dialog and bring to front
        self.show()
        self.raise_()
        self.activateWindow()
        
        # Update window title with feedback request info
        self.setWindowTitle(f"Training Feedback - Detection {feedback_request.id[:8]}")
    
    def display_card_image(self, image):
        """Display the card image in the label."""
        if image is None:
            self.image_label.setText("No card image provided")
            return
        
        if not CV2_AVAILABLE:
            self.image_label.setText("Image display requires OpenCV")
            return
        
        try:
            print(f"Displaying card image with shape: {image.shape}")
            
            # Ensure image is in the right format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 2:
                # Grayscale image, convert to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                # Already RGB or unknown format
                rgb_image = image
            
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            if pixmap.isNull():
                self.image_label.setText("Failed to create pixmap from image")
                return
            
            # Scale to fit label while maintaining aspect ratio
            label_size = self.image_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            
            print(f"Successfully displayed card image (original: {w}x{h}, scaled to fit: {label_size.width()}x{label_size.height()})")
            
        except Exception as e:
            error_msg = f"Error displaying image: {str(e)}"
            print(error_msg)
            self.image_label.setText(error_msg)
    
    def populate_detection_results(self, detection_result: Dict[str, Any]):
        """Populate the form with detection results."""
        self.detected_name_edit.setText(detection_result.get('name', ''))
        self.detected_set_edit.setText(detection_result.get('set', ''))
        self.detected_type_edit.setText(detection_result.get('type', ''))
        
        # Show raw OCR results
        ocr_results = detection_result.get('ocr_results', {})
        if ocr_results:
            ocr_text = "Raw OCR Text:\n"
            if isinstance(ocr_results, dict):
                for region, text in ocr_results.items():
                    ocr_text += f"{region}: {text}\n"
            else:
                ocr_text += str(ocr_results)
        else:
            ocr_text = "No OCR results available"
        
        self.ocr_results_text.setPlainText(ocr_text)
        
        # Pre-fill actual fields with detected values (user can modify)
        self.actual_name_edit.setText(detection_result.get('name', ''))
        self.actual_set_edit.setText(detection_result.get('set', ''))
        self.actual_type_edit.setText(detection_result.get('type', ''))
    
    def mark_as_correct(self):
        """Quick action: Mark the detection and recognition as correct."""
        if self.current_card_image is None:
            QMessageBox.warning(self, "Warning", "No card data available for training.")
            return
        
        # Set both checkboxes to True
        self.detection_correct_cb.setChecked(True)
        self.recognition_correct_cb.setChecked(True)
        
        # Use detected values as actual values (since they're correct)
        self.actual_name_edit.setText(self.detected_name_edit.text())
        self.actual_set_edit.setText(self.detected_set_edit.text())
        self.actual_type_edit.setText(self.detected_type_edit.text())
        
        # Add a note
        self.notes_edit.setPlainText("Marked as fully correct via quick action")
        
        # Submit immediately
        self._submit_training_example()
    
    def mark_as_incorrect(self):
        """Quick action: Mark the detection or recognition as incorrect."""
        if self.current_card_image is None:
            QMessageBox.warning(self, "Warning", "No card data available for training.")
            return
        
        # Set both checkboxes to False (user can adjust if needed)
        self.detection_correct_cb.setChecked(False)
        self.recognition_correct_cb.setChecked(False)
        
        # Add a note
        self.notes_edit.setPlainText("Marked as incorrect via quick action - please correct the actual values")
        
        # Don't submit immediately - let user correct the actual values
        QMessageBox.information(self, "Next Step", 
                              "Please fill in the correct card information in the 'Actual' fields, "
                              "then click 'Submit Custom Feedback'.")
    
    def submit_feedback(self):
        """Submit the training feedback with current form values."""
        self._submit_training_example()
    
    def _submit_training_example(self):
        """Internal method to submit training example."""
        if self.current_card_image is None:
            QMessageBox.warning(self, "Warning", "No card data available for training.")
            return
        
        try:
            # Prepare feedback data
            feedback_data = {
                'detection_correct': self.detection_correct_cb.isChecked(),
                'recognition_correct': self.recognition_correct_cb.isChecked(),
                'detected_name': self.detected_name_edit.text(),
                'actual_name': self.actual_name_edit.text(),
                'detected_set': self.detected_set_edit.text(),
                'actual_set': self.actual_set_edit.text(),
                'detected_type': self.detected_type_edit.text(),
                'actual_type': self.actual_type_edit.text(),
                'notes': self.notes_edit.toPlainText(),
                'timestamp': datetime.now().isoformat(),
                'session_id': self.current_session_id
            }
            
            # If we have a feedback request from the coordinator, send feedback there
            if self.current_feedback_request:
                self.feedback_completed.emit(self.current_feedback_request.id, feedback_data)
                self.current_feedback_request = None
                
                # Show success message
                QMessageBox.information(self, "Feedback Submitted", 
                                      "Thank you! Your feedback has been submitted to the training system.")
                
                # Clear the form for next feedback
                self._clear_form()
                return
            
            # Legacy path: Create training example for manual training
            example = TrainingExample(
                timestamp=datetime.now().isoformat(),
                detection_correct=feedback_data['detection_correct'],
                recognition_correct=feedback_data['recognition_correct'],
                detected_name=feedback_data['detected_name'],
                actual_name=feedback_data['actual_name'],
                detected_set=feedback_data['detected_set'],
                actual_set=feedback_data['actual_set'],
                detected_type=feedback_data['detected_type'],
                actual_type=feedback_data['actual_type'],
                confidence_score=self.current_detection_result.get('confidence', 0.0),
                user_notes=feedback_data['notes'],
                session_id=self.current_session_id,
                ocr_results=json.dumps(self.current_detection_result.get('ocr_results', {})),
                preprocessing_params=json.dumps(self.current_detection_result.get('preprocessing_params', {}))
            )
            
            # Save card image
            image_path = self.training_manager.save_card_image(
                self.current_card_image, self.current_session_id
            )
            example.image_path = image_path
            
            # Encode thumbnail for database
            example.image_base64 = self.training_manager.encode_image_base64(self.current_card_image)
            
            # Save to database
            example_id = self.training_manager.save_training_example(example)
            
            # Show appropriate message based on correctness
            if example.detection_correct and example.recognition_correct:
                QMessageBox.information(self, "✓ ML Training Data Added", 
                                      f"Perfect! This example will be used for ML training.\n"
                                      f"Training ID: {example_id}")
            else:
                QMessageBox.information(self, "Feedback Saved", 
                                      f"Training feedback saved for analysis.\n"
                                      f"(Not ML-ready - detection or recognition marked as incorrect)\n"
                                      f"Training ID: {example_id}")
            
            # Clear form
            self.clear_form()
            
            # Refresh statistics
            self.load_statistics()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save training feedback: {str(e)}")
    
    def clear_form(self):
        """Clear the training form."""
        self.detection_correct_cb.setChecked(True)
        self.recognition_correct_cb.setChecked(True)
        self.detected_name_edit.clear()
        self.actual_name_edit.clear()
        self.detected_set_edit.clear()
        self.actual_set_edit.clear()
        self.detected_type_edit.clear()
        self.actual_type_edit.clear()
        self.notes_edit.clear()
        self.image_label.clear()
        self.image_label.setText("No card image loaded")
        self.submit_btn.setEnabled(False)
        self.correct_btn.setEnabled(False)
        self.incorrect_btn.setEnabled(False)
        
        self.current_card_image = None
        self.current_detection_result = None
        self.current_card_region = None
    
    def load_training_history(self):
        """Load training history into the table."""
        try:
            examples = self.training_manager.get_training_examples(limit=200)
            
            self.history_table.setRowCount(len(examples))
            
            for row, example in enumerate(examples):
                self.history_table.setItem(row, 0, QTableWidgetItem(example.timestamp))
                self.history_table.setItem(row, 1, QTableWidgetItem("✓" if example.detection_correct else "✗"))
                self.history_table.setItem(row, 2, QTableWidgetItem("✓" if example.recognition_correct else "✗"))
                self.history_table.setItem(row, 3, QTableWidgetItem(example.detected_name))
                self.history_table.setItem(row, 4, QTableWidgetItem(example.actual_name))
                self.history_table.setItem(row, 5, QTableWidgetItem(example.detected_set))
                self.history_table.setItem(row, 6, QTableWidgetItem(example.actual_set))
                self.history_table.setItem(row, 7, QTableWidgetItem(example.user_notes))
                
                # Store example ID in the first column for deletion
                self.history_table.item(row, 0).setData(Qt.UserRole, example.id)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load training history: {str(e)}")
    
    def delete_selected_example(self):
        """Delete the selected training example."""
        current_row = self.history_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a training example to delete.")
            return
        
        # Confirm deletion
        reply = QMessageBox.question(self, "Confirm Deletion", 
                                   "Are you sure you want to delete this training example?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                example_id = self.history_table.item(current_row, 0).data(Qt.UserRole)
                if self.training_manager.delete_training_example(example_id):
                    self.load_training_history()
                    self.load_statistics()
                    QMessageBox.information(self, "Success", "Training example deleted successfully.")
                else:
                    QMessageBox.warning(self, "Warning", "Failed to delete training example.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete training example: {str(e)}")
    
    def load_statistics(self):
        """Load and display training statistics."""
        try:
            stats = self.training_manager.get_statistics()
            
            self.total_examples_label.setText(str(stats['total_examples']))
            self.detection_accuracy_label.setText(f"{stats['detection_accuracy']:.1f}%")
            self.recognition_accuracy_label.setText(f"{stats['recognition_accuracy']:.1f}%")
            self.total_sessions_label.setText(str(stats['total_sessions']))
            self.ml_ready_label.setText(str(stats['usable_training_data']))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load statistics: {str(e)}")
    
    def export_training_data(self):
        """Export all training data to file."""
        try:
            output_path = self.training_manager.export_training_data('json')
            QMessageBox.information(self, "Success", f"All training data exported to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export training data: {str(e)}")
    
    def export_ml_training_data(self):
        """Export only correct training data for machine learning."""
        try:
            # Check if we have any correct examples
            stats = self.training_manager.get_statistics()
            if stats['usable_training_data'] == 0:
                QMessageBox.warning(self, "No ML Data", 
                                  "No fully correct examples available for ML training.\n"
                                  "Mark some examples as both detection and recognition correct first.")
                return
            
            output_path = self.training_manager.export_ml_training_data('json')
            QMessageBox.information(self, "Success", 
                                  f"ML training data exported to:\n{output_path}\n\n"
                                  f"Exported {stats['usable_training_data']} correct examples "
                                  f"out of {stats['total_examples']} total examples.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export ML training data: {str(e)}")
    
    def clear_training_data(self):
        """Clear all training data."""
        reply = QMessageBox.question(self, "Confirm Clear", 
                                   "Are you sure you want to clear ALL training data?\n"
                                   "This action cannot be undone!",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                self.training_manager.clear_all_training_data()
                self.load_training_history()
                self.load_statistics()
                QMessageBox.information(self, "Success", "All training data cleared successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear training data: {str(e)}")
    
    def _clear_form(self):
        """Clear the training form for next feedback."""
        self.detection_correct_cb.setChecked(True)
        self.recognition_correct_cb.setChecked(True)
        self.detected_name_edit.clear()
        self.actual_name_edit.clear()
        self.detected_set_edit.clear()
        self.actual_set_edit.clear()
        self.detected_type_edit.clear()
        self.actual_type_edit.clear()
        self.notes_edit.clear()
        self.ocr_results_text.clear()
        self.image_label.setText("Waiting for next detection...")
        
        # Disable buttons until next feedback request
        self.submit_btn.setEnabled(False)
        self.correct_btn.setEnabled(False)
        self.incorrect_btn.setEnabled(False)
        
        # Reset window title
        self.setWindowTitle("Card Scanner Training Mode - Automated Feedback Loop")
    
    def showEvent(self, event):
        """Called when dialog is shown."""
        super().showEvent(event)
        self.load_training_history()
        self.load_statistics()