"""
Simplified Neural Network Training Dialog
Focused on automated training integration with the training coordinator.
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QProgressBar, QTextEdit, QGroupBox,
    QFormLayout, QMessageBox, QSplitter, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QImage
import yaml
import json
import logging
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List

from dataset_manager import DatasetManager, TrainingManager
from neural_networks import create_card_detector
from training_coordinator import TrainingCoordinator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingWorkerThread(QThread):
    """Worker thread for neural network training."""
    
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    training_complete = pyqtSignal(dict)
    training_error = pyqtSignal(str)
    
    def __init__(self, training_config: Dict):
        super().__init__()
        self.training_config = training_config
        self.training_manager = TrainingManager()
    
    def run(self):
        try:
            self.status_update.emit("Initializing training...")
            
            # Create experiment
            experiment_path = self.training_manager.create_experiment(
                self.training_config['experiment_name']
            )
            
            self.progress_update.emit(10)
            self.status_update.emit("Setting up training environment...")
            
            # Setup training
            model_type = self.training_config.get('model_type', 'yolo')
            dataset_path = self.training_config.get('dataset_path', 'datasets/training_images')
            
            self.progress_update.emit(20)
            self.status_update.emit(f"Training {model_type} model...")
            
            # Train model
            results = self.training_manager.train_model(
                model_type=model_type,
                dataset_path=dataset_path,
                experiment_path=experiment_path,
                config=self.training_config
            )
            
            self.progress_update.emit(90)
            self.status_update.emit("Training completed, saving results...")
            
            # Save results
            results_summary = {
                'experiment_path': str(experiment_path),
                'model_type': model_type,
                'training_config': self.training_config,
                'results': results,
                'completion_time': datetime.now().isoformat()
            }
            
            self.progress_update.emit(100)
            self.training_complete.emit(results_summary)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.training_error.emit(str(e))

class NeuralTrainingDialog(QDialog):
    """
    Simplified neural network training dialog focused on automated training.
    """
    
    def __init__(self, parent=None, training_coordinator=None):
        super().__init__(parent)
        self.setWindowTitle("Neural Network Training - Automated System")
        self.setMinimumSize(800, 600)
        
        # Initialize managers
        self.dataset_manager = DatasetManager()
        self.training_manager = TrainingManager()
        self.training_coordinator = training_coordinator
        
        # Worker thread
        self.training_thread = None
        
        # Connect to training coordinator if provided
        if self.training_coordinator:
            self.training_coordinator.training_triggered.connect(self.on_auto_training_triggered)
            self.training_coordinator.statistics_updated.connect(self.update_coordinator_stats)
        
        # Setup UI
        self.setup_ui()
        
        # Load configuration
        self.load_default_config()
        
        logger.info("Simplified neural training dialog initialized")
    
    def setup_ui(self):
        """Setup the simplified user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Automated Neural Network Training")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Description
        description = QLabel("""
This system automatically trains neural network models based on user feedback.
Training is triggered automatically when enough feedback data is collected.
        """)
        description.setStyleSheet("margin: 10px; padding: 10px; background-color: #f0f0f0;")
        layout.addWidget(description)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Training Configuration
        self.setup_training_config(splitter)
        
        # Right panel - Status and Statistics
        self.setup_status_panel(splitter)
        
        # Progress section
        self.setup_progress_section(layout)
        
        # Control buttons
        self.setup_buttons(layout)
    
    def setup_training_config(self, parent):
        """Setup training configuration panel."""
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        
        # Training Configuration Group
        config_group = QGroupBox("Training Configuration")
        config_form = QFormLayout(config_group)
        
        # Experiment name
        self.experiment_name_edit = QLineEdit()
        self.experiment_name_edit.setText(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        config_form.addRow("Experiment Name:", self.experiment_name_edit)
        
        # Model type
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(['yolo', 'detectron2', 'custom'])
        config_form.addRow("Model Type:", self.model_type_combo)
        
        # Dataset path
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setText("datasets/training_images")
        config_form.addRow("Dataset Path:", self.dataset_path_edit)
        
        # Training parameters
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        config_form.addRow("Epochs:", self.epochs_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(16)
        config_form.addRow("Batch Size:", self.batch_size_spin)
        
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 1.0)
        self.learning_rate_spin.setDecimals(4)
        self.learning_rate_spin.setValue(0.001)
        config_form.addRow("Learning Rate:", self.learning_rate_spin)
        
        config_layout.addWidget(config_group)
        
        # Auto-training settings
        auto_group = QGroupBox("Automated Training Settings")
        auto_layout = QVBoxLayout(auto_group)
        
        self.auto_training_enabled = QCheckBox("Enable Automatic Training")
        self.auto_training_enabled.setChecked(True)
        auto_layout.addWidget(self.auto_training_enabled)
        
        self.auto_deploy_enabled = QCheckBox("Auto-deploy Trained Models")
        self.auto_deploy_enabled.setChecked(True)
        auto_layout.addWidget(self.auto_deploy_enabled)
        
        config_layout.addWidget(auto_group)
        
        parent.addWidget(config_widget)
    
    def setup_status_panel(self, parent):
        """Setup status and statistics panel."""
        status_widget = QWidget()
        status_layout = QVBoxLayout(status_widget)
        
        # Training Coordinator Statistics
        stats_group = QGroupBox("Training System Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.coordinator_stats_text = QTextEdit()
        self.coordinator_stats_text.setMaximumHeight(200)
        self.coordinator_stats_text.setReadOnly(True)
        stats_layout.addWidget(self.coordinator_stats_text)
        
        status_layout.addWidget(stats_group)
        
        # Training Log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        status_layout.addWidget(log_group)
        
        parent.addWidget(status_widget)
    
    def setup_progress_section(self, layout):
        """Setup progress display section."""
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Status label
        self.status_label = QLabel("Ready for training")
        progress_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(progress_group)
    
    def setup_buttons(self, layout):
        """Setup control buttons."""
        button_layout = QHBoxLayout()
        
        # Start training button
        self.start_training_btn = QPushButton("Start Manual Training")
        self.start_training_btn.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_training_btn)
        
        # Stop training button
        self.stop_training_btn = QPushButton("Stop Training")
        self.stop_training_btn.clicked.connect(self.stop_training)
        self.stop_training_btn.setEnabled(False)
        button_layout.addWidget(self.stop_training_btn)
        
        # Clear log button
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        button_layout.addWidget(self.clear_log_btn)
        
        # Close button
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def load_default_config(self):
        """Load default training configuration."""
        try:
            # Try to load from training coordinator config
            if self.training_coordinator:
                config = self.training_coordinator.config
                training_settings = config.get('training_settings', {})
                
                self.epochs_spin.setValue(training_settings.get('epochs', 50))
                self.batch_size_spin.setValue(training_settings.get('batch_size', 16))
                self.learning_rate_spin.setValue(training_settings.get('learning_rate', 0.001))
                
                self.auto_training_enabled.setChecked(config.get('auto_training_enabled', True))
                self.auto_deploy_enabled.setChecked(config.get('auto_deploy_models', True))
            
            self.log_text.append("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.log_text.append(f"Warning: Failed to load configuration: {e}")
    
    def start_training(self):
        """Start manual training."""
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Training is already in progress!")
            return
        
        try:
            # Prepare training configuration
            training_config = {
                'experiment_name': self.experiment_name_edit.text(),
                'model_type': self.model_type_combo.currentText(),
                'dataset_path': self.dataset_path_edit.text(),
                'epochs': self.epochs_spin.value(),
                'batch_size': self.batch_size_spin.value(),
                'learning_rate': self.learning_rate_spin.value(),
                'auto_triggered': False
            }
            
            # Validate configuration
            if not training_config['experiment_name']:
                QMessageBox.warning(self, "Warning", "Please enter an experiment name!")
                return
            
            if not os.path.exists(training_config['dataset_path']):
                QMessageBox.warning(self, "Warning", f"Dataset path does not exist: {training_config['dataset_path']}")
                return
            
            # Start training
            self.training_thread = TrainingWorkerThread(training_config)
            self.training_thread.progress_update.connect(self.on_progress_update)
            self.training_thread.status_update.connect(self.on_status_update)
            self.training_thread.training_complete.connect(self.on_training_complete)
            self.training_thread.training_error.connect(self.on_training_error)
            
            self.training_thread.start()
            
            # Update UI
            self.start_training_btn.setEnabled(False)
            self.stop_training_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            
            self.log_text.append(f"\n=== MANUAL TRAINING STARTED ===")
            self.log_text.append(f"Experiment: {training_config['experiment_name']}")
            self.log_text.append(f"Model: {training_config['model_type']}")
            self.log_text.append(f"Dataset: {training_config['dataset_path']}")
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start training: {str(e)}")
    
    def stop_training(self):
        """Stop training."""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Stop", 
                "Are you sure you want to stop training?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.training_thread.terminate()
                self.training_thread.wait(3000)
                
                self.start_training_btn.setEnabled(True)
                self.stop_training_btn.setEnabled(False)
                self.progress_bar.setValue(0)
                self.status_label.setText("Training stopped")
                
                self.log_text.append("Training stopped by user")
    
    def clear_log(self):
        """Clear the training log."""
        self.log_text.clear()
    
    def on_progress_update(self, progress: int):
        """Handle progress update."""
        self.progress_bar.setValue(progress)
    
    def on_status_update(self, status: str):
        """Handle status update."""
        self.status_label.setText(status)
        self.log_text.append(f"Status: {status}")
    
    def on_training_complete(self, results: Dict):
        """Handle training completion and notify coordinator."""
        # Update UI
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Training completed")
        
        # Log results
        self.log_text.append(f"\n=== TRAINING COMPLETED ===")
        self.log_text.append(f"Model saved to: {results.get('experiment_path', 'Unknown')}")
        
        # Notify training coordinator if available
        if self.training_coordinator:
            try:
                # Extract relevant results for coordinator
                coordinator_results = {
                    'model_path': results.get('experiment_path'),
                    'accuracy': results.get('results', {}).get('accuracy', 0),
                    'training_examples': results.get('results', {}).get('training_examples', 0),
                    'completion_time': results.get('completion_time'),
                    'model_type': results.get('model_type', 'unknown')
                }
                
                self.training_coordinator.on_training_completed(coordinator_results)
                self.log_text.append("Results sent to training coordinator")
                
            except Exception as e:
                logger.error(f"Error notifying training coordinator: {e}")
                self.log_text.append(f"Error notifying coordinator: {e}")
        
        # Show completion message
        QMessageBox.information(
            self, "Training Complete", 
            f"Training completed successfully!\n\n"
            f"Model saved to: {results.get('experiment_path', 'Unknown')}\n"
            f"The new model will be automatically deployed if auto-deployment is enabled."
        )
    
    def on_training_error(self, error: str):
        """Handle training error."""
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Training failed")
        
        self.log_text.append(f"Training error: {error}")
        QMessageBox.critical(self, "Error", f"Training failed: {error}")
    
    def on_auto_training_triggered(self, training_config: Dict):
        """Handle automatic training trigger from coordinator."""
        try:
            self.log_text.append(f"\n=== AUTO-TRAINING TRIGGERED ===")
            self.log_text.append(f"Experiment: {training_config['experiment_name']}")
            self.log_text.append(f"Feedback count: {training_config.get('feedback_count', 'Unknown')}")
            self.log_text.append(f"Correct examples: {training_config.get('correct_examples', 'Unknown')}")
            
            # Update UI with auto-training config
            self.experiment_name_edit.setText(training_config['experiment_name'])
            self.model_type_combo.setCurrentText(training_config.get('model_type', 'yolo'))
            self.epochs_spin.setValue(training_config.get('epochs', 50))
            self.batch_size_spin.setValue(training_config.get('batch_size', 16))
            
            # Set dataset path if provided
            if training_config.get('dataset_path'):
                self.dataset_path_edit.setText(training_config['dataset_path'])
            
            # Start training automatically
            self.start_training()
            
            # Show the dialog to monitor progress
            self.show()
            self.raise_()
            self.activateWindow()
            
        except Exception as e:
            logger.error(f"Error handling auto-training trigger: {e}")
            self.log_text.append(f"Error starting auto-training: {e}")
    
    def update_coordinator_stats(self, stats: Dict):
        """Update display with training coordinator statistics."""
        try:
            stats_text = f"""Training Coordinator Statistics:

🎯 Total Detections: {stats.get('total_detections', 0)}
📋 Feedback Received: {stats.get('feedback_received', 0)}
✅ Correct Detections: {stats.get('correct_detections', 0)}
❌ Incorrect Detections: {stats.get('incorrect_detections', 0)}
🔄 Training Sessions: {stats.get('training_sessions', 0)}
⏳ Pending Feedback: {stats.get('pending_feedback', 0)}

🤖 Auto-training: {'Enabled' if stats.get('auto_training_enabled', False) else 'Disabled'}
📈 Accuracy: {(stats.get('correct_detections', 0) / max(stats.get('feedback_received', 1), 1) * 100):.1f}%

Last Training: {stats.get('last_training', 'Never')}
"""
            self.coordinator_stats_text.setText(stats_text)
            
        except Exception as e:
            logger.error(f"Error updating coordinator stats: {e}")
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        # Stop any running threads
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Close", 
                "Training is in progress. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            
            self.training_thread.terminate()
            self.training_thread.wait(3000)
        
        event.accept()

def main():
    """Test the simplified training dialog."""
    app = QApplication(sys.argv)
    dialog = NeuralTrainingDialog()
    dialog.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()