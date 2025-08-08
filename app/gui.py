import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QSplitter,
    QGroupBox, QTextEdit, QSpinBox, QComboBox, QCheckBox, QProgressBar,
    QTabWidget, QScrollArea, QDialog, QDialogButtonBox, QFormLayout,
    QLineEdit, QMessageBox, QFileDialog, QStatusBar, QMenuBar, QMenu,
    QAction, QHeaderView, QFrame
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor

from .image_capture import ImageCapture, CardDetector, CardProcessor
from .ocr import OCREngine, CardMatcher
from .scryfall import CardDataManager
from .utils import DatabaseManager, ExportManager, ConfigManager
import os
from datetime import datetime
import json

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    cards_detected = pyqtSignal(list)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.capture = ImageCapture(camera_index)
        self.detector = CardDetector()
        self.running = False
        
    def run(self):
        if not self.capture.initialize_camera():
            return
            
        self.running = True
        
        while self.running:
            frame = self.capture.get_frame()
            if frame is not None:
                self.frame_ready.emit(frame)
                
                # Detect cards
                detected_cards = self.detector.detect_cards(frame)
                if detected_cards:
                    self.cards_detected.emit(detected_cards)
            
            self.msleep(33)  # ~30 FPS
    
    def stop(self):
        self.running = False
        self.capture.release_camera()
        self.wait()

class CardProcessingThread(QThread):
    processing_complete = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    
    def __init__(self, card_image, ocr_engine, card_matcher, card_data_manager):
        super().__init__()
        self.card_image = card_image
        self.ocr_engine = ocr_engine
        self.card_matcher = card_matcher
        self.card_data_manager = card_data_manager
        self.processor = CardProcessor()
    
    def run(self):
        try:
            print("Starting card processing thread...")
            print(f"Card image shape: {self.card_image.shape}")
            
            # Extract name region and perform OCR
            print("Cropping name region...")
            name_region = self.processor.crop_name_region(self.card_image)
            print(f"Name region shape: {name_region.shape}")
            
            print("Preprocessing for OCR...")
            preprocessed_name = self.processor.preprocess_for_ocr(name_region)
            print(f"Preprocessed shape: {preprocessed_name.shape}")
            
            print("Extracting card name with OCR...")
            raw_name = self.ocr_engine.extract_card_name(preprocessed_name)
            print(f"Raw name extracted: '{raw_name}'")
            
            if not raw_name:
                print("No card name extracted from OCR")
                self.processing_error.emit("Could not extract card name")
                return
            
            # Match card name
            print("Matching card name...")
            match_result = self.card_matcher.match_card_name(raw_name)
            if not match_result:
                print(f"No match found for card name: {raw_name}")
                self.processing_error.emit(f"Could not match card name: {raw_name}")
                return
            
            # Get card data from API
            card_data = self.card_data_manager.get_card_data(
                match_result['name'], 
                match_result['type']
            )
            
            if card_data:
                result = {
                    'raw_name': raw_name,
                    'matched_name': match_result['name'],
                    'card_type': match_result['type'],
                    'confidence': match_result['confidence'],
                    'card_data': card_data,
                    'card_image': self.card_image
                }
                self.processing_complete.emit(result)
            else:
                self.processing_error.emit(f"Could not fetch data for: {match_result['name']}")
                
        except Exception as e:
            print(f"Exception in card processing thread: {e}")
            import traceback
            traceback.print_exc()
            self.processing_error.emit(f"Processing error: {str(e)}")

class CardInfoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_card_data = None
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Card header
        self.header_frame = QFrame()
        self.header_frame.setFrameStyle(QFrame.StyledPanel)
        header_layout = QVBoxLayout()
        
        self.name_label = QLabel("No card selected")
        self.name_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.name_label.setWordWrap(True)
        
        self.type_label = QLabel("")
        self.type_label.setFont(QFont("Arial", 10))
        
        self.mana_cost_label = QLabel("")
        self.set_label = QLabel("")
        
        header_layout.addWidget(self.name_label)
        header_layout.addWidget(self.type_label)
        header_layout.addWidget(self.mana_cost_label)
        header_layout.addWidget(self.set_label)
        self.header_frame.setLayout(header_layout)
        
        # Card image
        self.image_label = QLabel()
        self.image_label.setMinimumSize(200, 280)
        self.image_label.setMaximumSize(300, 420)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.image_label.setText("No image")
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # Card details
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(200)
        self.details_text.setReadOnly(True)
        
        # Price info
        self.price_label = QLabel("Price: N/A")
        self.price_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        # Action buttons
        button_layout = QHBoxLayout()
        self.add_to_batch_btn = QPushButton("Add to Batch")
        self.view_details_btn = QPushButton("View Full Details")
        
        button_layout.addWidget(self.add_to_batch_btn)
        button_layout.addWidget(self.view_details_btn)
        
        # Add all widgets to main layout
        layout.addWidget(self.header_frame)
        layout.addWidget(self.image_label)
        layout.addWidget(self.details_text)
        layout.addWidget(self.price_label)
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def update_card_info(self, card_data):
        """Update the widget with new card information."""
        self.current_card_data = card_data
        
        if not card_data:
            self.clear_info()
            return
        
        # Update header
        self.name_label.setText(card_data.get('name', 'Unknown'))
        self.type_label.setText(card_data.get('type_line', ''))
        self.mana_cost_label.setText(f"Mana Cost: {card_data.get('mana_cost', 'N/A')}")
        self.set_label.setText(f"Set: {card_data.get('set_name', 'Unknown')} ({card_data.get('set_code', '')})")
        
        # Update details
        oracle_text = card_data.get('oracle_text', 'No text available')
        details = f"Oracle Text:\n{oracle_text}\n\n"
        
        if card_data.get('power') and card_data.get('toughness'):
            details += f"Power/Toughness: {card_data['power']}/{card_data['toughness']}\n"
        
        if card_data.get('loyalty'):
            details += f"Loyalty: {card_data['loyalty']}\n"
        
        details += f"Rarity: {card_data.get('rarity', 'Unknown').title()}\n"
        details += f"Artist: {card_data.get('artist', 'Unknown')}"
        
        self.details_text.setText(details)
        
        # Update price
        prices = card_data.get('prices', {})
        if isinstance(prices, dict) and prices.get('usd'):
            self.price_label.setText(f"Price: ${prices['usd']} USD")
        else:
            self.price_label.setText("Price: N/A")
        
        # Load image if available
        image_url = card_data.get('image_url', '')
        if image_url:
            # In a real implementation, you'd download and display the image
            self.image_label.setText("Image loading...")
        else:
            self.image_label.setText("No image available")
    
    def clear_info(self):
        """Clear all card information."""
        self.name_label.setText("No card selected")
        self.type_label.setText("")
        self.mana_cost_label.setText("")
        self.set_label.setText("")
        self.details_text.setText("")
        self.price_label.setText("Price: N/A")
        self.image_label.setText("No image")

class BatchDialog(QDialog):
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.selected_batch_id = None
        self.init_ui()
        self.load_batches()
    
    def init_ui(self):
        self.setWindowTitle("Select Batch")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout()
        
        # Create new batch section
        new_batch_group = QGroupBox("Create New Batch")
        new_batch_layout = QFormLayout()
        
        self.batch_name_edit = QLineEdit()
        self.batch_desc_edit = QLineEdit()
        
        new_batch_layout.addRow("Name:", self.batch_name_edit)
        new_batch_layout.addRow("Description:", self.batch_desc_edit)
        
        self.create_batch_btn = QPushButton("Create New Batch")
        self.create_batch_btn.clicked.connect(self.create_new_batch)
        new_batch_layout.addRow(self.create_batch_btn)
        
        new_batch_group.setLayout(new_batch_layout)
        
        # Existing batches section
        existing_batch_group = QGroupBox("Existing Batches")
        existing_batch_layout = QVBoxLayout()
        
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(4)
        self.batch_table.setHorizontalHeaderLabels(["Name", "Cards", "Value", "Created"])
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        self.batch_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        existing_batch_layout.addWidget(self.batch_table)
        existing_batch_group.setLayout(existing_batch_layout)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(new_batch_group)
        layout.addWidget(existing_batch_group)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def load_batches(self):
        """Load existing batches into the table."""
        batches = self.db_manager.get_batches()
        
        self.batch_table.setRowCount(len(batches))
        
        for row, batch in enumerate(batches):
            self.batch_table.setItem(row, 0, QTableWidgetItem(batch['name']))
            self.batch_table.setItem(row, 1, QTableWidgetItem(str(batch['total_cards'])))
            self.batch_table.setItem(row, 2, QTableWidgetItem(f"${batch['total_value']:.2f}"))
            
            created_date = datetime.fromisoformat(batch['created_at']).strftime("%Y-%m-%d")
            self.batch_table.setItem(row, 3, QTableWidgetItem(created_date))
            
            # Store batch ID in the first item
            self.batch_table.item(row, 0).setData(Qt.UserRole, batch['id'])
    
    def create_new_batch(self):
        """Create a new batch."""
        name = self.batch_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a batch name.")
            return
        
        description = self.batch_desc_edit.text().strip()
        batch_id = self.db_manager.create_batch(name, description)
        
        self.selected_batch_id = batch_id
        self.accept()
    
    def accept(self):
        """Handle dialog acceptance."""
        if not self.selected_batch_id:
            # Get selected batch from table
            current_row = self.batch_table.currentRow()
            if current_row >= 0:
                item = self.batch_table.item(current_row, 0)
                self.selected_batch_id = item.data(Qt.UserRole)
        
        if self.selected_batch_id:
            super().accept()
        else:
            QMessageBox.warning(self, "Warning", "Please select or create a batch.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.db_manager = DatabaseManager(self.config_manager.get('database_path'))
        self.export_manager = ExportManager(self.db_manager)
        self.ocr_engine = OCREngine()
        self.card_matcher = CardMatcher()
        self.card_data_manager = CardDataManager(self.config_manager.get('cache_directory'))
        
        # Initialize threads
        self.camera_thread = None
        self.processing_thread = None
        
        # Current state
        self.current_frame = None
        self.detected_cards = []
        self.auto_capture_enabled = False
        self.current_batch_id = None
        
        self.init_ui()
        self.init_camera()
        
        # Auto-capture timer
        self.auto_capture_timer = QTimer()
        self.auto_capture_timer.timeout.connect(self.auto_capture_card)
    
    def init_ui(self):
        self.setWindowTitle("Card Scanner - Real-time MTG/Pokemon/Sports Card Recognition")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Top section - dual panel
        top_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - camera feed
        camera_group = QGroupBox("Live Camera Feed")
        camera_layout = QVBoxLayout()
        
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid gray;")
        self.camera_label.setText("Camera not initialized")
        self.camera_label.setAlignment(Qt.AlignCenter)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        self.capture_btn = QPushButton("Capture Card")
        self.capture_btn.setMinimumHeight(40)
        self.capture_btn.clicked.connect(self.capture_card)
        
        self.auto_capture_checkbox = QCheckBox("Auto Capture")
        self.auto_capture_checkbox.toggled.connect(self.toggle_auto_capture)
        
        self.batch_select_btn = QPushButton("Select Batch")
        self.batch_select_btn.clicked.connect(self.select_batch)
        
        camera_controls.addWidget(self.capture_btn)
        camera_controls.addWidget(self.auto_capture_checkbox)
        camera_controls.addWidget(self.batch_select_btn)
        camera_controls.addStretch()
        
        camera_layout.addWidget(self.camera_label)
        camera_layout.addLayout(camera_controls)
        camera_group.setLayout(camera_layout)
        
        # Right panel - card information
        self.card_info_widget = CardInfoWidget()
        self.card_info_widget.add_to_batch_btn.clicked.connect(self.add_current_card_to_batch)
        
        top_splitter.addWidget(camera_group)
        top_splitter.addWidget(self.card_info_widget)
        top_splitter.setSizes([800, 400])
        
        # Bottom section - capture log
        bottom_group = QGroupBox("Captured Cards")
        bottom_layout = QVBoxLayout()
        
        # Table controls
        table_controls = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_card_table)
        
        self.export_btn = QPushButton("Export Batch")
        self.export_btn.clicked.connect(self.export_batch)
        
        table_controls.addWidget(self.refresh_btn)
        table_controls.addWidget(self.export_btn)
        table_controls.addStretch()
        
        # Cards table
        self.cards_table = QTableWidget()
        self.cards_table.setColumnCount(8)
        self.cards_table.setHorizontalHeaderLabels([
            "Name", "Type", "Set", "Rarity", "Condition", "Quantity", "Value", "Captured"
        ])
        self.cards_table.horizontalHeader().setStretchLastSection(True)
        self.cards_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cards_table.setAlternatingRowColors(True)
        
        bottom_layout.addLayout(table_controls)
        bottom_layout.addWidget(self.cards_table)
        bottom_group.setLayout(bottom_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Add to main layout
        main_layout.addWidget(top_splitter, 2)
        main_layout.addWidget(bottom_group, 1)
        main_layout.addWidget(self.progress_bar)
        
        central_widget.setLayout(main_layout)
        
        # Load initial data
        self.refresh_card_table()
    
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        export_action = QAction('Export Batch...', self)
        export_action.triggered.connect(self.export_batch)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        stats_action = QAction('Statistics', self)
        stats_action.triggered.connect(self.show_statistics)
        view_menu.addAction(stats_action)
        
        # Settings menu
        settings_menu = menubar.addMenu('Settings')
        
        config_action = QAction('Preferences...', self)
        config_action.triggered.connect(self.show_preferences)
        settings_menu.addAction(config_action)
    
    def init_camera(self):
        """Initialize the camera thread."""
        camera_index = self.config_manager.get('camera_index', 0)
        self.camera_thread = CameraThread(camera_index)
        self.camera_thread.frame_ready.connect(self.update_camera_display)
        self.camera_thread.cards_detected.connect(self.update_detected_cards)
        self.camera_thread.start()
    
    def update_camera_display(self, frame):
        """Update the camera display with the latest frame."""
        self.current_frame = frame.copy()
        
        # Draw bounding boxes for detected cards
        display_frame = frame.copy()
        for contour, approx in self.detected_cards:
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
            
            # Add confidence text
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(display_frame, "Card Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Convert to Qt format and display
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled_pixmap)
    
    def update_detected_cards(self, detected_cards):
        """Update the list of detected cards."""
        self.detected_cards = detected_cards
        
        # Update status
        if detected_cards:
            self.status_bar.showMessage(f"Detected {len(detected_cards)} card(s)")
        else:
            self.status_bar.showMessage("No cards detected")
    
    def capture_card(self):
        """Capture and process a card."""
        if not self.current_frame is not None or not self.detected_cards:
            QMessageBox.warning(self, "Warning", "No cards detected in current frame.")
            return
        
        # Use the largest detected card
        largest_card = max(self.detected_cards, key=lambda x: cv2.contourArea(x[0]))
        contour = largest_card[0]
        
        # Extract card ROI
        detector = CardDetector()
        card_image = detector.extract_card_roi(self.current_frame, contour)
        
        if card_image is None:
            QMessageBox.warning(self, "Warning", "Could not extract card from image.")
            return
        
        # Start processing in background thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_bar.showMessage("Processing card...")
        
        self.processing_thread = CardProcessingThread(
            card_image, self.ocr_engine, self.card_matcher, self.card_data_manager
        )
        self.processing_thread.processing_complete.connect(self.on_card_processed)
        self.processing_thread.processing_error.connect(self.on_processing_error)
        self.processing_thread.start()
    
    def on_card_processed(self, result):
        """Handle successful card processing."""
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Card processed successfully")
        
        # Update card info display
        self.card_info_widget.update_card_info(result['card_data'])
        
        # Save card image
        if self.config_manager.get('image_save_enabled', True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result['matched_name'].replace(' ', '_')}_{timestamp}.jpg"
            image_dir = self.config_manager.get('image_save_directory', 'captured_cards')
            
            processor = CardProcessor()
            image_path = processor.save_card_image(result['card_image'], filename, image_dir)
        else:
            image_path = ""
        
        # Add to database
        card_id = self.db_manager.add_card(
            result['card_data'],
            image_path=image_path,
            condition="Near Mint",
            quantity=1
        )
        
        # Add to current batch if selected
        if self.current_batch_id:
            self.db_manager.add_card_to_batch(self.current_batch_id, card_id)
        
        # Refresh table
        self.refresh_card_table()
        
        # Show success message
        QMessageBox.information(
            self, 
            "Success", 
            f"Card '{result['matched_name']}' captured and added to database."
        )
    
    def on_processing_error(self, error_message):
        """Handle card processing errors."""
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Processing failed")
        QMessageBox.warning(self, "Processing Error", error_message)
    
    def toggle_auto_capture(self, enabled):
        """Toggle auto-capture mode."""
        self.auto_capture_enabled = enabled
        
        if enabled:
            interval = self.config_manager.get('auto_capture_interval', 2.0) * 1000
            self.auto_capture_timer.start(int(interval))
            self.status_bar.showMessage("Auto-capture enabled")
        else:
            self.auto_capture_timer.stop()
            self.status_bar.showMessage("Auto-capture disabled")
    
    def auto_capture_card(self):
        """Automatically capture a card if one is detected."""
        if self.detected_cards and not self.processing_thread:
            self.capture_card()
    
    def select_batch(self):
        """Open batch selection dialog."""
        dialog = BatchDialog(self.db_manager, self)
        if dialog.exec_() == QDialog.Accepted:
            self.current_batch_id = dialog.selected_batch_id
            
            # Update UI to show selected batch
            batches = self.db_manager.get_batches()
            batch_name = "Unknown"
            for batch in batches:
                if batch['id'] == self.current_batch_id:
                    batch_name = batch['name']
                    break
            
            self.batch_select_btn.setText(f"Batch: {batch_name}")
            self.status_bar.showMessage(f"Selected batch: {batch_name}")
    
    def add_current_card_to_batch(self):
        """Add the currently displayed card to a batch."""
        if not self.card_info_widget.current_card_data:
            QMessageBox.warning(self, "Warning", "No card selected.")
            return
        
        if not self.current_batch_id:
            self.select_batch()
            if not self.current_batch_id:
                return
        
        # This would need to be implemented based on how cards are tracked
        QMessageBox.information(self, "Info", "Feature not yet implemented.")
    
    def refresh_card_table(self):
        """Refresh the cards table."""
        cards = self.db_manager.get_cards(limit=100)
        
        self.cards_table.setRowCount(len(cards))
        
        for row, card in enumerate(cards):
            self.cards_table.setItem(row, 0, QTableWidgetItem(card['name']))
            self.cards_table.setItem(row, 1, QTableWidgetItem(card['card_type']))
            self.cards_table.setItem(row, 2, QTableWidgetItem(card['set_name']))
            self.cards_table.setItem(row, 3, QTableWidgetItem(card['rarity']))
            self.cards_table.setItem(row, 4, QTableWidgetItem(card['condition']))
            self.cards_table.setItem(row, 5, QTableWidgetItem(str(card['quantity'])))
            self.cards_table.setItem(row, 6, QTableWidgetItem(f"${card['estimated_value']:.2f}"))
            
            # Format timestamp
            timestamp = datetime.fromisoformat(card['capture_timestamp'])
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M")
            self.cards_table.setItem(row, 7, QTableWidgetItem(formatted_time))
    
    def export_batch(self):
        """Export a batch to file."""
        if not self.current_batch_id:
            QMessageBox.warning(self, "Warning", "No batch selected.")
            return
        
        # Get export format
        formats = ["JSON (*.json)", "CSV (*.csv)"]
        format_choice, ok = QMessageBox.question(
            self, "Export Format", "Choose export format:", 
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        
        if not ok:
            return
        
        # Get file path
        if format_choice == QMessageBox.Yes:  # JSON
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Batch", f"batch_{self.current_batch_id}.json", "JSON files (*.json)"
            )
            if file_path:
                success = self.export_manager.export_batch_to_json(self.current_batch_id, file_path)
        else:  # CSV
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Batch", f"batch_{self.current_batch_id}.csv", "CSV files (*.csv)"
            )
            if file_path:
                success = self.export_manager.export_batch_to_csv(self.current_batch_id, file_path)
        
        if success:
            QMessageBox.information(self, "Success", f"Batch exported to {file_path}")
        else:
            QMessageBox.warning(self, "Error", "Failed to export batch.")
    
    def show_statistics(self):
        """Show database statistics."""
        stats = self.db_manager.get_statistics()
        
        stats_text = f"""
Database Statistics:

Total Cards: {stats['total_cards']}
Total Estimated Value: ${stats['total_value']:.2f}
Total Batches: {stats['total_batches']}

Cards by Type:
"""
        
        for card_type, count in stats['cards_by_type'].items():
            stats_text += f"  {card_type}: {count}\n"
        
        stats_text += f"""
Upload Status:
  Uploaded: {stats['uploaded_cards']}
  Pending: {stats['pending_cards']}
"""
        
        QMessageBox.information(self, "Statistics", stats_text)
    
    def show_preferences(self):
        """Show preferences dialog."""
        QMessageBox.information(self, "Preferences", "Preferences dialog not yet implemented.")
    
    def closeEvent(self, event):
        """Handle application close."""
        if self.camera_thread:
            self.camera_thread.stop()
        
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.wait()
        
        # Save configuration
        self.config_manager.save_config()
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Card Scanner")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Card Recognition Tools")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())