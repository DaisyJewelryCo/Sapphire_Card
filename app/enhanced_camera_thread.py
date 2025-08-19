"""
Enhanced Camera Thread with Neural Network Integration
Implements real-time card detection using the neural network pipeline.
"""

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from typing import Optional, Dict, List
import logging
import time
from datetime import datetime

from .image_capture import ImageCapture
from .enhanced_card_processor import create_enhanced_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCameraThread(QThread):
    """
    Enhanced camera thread with neural network-based card detection.
    Implements the real-time pipeline from section 4.1 of the technical guide.
    """
    
    # Signals
    frame_ready = pyqtSignal(np.ndarray)
    cards_detected = pyqtSignal(list)
    processing_complete = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    performance_update = pyqtSignal(dict)
    
    def __init__(self, camera_index: int = 0, config_path: Optional[str] = None):
        """
        Initialize enhanced camera thread.
        
        Args:
            camera_index: Camera device index
            config_path: Path to neural network configuration file
        """
        super().__init__()
        
        # Camera setup
        self.camera_index = camera_index
        self.capture = ImageCapture(camera_index)
        
        # Neural network processor
        self.processor = None
        self.config_path = config_path
        
        # Thread control
        self.running = False
        self.processing_enabled = True
        self.mutex = QMutex()
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        self.last_performance_update = time.time()
        self.performance_update_interval = 5.0  # seconds
        
        # Frame rate control
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps
        self.last_frame_time = 0
        
        # Processing control
        self.skip_frames = False
        self.frame_skip_count = 0
        self.max_processing_time = 1.0  # seconds
        
        logger.info(f"Enhanced camera thread initialized for camera {camera_index}")
    
    def initialize_processor(self) -> bool:
        """Initialize the neural network processor."""
        try:
            self.processor = create_enhanced_processor(self.config_path)
            logger.info("Neural network processor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize processor: {e}")
            self.processing_error.emit(f"Processor initialization failed: {str(e)}")
            return False
    
    def run(self):
        """Main thread execution loop."""
        # Initialize camera
        if not self.capture.initialize_camera():
            self.processing_error.emit("Failed to initialize camera")
            return
        
        # Initialize processor
        if not self.initialize_processor():
            self.processing_error.emit("Failed to initialize neural network processor")
            return
        
        self.running = True
        logger.info("Enhanced camera thread started")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Frame rate control
                if current_time - self.last_frame_time < self.frame_interval:
                    self.msleep(1)
                    continue
                
                # Capture frame
                frame = self.capture.get_frame()
                if frame is None:
                    continue
                
                self.last_frame_time = current_time
                self.frame_count += 1
                
                # Emit frame for display
                self.frame_ready.emit(frame.copy())
                
                # Process frame if enabled
                if self.processing_enabled and not self.skip_frames:
                    self._process_frame(frame)
                
                # Update performance metrics
                self._update_performance_metrics()
                
            except Exception as e:
                logger.error(f"Error in camera thread: {e}")
                self.processing_error.emit(f"Camera thread error: {str(e)}")
                break
        
        # Cleanup
        self.capture.release_camera()
        logger.info("Enhanced camera thread stopped")
    
    def _process_frame(self, frame: np.ndarray):
        """Process a single frame for card detection."""
        try:
            processing_start = time.time()
            
            # Process frame using neural network pipeline
            results = self.processor.process_frame(frame)
            
            processing_time = time.time() - processing_start
            self.processing_times.append(processing_time)
            
            # Emit results
            if results['success']:
                # Emit detected cards for GUI updates
                if results.get('detections'):
                    self.cards_detected.emit(results['detections'])
                
                # Enhance results with additional data for training coordinator
                enhanced_results = results.copy()
                enhanced_results.update({
                    'processing_time': processing_time,
                    'frame_timestamp': datetime.now().isoformat(),
                    'frame_count': self.frame_count,
                    'camera_index': self.camera_index
                })
                
                # Emit complete processing results
                self.processing_complete.emit(enhanced_results)
                
                logger.debug(f"Frame processed successfully in {processing_time:.3f}s")
            else:
                # Even emit failed processing for training coordinator to learn from
                failed_results = {
                    'success': False,
                    'message': results.get('message', 'Unknown error'),
                    'processing_time': processing_time,
                    'frame_timestamp': datetime.now().isoformat(),
                    'frame_count': self.frame_count,
                    'camera_index': self.camera_index,
                    'detections': [],
                    'error_type': 'processing_failure'
                }
                self.processing_complete.emit(failed_results)
                logger.debug(f"Frame processing failed: {results.get('message', 'Unknown error')}")
            
            # Adaptive frame skipping based on processing time
            if processing_time > self.max_processing_time:
                self.skip_frames = True
                self.frame_skip_count = min(5, int(processing_time / self.frame_interval))
                logger.debug(f"Enabling frame skipping for {self.frame_skip_count} frames")
            elif self.skip_frames:
                self.frame_skip_count -= 1
                if self.frame_skip_count <= 0:
                    self.skip_frames = False
                    logger.debug("Disabling frame skipping")
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.processing_error.emit(f"Frame processing error: {str(e)}")
    
    def _update_performance_metrics(self):
        """Update and emit performance metrics."""
        current_time = time.time()
        
        if current_time - self.last_performance_update >= self.performance_update_interval:
            try:
                # Calculate FPS
                fps = self.frame_count / self.performance_update_interval
                
                # Calculate processing statistics
                if self.processing_times:
                    avg_processing_time = np.mean(self.processing_times)
                    max_processing_time = np.max(self.processing_times)
                    min_processing_time = np.min(self.processing_times)
                else:
                    avg_processing_time = max_processing_time = min_processing_time = 0
                
                # Get processor statistics if available
                processor_stats = {}
                if self.processor:
                    try:
                        processor_stats = self.processor.get_performance_report()
                    except Exception as e:
                        logger.warning(f"Failed to get processor stats: {e}")
                
                # Compile performance metrics
                performance_metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'fps': fps,
                    'frame_count': self.frame_count,
                    'processing_times': {
                        'avg': avg_processing_time,
                        'max': max_processing_time,
                        'min': min_processing_time,
                        'count': len(self.processing_times)
                    },
                    'frame_skipping': {
                        'enabled': self.skip_frames,
                        'skip_count': self.frame_skip_count
                    },
                    'processor_stats': processor_stats
                }
                
                # Emit performance update
                self.performance_update.emit(performance_metrics)
                
                # Reset counters
                self.frame_count = 0
                self.processing_times = []
                self.last_performance_update = current_time
                
                logger.debug(f"Performance update: {fps:.1f} FPS, {avg_processing_time:.3f}s avg processing")
                
            except Exception as e:
                logger.error(f"Performance metrics update failed: {e}")
    
    def stop(self):
        """Stop the camera thread."""
        with QMutexLocker(self.mutex):
            self.running = False
        
        self.wait(5000)  # Wait up to 5 seconds for thread to finish
        
        if self.isRunning():
            logger.warning("Camera thread did not stop gracefully, terminating")
            self.terminate()
            self.wait(2000)
    
    def set_processing_enabled(self, enabled: bool):
        """Enable or disable neural network processing."""
        with QMutexLocker(self.mutex):
            self.processing_enabled = enabled
            logger.info(f"Neural network processing {'enabled' if enabled else 'disabled'}")
    
    def set_target_fps(self, fps: int):
        """Set target frame rate."""
        with QMutexLocker(self.mutex):
            self.target_fps = max(1, min(60, fps))
            self.frame_interval = 1.0 / self.target_fps
            logger.info(f"Target FPS set to {self.target_fps}")
    
    def set_max_processing_time(self, max_time: float):
        """Set maximum allowed processing time before frame skipping."""
        with QMutexLocker(self.mutex):
            self.max_processing_time = max(0.1, max_time)
            logger.info(f"Max processing time set to {self.max_processing_time}s")
    
    def get_current_stats(self) -> Dict:
        """Get current thread statistics."""
        with QMutexLocker(self.mutex):
            return {
                'running': self.running,
                'processing_enabled': self.processing_enabled,
                'target_fps': self.target_fps,
                'skip_frames': self.skip_frames,
                'frame_skip_count': self.frame_skip_count,
                'max_processing_time': self.max_processing_time
            }
    
    def save_debug_frame(self, frame: np.ndarray, results: Dict):
        """Save debug information for the current frame."""
        try:
            if self.processor:
                self.processor.save_debug_images(frame, results)
        except Exception as e:
            logger.error(f"Failed to save debug frame: {e}")

class LegacyCameraThread(QThread):
    """
    Legacy camera thread for fallback when neural networks are not available.
    Uses the original card detection system.
    """
    
    frame_ready = pyqtSignal(np.ndarray)
    cards_detected = pyqtSignal(list)
    
    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.camera_index = camera_index
        self.capture = ImageCapture(camera_index)
        
        # Import legacy detector
        from .image_capture import CardDetector
        self.detector = CardDetector(debug_mode=False)
        
        self.running = False
        logger.info(f"Legacy camera thread initialized for camera {camera_index}")
    
    def run(self):
        if not self.capture.initialize_camera():
            return
        
        self.running = True
        
        while self.running:
            frame = self.capture.get_frame()
            if frame is not None:
                self.frame_ready.emit(frame)
                
                # Use legacy detection
                try:
                    detected_cards = self.detector.detect_cards_adaptive(frame)
                    if detected_cards:
                        # Convert to neural network format for compatibility
                        neural_format = []
                        for card_img, contour in detected_cards:
                            x, y, w, h = cv2.boundingRect(contour)
                            detection = {
                                'bbox': [x, y, x + w, y + h],
                                'confidence': 0.8,
                                'class': 0,
                                'center': [x + w/2, y + h/2],
                                'width': w,
                                'height': h
                            }
                            neural_format.append(detection)
                        
                        self.cards_detected.emit(neural_format)
                except Exception as e:
                    logger.error(f"Legacy detection failed: {e}")
            
            self.msleep(33)  # ~30 FPS
    
    def stop(self):
        self.running = False
        self.capture.release_camera()
        self.wait()

def create_camera_thread(camera_index: int = 0, config_path: Optional[str] = None, 
                        use_neural_networks: bool = True) -> QThread:
    """
    Factory function to create appropriate camera thread.
    
    Args:
        camera_index: Camera device index
        config_path: Path to neural network configuration
        use_neural_networks: Whether to use neural network processing
        
    Returns:
        Camera thread instance
    """
    if use_neural_networks:
        try:
            return EnhancedCameraThread(camera_index, config_path)
        except Exception as e:
            logger.warning(f"Failed to create enhanced camera thread: {e}")
            logger.info("Falling back to legacy camera thread")
            return LegacyCameraThread(camera_index)
    else:
        return LegacyCameraThread(camera_index)