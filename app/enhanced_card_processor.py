"""
Enhanced Card Processor integrating Neural Networks and Advanced OCR
Implements the complete pipeline from section 4 of the technical guide.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
import os
from pathlib import Path
import json
from datetime import datetime

# Import our neural network and OCR modules
from .neural_networks import create_card_detector, EnsembleCardDetector
from .enhanced_ocr import create_ocr_engine, ImagePreprocessor
from .utils import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCardProcessor:
    """
    Enhanced card processor that integrates neural networks and advanced OCR.
    Implements the complete pipeline from the technical guide.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize enhanced card processor.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Initialize components
        self.detector = None
        self.ocr_engine = None
        self.preprocessor = ImagePreprocessor()
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'successful_detections': 0,
            'successful_ocr': 0,
            'processing_times': []
        }
        
        # Initialize neural network detector
        self._initialize_detector()
        
        # Initialize OCR engine
        self._initialize_ocr()
        
        logger.info("Enhanced card processor initialized")
    
    def _initialize_detector(self):
        """Initialize neural network card detector."""
        try:
            detector_config = self.config.get('detector', {})
            detector_type = detector_config.get('type', 'ensemble')
            
            # Create detector based on configuration
            if detector_type == 'ensemble':
                self.detector = create_card_detector(
                    'ensemble',
                    use_yolo=detector_config.get('use_yolo', True),
                    use_maskrcnn=detector_config.get('use_maskrcnn', True),
                    use_jax=detector_config.get('use_jax', False)  # Disabled by default for performance
                )
            else:
                self.detector = create_card_detector(detector_type, **detector_config)
            
            logger.info(f"Initialized {detector_type} detector")
            
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            # Fallback to basic detection
            from .image_capture import CardDetector
            self.detector = CardDetector(debug_mode=False)
            logger.info("Using fallback basic detector")
    
    def _initialize_ocr(self):
        """Initialize OCR engine."""
        try:
            ocr_config = self.config.get('ocr', {})
            ocr_type = ocr_config.get('type', 'ensemble')
            
            self.ocr_engine = create_ocr_engine(ocr_type, **ocr_config)
            logger.info(f"Initialized {ocr_type} OCR engine")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine: {e}")
            # Fallback to basic OCR
            from .ocr import OCREngine
            self.ocr_engine = OCREngine()
            logger.info("Using fallback basic OCR")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame for card detection and recognition.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processing results dictionary
        """
        start_time = datetime.now()
        
        try:
            # Update statistics
            self.stats['total_processed'] += 1
            
            # Detect cards using neural networks
            detections = self._detect_cards_neural(frame)
            
            if not detections:
                return {
                    'success': False,
                    'message': 'No cards detected',
                    'detections': [],
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            self.stats['successful_detections'] += 1
            
            # Process each detected card
            processed_cards = []
            for detection in detections:
                card_result = self._process_single_card(frame, detection)
                if card_result:
                    processed_cards.append(card_result)
            
            if processed_cards:
                self.stats['successful_ocr'] += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['processing_times'].append(processing_time)
            
            return {
                'success': True,
                'detections': detections,
                'cards': processed_cards,
                'processing_time': processing_time,
                'stats': self._get_current_stats()
            }
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return {
                'success': False,
                'message': f'Processing error: {str(e)}',
                'detections': [],
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _detect_cards_neural(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect cards using neural network detector.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Use neural network detector if available
            if hasattr(self.detector, 'detect') and callable(self.detector.detect):
                detections = self.detector.detect(frame)
                logger.debug(f"Neural detector found {len(detections)} cards")
                return detections
            
            # Fallback to basic detection
            elif hasattr(self.detector, 'detect_cards'):
                basic_detections = self.detector.detect_cards(frame)
                # Convert to neural network format
                neural_detections = []
                for card_img, contour in basic_detections:
                    # Calculate bounding box from contour
                    x, y, w, h = cv2.boundingRect(contour)
                    detection = {
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.8,  # Default confidence
                        'class': 0,
                        'center': [x + w/2, y + h/2],
                        'width': w,
                        'height': h,
                        'contour': contour
                    }
                    neural_detections.append(detection)
                
                logger.debug(f"Basic detector found {len(neural_detections)} cards")
                return neural_detections
            
            else:
                logger.warning("No valid detector available")
                return []
                
        except Exception as e:
            logger.error(f"Card detection failed: {e}")
            return []
    
    def _process_single_card(self, frame: np.ndarray, detection: Dict) -> Optional[Dict]:
        """
        Process a single detected card.
        
        Args:
            frame: Original frame
            detection: Detection dictionary
            
        Returns:
            Card processing result or None if failed
        """
        try:
            # Extract card region
            card_image = self._extract_card_region(frame, detection)
            if card_image is None:
                return None
            
            # Apply perspective correction if mask is available
            if 'mask' in detection:
                card_image = self._apply_mask_correction(card_image, detection['mask'])
            else:
                card_image = self._apply_perspective_correction(card_image, detection)
            
            # Extract regions of interest
            regions = self._extract_card_regions(card_image)
            
            # Perform OCR on each region
            ocr_results = {}
            for region_name, region_image in regions.items():
                if region_image is not None:
                    text = self._extract_text_from_region(region_image, region_name)
                    ocr_results[region_name] = text
            
            # Compile card information
            card_info = {
                'detection': detection,
                'card_image': card_image,
                'regions': regions,
                'ocr_results': ocr_results,
                'card_name': ocr_results.get('name', ''),
                'card_text': ocr_results.get('text', ''),
                'mana_cost': ocr_results.get('mana_cost', ''),
                'type_line': ocr_results.get('type', ''),
                'power_toughness': ocr_results.get('power_toughness', '')
            }
            
            return card_info
            
        except Exception as e:
            logger.error(f"Single card processing failed: {e}")
            return None
    
    def _extract_card_region(self, frame: np.ndarray, detection: Dict) -> Optional[np.ndarray]:
        """Extract card region from frame based on detection."""
        try:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Extract region
            card_region = frame[y1:y2, x1:x2]
            
            if card_region.size == 0:
                return None
            
            return card_region
            
        except Exception as e:
            logger.error(f"Card region extraction failed: {e}")
            return None
    
    def _apply_mask_correction(self, card_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply mask-based correction for irregular card shapes.
        Implements Mask R-CNN post-processing from section 4.2.
        """
        try:
            # Resize mask to match card image
            mask_resized = cv2.resize(mask, (card_image.shape[1], card_image.shape[0]))
            
            # Apply mask
            masked_card = cv2.bitwise_and(card_image, card_image, mask=mask_resized.astype(np.uint8))
            
            # Find contour of mask for perspective correction
            contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                return self._apply_contour_correction(masked_card, largest_contour)
            
            return masked_card
            
        except Exception as e:
            logger.error(f"Mask correction failed: {e}")
            return card_image
    
    def _apply_perspective_correction(self, card_image: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Apply perspective correction to card image.
        Implements the approach from section 4.2.
        """
        try:
            # If contour is available, use it
            if 'contour' in detection:
                return self._apply_contour_correction(card_image, detection['contour'])
            
            # Otherwise, use bounding box for basic correction
            h, w = card_image.shape[:2]
            
            # Standard MTG card aspect ratio (approximately 2.5:3.5)
            target_width = 350
            target_height = 250
            
            # Simple resize for now - could be enhanced with perspective transform
            corrected = cv2.resize(card_image, (target_width, target_height))
            
            return corrected
            
        except Exception as e:
            logger.error(f"Perspective correction failed: {e}")
            return card_image
    
    def _apply_contour_correction(self, card_image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """Apply perspective correction using contour."""
        try:
            # Approximate contour to quadrilateral
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                # Get corner points
                corners = approx.reshape(4, 2).astype(np.float32)
                
                # Order corners: top-left, top-right, bottom-right, bottom-left
                corners = self._order_corners(corners)
                
                # Define target rectangle
                target_width = 350
                target_height = 250
                target_corners = np.array([
                    [0, 0],
                    [target_width, 0],
                    [target_width, target_height],
                    [0, target_height]
                ], dtype=np.float32)
                
                # Apply perspective transform
                transform_matrix = cv2.getPerspectiveTransform(corners, target_corners)
                corrected = cv2.warpPerspective(card_image, transform_matrix, (target_width, target_height))
                
                return corrected
            
            # Fallback to simple resize
            return cv2.resize(card_image, (350, 250))
            
        except Exception as e:
            logger.error(f"Contour correction failed: {e}")
            return cv2.resize(card_image, (350, 250))
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners in clockwise order starting from top-left."""
        # Calculate center
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        sorted_corners = sorted(corners, key=angle_from_center)
        
        # Find top-left corner (smallest sum of coordinates)
        sums = [pt[0] + pt[1] for pt in sorted_corners]
        top_left_idx = np.argmin(sums)
        
        # Reorder starting from top-left
        ordered = sorted_corners[top_left_idx:] + sorted_corners[:top_left_idx]
        
        return np.array(ordered, dtype=np.float32)
    
    def _extract_card_regions(self, card_image: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract different regions of the card for OCR.
        Implements ROI extraction from section 4.3.
        """
        try:
            h, w = card_image.shape[:2]
            logger.info(f"Extracting regions from card image of size: {w}x{h}")
            regions = {}
            
            # Name region (top 15% of card)
            name_region = card_image[0:int(0.15*h), :]
            regions['name'] = name_region if name_region.size > 0 else None
            logger.info(f"Name region extracted: {name_region.shape if name_region.size > 0 else 'None'}")
            
            # Mana cost region (top-right corner)
            mana_region = card_image[0:int(0.15*h), int(0.7*w):]
            regions['mana_cost'] = mana_region if mana_region.size > 0 else None
            
            # Type line region (around 50-60% down)
            type_region = card_image[int(0.5*h):int(0.6*h), :]
            regions['type'] = type_region if type_region.size > 0 else None
            
            # Text box region (bottom 30% of card)
            text_region = card_image[int(0.7*h):, :]
            regions['text'] = text_region if text_region.size > 0 else None
            
            # Power/Toughness region (bottom-right corner)
            pt_region = card_image[int(0.85*h):, int(0.8*w):]
            regions['power_toughness'] = pt_region if pt_region.size > 0 else None
            
            return regions
            
        except Exception as e:
            logger.error(f"Region extraction failed: {e}")
            return {}
    
    def _extract_text_from_region(self, region_image: np.ndarray, region_name: str) -> str:
        """Extract text from a specific card region."""
        try:
            if region_image is None or region_image.size == 0:
                return ""
            
            # Preprocess region for OCR
            preprocessed = self.preprocessor.preprocess_for_ocr(region_image)
            
            # Apply region-specific preprocessing
            if region_name == 'name':
                # Enhance name region
                preprocessed = self.preprocessor.enhance_text_regions(preprocessed)
            elif region_name == 'mana_cost':
                # Special handling for mana symbols
                preprocessed = self._preprocess_mana_region(preprocessed)
            
            # Extract text using OCR engine
            logger.info(f"Extracting text from {region_name} region, image shape: {preprocessed.shape}")
            
            if hasattr(self.ocr_engine, 'extract_card_name') and region_name == 'name':
                text = self.ocr_engine.extract_card_name(preprocessed)
                logger.info(f"Card name extracted: '{text}'")
            else:
                text_results = self.ocr_engine.extract_text(preprocessed)
                logger.info(f"OCR results for {region_name}: {len(text_results)} detections")
                for i, result in enumerate(text_results):
                    logger.info(f"  Detection {i}: '{result.get('text', '')}' (confidence: {result.get('confidence', 0)})")
                text = ' '.join([result['text'] for result in text_results])
            
            logger.info(f"Final extracted text for {region_name}: '{text.strip()}'")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text extraction failed for {region_name}: {e}")
            return ""
    
    def _preprocess_mana_region(self, mana_region: np.ndarray) -> np.ndarray:
        """Special preprocessing for mana cost regions."""
        try:
            # Enhance circular mana symbols
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            enhanced = cv2.morphologyEx(mana_region, cv2.MORPH_CLOSE, kernel)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Mana region preprocessing failed: {e}")
            return mana_region
    
    def _get_current_stats(self) -> Dict:
        """Get current processing statistics."""
        if self.stats['processing_times']:
            avg_time = np.mean(self.stats['processing_times'])
            max_time = np.max(self.stats['processing_times'])
            min_time = np.min(self.stats['processing_times'])
        else:
            avg_time = max_time = min_time = 0
        
        return {
            'total_processed': self.stats['total_processed'],
            'successful_detections': self.stats['successful_detections'],
            'successful_ocr': self.stats['successful_ocr'],
            'detection_rate': self.stats['successful_detections'] / max(1, self.stats['total_processed']),
            'ocr_rate': self.stats['successful_ocr'] / max(1, self.stats['successful_detections']),
            'avg_processing_time': avg_time,
            'max_processing_time': max_time,
            'min_processing_time': min_time
        }
    
    def save_debug_images(self, frame: np.ndarray, results: Dict, output_dir: str = "debug_output"):
        """Save debug images for analysis - DISABLED to prevent unwanted file creation."""
        # Debug image saving has been disabled to prevent automatic file creation
        logger.debug("Debug image saving is disabled")
        return
    
    def get_performance_report(self) -> Dict:
        """Generate performance report."""
        stats = self._get_current_stats()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'configuration': {
                'detector_type': self.config.get('detector', {}).get('type', 'unknown'),
                'ocr_type': self.config.get('ocr', {}).get('type', 'unknown')
            },
            'recommendations': []
        }
        
        # Add performance recommendations
        if stats['detection_rate'] < 0.5:
            report['recommendations'].append("Consider adjusting detection thresholds or using ensemble detector")
        
        if stats['ocr_rate'] < 0.7:
            report['recommendations'].append("Consider using ensemble OCR or improving image preprocessing")
        
        if stats['avg_processing_time'] > 1.0:
            report['recommendations'].append("Consider using lighter models for real-time performance")
        
        return report

# Factory function
def create_enhanced_processor(config_path: Optional[str] = None) -> EnhancedCardProcessor:
    """
    Factory function to create enhanced card processor.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Enhanced card processor instance
    """
    return EnhancedCardProcessor(config_path)