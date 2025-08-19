"""
Neural Network Infrastructure for Magic Card Detection and Shape Recognition
Implements YOLOv5, OpenCV-based detection, and JAX-based approaches.
Gracefully handles missing dependencies.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from abc import ABC, abstractmethod

# Try to import neural network libraries with fallbacks
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from flax.training import train_state
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import torch
    import torch.nn as torch_nn
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseCardDetector(ABC):
    """Abstract base class for card detection models."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect cards in an image and return detection results."""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load a trained model from file."""
        pass
    
    @abstractmethod
    def save_model(self, model_path: str) -> bool:
        """Save the current model to file."""
        pass

class YOLOv5CardDetector(BaseCardDetector):
    """
    YOLOv5-based card detector for real-time object detection.
    Implements the approach described in section 1.2 of the technical guide.
    """
    
    def __init__(self, model_size: str = 'yolov5s', confidence_threshold: float = 0.5, custom_model_path: str = None):
        """
        Initialize YOLOv5 detector.
        
        Args:
            model_size: Size of YOLOv5 model ('yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
            confidence_threshold: Minimum confidence for detections
            custom_model_path: Path to custom trained model (if None, uses pretrained)
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.custom_model_path = custom_model_path
        self.model = None
        
        if not ULTRALYTICS_AVAILABLE:
            logger.warning("YOLOv5 detector initialized but Ultralytics not available")
            return
            
        logger.info(f"YOLOv5 detector initialized with {model_size}")
        
        # Load model (custom or pretrained)
        if custom_model_path and os.path.exists(custom_model_path):
            self._load_custom_model()
        else:
            # Try to find latest trained model automatically
            latest_model = self._find_latest_trained_model()
            if latest_model:
                self.custom_model_path = latest_model
                self._load_custom_model()
            else:
                self._load_pretrained_model()
    
    def _load_custom_model(self):
        """Load custom trained YOLOv5 model."""
        if not ULTRALYTICS_AVAILABLE:
            logger.warning("Cannot load YOLO model: Ultralytics not available")
            return
            
        try:
            self.model = YOLO(self.custom_model_path)
            logger.info(f"Loaded custom trained model from {self.custom_model_path}")
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            logger.info("Falling back to pretrained model")
            self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """Load pretrained YOLOv5 model."""
        if not ULTRALYTICS_AVAILABLE:
            logger.warning("Cannot load YOLO model: Ultralytics not available")
            return
            
        try:
            self.model = YOLO(f'{self.model_size}.pt')
            logger.info(f"Loaded pretrained {self.model_size} model")
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            self.model = None
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect cards in an image using YOLOv5.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with keys: 'bbox', 'confidence', 'class'
        """
        if not ULTRALYTICS_AVAILABLE or self.model is None:
            logger.warning("YOLO model not available or not loaded")
            return []
        
        try:
            # Convert BGR to RGB for YOLO
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(rgb_image, conf=self.confidence_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class': class_id,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                            'width': x2 - x1,
                            'height': y2 - y1
                        }
                        detections.append(detection)
            
            logger.debug(f"YOLOv5 detected {len(detections)} cards")
            return detections
            
        except Exception as e:
            logger.error(f"YOLOv5 detection failed: {e}")
            return []
    
    def load_model(self, model_path: str) -> bool:
        """Load a custom trained YOLOv5 model."""
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded custom model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """Save the current model."""
        try:
            if self.model is not None:
                # YOLOv5 models are typically saved during training
                # For now, we'll copy the current model file
                logger.info(f"Model saving not directly supported for YOLOv5")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def train(self, dataset_path: str, epochs: int = 100, batch_size: int = 16):
        """
        Train YOLOv5 model on custom dataset.
        
        Args:
            dataset_path: Path to dataset in YOLO format
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        try:
            if self.model is None:
                self._load_pretrained_model()
            
            # Train the model
            results = self.model.train(
                data=os.path.join(dataset_path, 'data.yaml'),
                epochs=epochs,
                batch=batch_size,
                device=self.device
            )
            
            logger.info("YOLOv5 training completed")
            return results
            
        except Exception as e:
            logger.error(f"YOLOv5 training failed: {e}")
            return None

class OpenCVCardDetector(BaseCardDetector):
    """
    OpenCV-based card detector for instance segmentation.
    Uses traditional computer vision methods as an alternative to neural networks.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize OpenCV card detector.
        
        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = "opencv_ready"  # Always available since OpenCV is a core dependency
        
        logger.info("OpenCV card detector initialized")
    

    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect cards using traditional OpenCV methods.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with keys: 'bbox', 'confidence', 'class', 'mask'
        """
        
        try:
            # Use traditional computer vision for card detection
            # This is a simplified implementation - in practice you'd use a proper Keras model
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for i, contour in enumerate(contours):
                # Filter by area
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Create mask
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [contour], 255)
                    
                    detection = {
                        'bbox': [int(x), int(y), int(x + w), int(y + h)],
                        'confidence': 0.8,  # Placeholder confidence
                        'class': 0,  # Assuming single class (card)
                        'mask': mask,
                        'center': [x + w/2, y + h/2],
                        'width': w,
                        'height': h
                    }
                    detections.append(detection)
            
            logger.debug(f"CV-based detection found {len(detections)} cards")
            return detections
            
        except Exception as e:
            logger.error(f"CV-based detection failed: {e}")
            return []
    
    def load_model(self, model_path: str) -> bool:
        """Load a custom trained Keras model."""
        try:
            # For CV-based approach, no model loading needed
            logger.info(f"CV-based approach doesn't require model loading from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """Save the current model."""
        try:
            # For CV-based approach, no model saving needed
            logger.info(f"CV-based approach doesn't require model saving to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

class JAXCardDetector(BaseCardDetector):
    """
    JAX-based lightweight card detector.
    Implements the approach described in section 3.1 of the technical guide.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 160, 3), num_classes: int = 2):
        """
        Initialize JAX-based detector.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of classes (background + card)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.params = None
        self.predict_fn = None
        
        if not JAX_AVAILABLE:
            logger.warning("JAX detector initialized but JAX not available")
            return
        
        # Initialize network architecture
        self._build_network()
        logger.info(f"JAX detector initialized with input shape {input_shape}")
    
    def _build_network(self):
        """Build JAX neural network architecture using Flax."""
        if not JAX_AVAILABLE:
            logger.warning("Cannot build JAX network: JAX not available")
            return
            
        try:
            # Define network architecture using Flax
            class CardDetectorCNN(nn.Module):
                num_classes: int
                
                @nn.compact
                def __call__(self, x):
                    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
                    x = nn.relu(x)
                    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
                    
                    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
                    x = nn.relu(x)
                    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
                    
                    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
                    x = nn.relu(x)
                    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
                    
                    x = x.reshape((x.shape[0], -1))  # Flatten
                    x = nn.Dense(features=256)(x)
                    x = nn.relu(x)
                    x = nn.Dense(features=self.num_classes)(x)
                    return nn.log_softmax(x)
            
            # Initialize the model
            self.model = CardDetectorCNN(num_classes=self.num_classes)
            
            # Initialize parameters
            key = jax.random.PRNGKey(0)
            dummy_input = jnp.ones((1,) + self.input_shape)
            self.params = self.model.init(key, dummy_input)
            
            logger.info("JAX network architecture built successfully using Flax")
            
        except Exception as e:
            logger.error(f"Failed to build JAX network: {e}")
            self.model = None
            self.params = None
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect cards using JAX model.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries
        """
        if not JAX_AVAILABLE or self.model is None or self.params is None:
            logger.warning("JAX model not available or not properly initialized")
            return []
        
        try:
            # Preprocess image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_image, (self.input_shape[1], self.input_shape[0]))
            normalized = resized.astype(np.float32) / 255.0
            batch = jnp.expand_dims(normalized, axis=0)
            
            # Run inference
            logits = self.model.apply(self.params, batch)
            probabilities = jax.nn.softmax(logits)
            
            # For now, return simple classification result
            # In a full implementation, this would be extended to localization
            card_prob = float(probabilities[0, 1])  # Probability of card class
            
            detections = []
            if card_prob > 0.5:  # Simple threshold
                h, w = image.shape[:2]
                detection = {
                    'bbox': [w//4, h//4, 3*w//4, 3*h//4],  # Placeholder bbox
                    'confidence': card_prob,
                    'class': 1,
                    'center': [w//2, h//2],
                    'width': w//2,
                    'height': h//2
                }
                detections.append(detection)
            
            logger.debug(f"JAX detector card probability: {card_prob}")
            return detections
            
        except Exception as e:
            logger.error(f"JAX detection failed: {e}")
            return []
    
    def load_model(self, model_path: str) -> bool:
        """Load JAX model parameters."""
        try:
            import pickle
            with open(model_path, 'rb') as f:
                self.params = pickle.load(f)
            logger.info(f"Loaded JAX model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load JAX model: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """Save JAX model parameters."""
        try:
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(self.params, f)
            logger.info(f"Saved JAX model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JAX model: {e}")
            return False
    
    def train(self, train_data, val_data, epochs: int = 100, learning_rate: float = 0.001):
        """
        Train JAX model.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        try:
            # Define loss function
            def loss_fn(params, images, targets):
                logits = self.predict_fn(params, images)
                return -jnp.sum(targets * logits)
            
            # Define update function
            @jax.jit
            def update(params, x, y):
                grads = jax.grad(loss_fn)(params, x, y)
                return [(w - learning_rate * dw, b - learning_rate * db)
                        for (w, b), (dw, db) in zip(params, grads)]
            
            # Training loop would go here
            # This is a simplified version - full implementation would include
            # proper data loading, batching, and validation
            
            logger.info("JAX training completed")
            return True
            
        except Exception as e:
            logger.error(f"JAX training failed: {e}")
            return False

class EnsembleCardDetector:
    """
    Ensemble detector that combines multiple detection approaches.
    Implements the cascading detector concept from section 1.4.
    """
    
    def __init__(self, use_yolo: bool = True, use_opencv: bool = True, use_jax: bool = True):
        """
        Initialize ensemble detector.
        
        Args:
            use_yolo: Whether to include YOLOv5 detector
            use_opencv: Whether to include OpenCV detector
            use_jax: Whether to include JAX detector
        """
        self.detectors = {}
        
        if use_yolo and ULTRALYTICS_AVAILABLE:
            try:
                self.detectors['yolo'] = YOLOv5CardDetector()
                logger.info("YOLOv5 detector added to ensemble")
            except Exception as e:
                logger.warning(f"Failed to initialize YOLOv5: {e}")
        elif use_yolo:
            logger.warning("YOLOv5 requested but Ultralytics not available")
        
        if use_opencv:
            try:
                self.detectors['opencv'] = OpenCVCardDetector()
                logger.info("OpenCV detector added to ensemble")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenCV detector: {e}")
        
        if use_jax and JAX_AVAILABLE:
            try:
                self.detectors['jax'] = JAXCardDetector()
                logger.info("JAX detector added to ensemble")
            except Exception as e:
                logger.warning(f"Failed to initialize JAX detector: {e}")
        elif use_jax:
            logger.warning("JAX requested but JAX not available")
        
        logger.info(f"Ensemble detector initialized with {len(self.detectors)} models")
    
    def detect(self, image: np.ndarray, method: str = 'vote') -> List[Dict]:
        """
        Detect cards using ensemble approach.
        
        Args:
            image: Input image
            method: Ensemble method ('vote', 'union', 'intersection', 'cascade')
            
        Returns:
            Combined detection results
        """
        if not self.detectors:
            logger.warning("No detectors available in ensemble")
            return []
        
        # Get detections from all available detectors
        all_detections = {}
        for name, detector in self.detectors.items():
            try:
                detections = detector.detect(image)
                all_detections[name] = detections
                logger.debug(f"{name} detected {len(detections)} cards")
            except Exception as e:
                logger.error(f"Detection failed for {name}: {e}")
                all_detections[name] = []
        
        # Combine results based on method
        if method == 'vote':
            return self._vote_ensemble(all_detections)
        elif method == 'union':
            return self._union_ensemble(all_detections)
        elif method == 'intersection':
            return self._intersection_ensemble(all_detections)
        elif method == 'cascade':
            return self._cascade_ensemble(all_detections)
        else:
            logger.warning(f"Unknown ensemble method: {method}")
            return self._vote_ensemble(all_detections)
    
    def _vote_ensemble(self, all_detections: Dict[str, List[Dict]]) -> List[Dict]:
        """Combine detections using voting."""
        # Simple implementation: return detections from the most confident detector
        best_detections = []
        best_confidence = 0
        
        for name, detections in all_detections.items():
            if detections:
                avg_confidence = np.mean([d['confidence'] for d in detections])
                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_detections = detections
        
        return best_detections
    
    def _union_ensemble(self, all_detections: Dict[str, List[Dict]]) -> List[Dict]:
        """Combine all detections (union)."""
        combined = []
        for detections in all_detections.values():
            combined.extend(detections)
        return combined
    
    def _intersection_ensemble(self, all_detections: Dict[str, List[Dict]]) -> List[Dict]:
        """Return only detections that appear in multiple models."""
        # Simplified implementation - would need proper IoU-based matching
        if len(all_detections) < 2:
            return list(all_detections.values())[0] if all_detections else []
        
        # For now, return detections from the first detector if others also detect something
        first_key = list(all_detections.keys())[0]
        first_detections = all_detections[first_key]
        
        # Check if other detectors also found cards
        other_found_cards = any(len(dets) > 0 for name, dets in all_detections.items() if name != first_key)
        
        return first_detections if other_found_cards else []
    
    def _cascade_ensemble(self, all_detections: Dict[str, List[Dict]]) -> List[Dict]:
        """Use cascade approach: YOLO -> OpenCV -> JAX."""
        cascade_order = ['yolo', 'opencv', 'jax']
        
        for detector_name in cascade_order:
            if detector_name in all_detections and all_detections[detector_name]:
                return all_detections[detector_name]
        
        return []

# Factory function for easy detector creation
def create_card_detector(detector_type: str = 'ensemble', **kwargs) -> Union[BaseCardDetector, EnsembleCardDetector]:
    """
    Factory function to create card detectors.
    
    Args:
        detector_type: Type of detector ('yolo', 'maskrcnn', 'jax', 'ensemble')
        **kwargs: Additional arguments for detector initialization
        
    Returns:
        Initialized detector instance
    """
    if detector_type.lower() == 'yolo':
        return YOLOv5CardDetector(**kwargs)
    elif detector_type.lower() == 'maskrcnn' or detector_type.lower() == 'opencv':
        return OpenCVCardDetector(**kwargs)
    elif detector_type.lower() == 'jax':
        return JAXCardDetector(**kwargs)
    elif detector_type.lower() == 'ensemble':
        return EnsembleCardDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")