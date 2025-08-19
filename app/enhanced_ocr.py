"""
Enhanced OCR System for Magic Card Text Recognition
Implements advanced OCR approaches from section 5 of the technical guide.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from abc import ABC, abstractmethod
import os

# OCR Engine imports
try:
    import keras_ocr
    KERAS_OCR_AVAILABLE = True
except ImportError:
    KERAS_OCR_AVAILABLE = False
    logger.warning("keras-ocr not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr not available")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.warning("paddleocr not available")

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available")

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow.keras.backend as K
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    import flax.linen as nn
    from flax.training import train_state
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """Extract text from image."""
        pass
    
    @abstractmethod
    def extract_card_name(self, name_region: np.ndarray) -> str:
        """Extract card name from name region."""
        pass

class ImagePreprocessor:
    """
    Advanced image preprocessing for OCR.
    Implements preprocessing techniques from section 5.2.1.
    """
    
    def __init__(self):
        self.debug_mode = False
    
    def preprocess_for_ocr(self, image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """
        Preprocess image for optimal OCR performance.
        
        Args:
            image: Input image
            method: Preprocessing method ('adaptive', 'otsu', 'gaussian', 'morphological')
            
        Returns:
            Preprocessed image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'adaptive':
            return self._adaptive_preprocessing(gray)
        elif method == 'otsu':
            return self._otsu_preprocessing(gray)
        elif method == 'gaussian':
            return self._gaussian_preprocessing(gray)
        elif method == 'morphological':
            return self._morphological_preprocessing(gray)
        else:
            return self._adaptive_preprocessing(gray)
    
    def _adaptive_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive preprocessing pipeline."""
        # Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _otsu_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Otsu thresholding preprocessing."""
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _gaussian_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Gaussian-based preprocessing."""
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive Gaussian thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 4
        )
        
        return binary
    
    def _morphological_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Morphological operations preprocessing."""
        # Noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image using Hough transform.
        Implements deskewing from section 5.2.1.
        """
        try:
            # Convert to binary if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:10]:  # Use first 10 lines
                    angle = theta * 180 / np.pi
                    if angle > 90:
                        angle = angle - 180
                    angles.append(angle)
                
                if angles:
                    avg_angle = np.mean(angles)
                    
                    # Rotate image
                    h, w = image.shape[:2]
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                    deskewed = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                            flags=cv2.INTER_CUBIC, 
                                            borderMode=cv2.BORDER_REPLICATE)
                    
                    return deskewed
            
            return image
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image
    
    def enhance_text_regions(self, image: np.ndarray) -> np.ndarray:
        """Enhance text regions for better OCR."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Text enhancement using morphological operations
        # Horizontal kernel to connect text
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Vertical kernel
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, vertical_kernel)
        
        # Combine
        enhanced = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        
        return enhanced

class KerasOCREngine(BaseOCREngine):
    """
    Keras-OCR based text extraction.
    Implements the approach from section 5.2.2.
    """
    
    def __init__(self):
        self.pipeline = None
        self.preprocessor = ImagePreprocessor()
        
        if KERAS_OCR_AVAILABLE:
            try:
                self.pipeline = keras_ocr.pipeline.Pipeline()
                logger.info("Keras-OCR pipeline initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Keras-OCR: {e}")
                self.pipeline = None
        else:
            logger.warning("Keras-OCR not available")
    
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """Extract text using Keras-OCR."""
        if self.pipeline is None:
            return []
        
        try:
            # Preprocess image
            preprocessed = self.preprocessor.preprocess_for_ocr(image)
            
            # Convert to RGB if needed
            if len(preprocessed.shape) == 2:
                rgb_image = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            
            # Run OCR
            predictions = self.pipeline.recognize([rgb_image])
            
            # Format results
            results = []
            for text, box in predictions[0]:
                if text.strip():
                    # Calculate bounding box
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]
                    
                    bbox = {
                        'x1': int(min(x_coords)),
                        'y1': int(min(y_coords)),
                        'x2': int(max(x_coords)),
                        'y2': int(max(y_coords))
                    }
                    
                    results.append({
                        'text': text,
                        'confidence': 1.0,  # Keras-OCR doesn't provide confidence
                        'bbox': bbox
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Keras-OCR extraction failed: {e}")
            return []
    
    def extract_card_name(self, name_region: np.ndarray) -> str:
        """Extract card name from name region."""
        text_results = self.extract_text(name_region)
        
        if text_results:
            # Combine all text, prioritizing by position (left to right, top to bottom)
            sorted_results = sorted(text_results, key=lambda x: (x['bbox']['y1'], x['bbox']['x1']))
            card_name = ' '.join([result['text'] for result in sorted_results])
            return card_name.strip()
        
        return ""

class EasyOCREngine(BaseOCREngine):
    """EasyOCR based text extraction."""
    
    def __init__(self, languages: List[str] = ['en']):
        self.reader = None
        self.preprocessor = ImagePreprocessor()
        
        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(languages)
                logger.info(f"EasyOCR initialized with languages: {languages}")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self.reader = None
        else:
            logger.warning("EasyOCR not available")
    
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """Extract text using EasyOCR."""
        if self.reader is None:
            return []
        
        try:
            # Preprocess image
            preprocessed = self.preprocessor.preprocess_for_ocr(image)
            
            # Run OCR
            results = self.reader.readtext(preprocessed)
            
            # Format results
            formatted_results = []
            for bbox, text, confidence in results:
                if text.strip() and confidence > 0.1:
                    # Convert bbox format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    formatted_bbox = {
                        'x1': int(min(x_coords)),
                        'y1': int(min(y_coords)),
                        'x2': int(max(x_coords)),
                        'y2': int(max(y_coords))
                    }
                    
                    formatted_results.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': formatted_bbox
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []
    
    def extract_card_name(self, name_region: np.ndarray) -> str:
        """Extract card name from name region."""
        text_results = self.extract_text(name_region)
        
        if text_results:
            # Filter by confidence and combine
            high_conf_results = [r for r in text_results if r['confidence'] > 0.5]
            if high_conf_results:
                sorted_results = sorted(high_conf_results, key=lambda x: (x['bbox']['y1'], x['bbox']['x1']))
                card_name = ' '.join([result['text'] for result in sorted_results])
                return card_name.strip()
        
        return ""

class CustomCRNNEngine(BaseOCREngine):
    """
    Custom CRNN (Convolutional Recurrent Neural Network) for MTG-specific OCR.
    Implements the custom approach from section 5.2.2 using Keras/TensorFlow.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.preprocessor = ImagePreprocessor()
        
        # Character set for MTG cards (letters, numbers, common symbols)
        self.charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,'-/: "
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        if TF_AVAILABLE:
            self._build_model()
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)
        else:
            logger.warning("TensorFlow not available for CRNN")
    
    def _build_model(self):
        """Build CRNN architecture using Keras."""
        if not TF_AVAILABLE:
            return
        
        try:
            self.model = self._create_keras_crnn(
                img_height=32,
                img_width=128,
                num_classes=len(self.charset) + 1,  # +1 for CTC blank
                hidden_size=256
            )
            logger.info("Keras CRNN model built successfully")
        except Exception as e:
            logger.error(f"Failed to build CRNN model: {e}")
            self.model = None
    
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """Extract text using custom CRNN."""
        if self.model is None:
            return []
        
        try:
            # Preprocess image
            preprocessed = self.preprocessor.preprocess_for_ocr(image)
            
            # Resize to model input size
            resized = cv2.resize(preprocessed, (128, 32))
            
            # Convert to tensor and normalize
            tensor = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
            tensor = np.expand_dims(tensor, axis=-1)  # Add channel dimension
            
            # Run inference
            output = self.model.predict(tensor, verbose=0)
                
            # Decode CTC output
            text = self._decode_ctc(output)
            
            if text.strip():
                h, w = image.shape[:2]
                return [{
                    'text': text,
                    'confidence': 0.8,  # Placeholder confidence
                    'bbox': {'x1': 0, 'y1': 0, 'x2': w, 'y2': h}
                }]
            
            return []
            
        except Exception as e:
            logger.error(f"CRNN extraction failed: {e}")
            return []
    
    def extract_card_name(self, name_region: np.ndarray) -> str:
        """Extract card name using CRNN."""
        text_results = self.extract_text(name_region)
        return text_results[0]['text'] if text_results else ""
    
    def _decode_ctc(self, output: np.ndarray) -> str:
        """Decode CTC output to text."""
        try:
            # Get most likely character at each time step
            preds = np.argmax(output, axis=2)
            preds = preds.squeeze(0)  # Remove batch dimension
            
            # Remove duplicates and blanks
            decoded = []
            prev_char = None
            
            for pred in preds:
                char_idx = int(pred)
                if char_idx < len(self.charset) and char_idx != prev_char:
                    decoded.append(self.idx_to_char[char_idx])
                prev_char = char_idx
            
            return ''.join(decoded)
            
        except Exception as e:
            logger.error(f"CTC decoding failed: {e}")
            return ""
    
    def load_model(self, model_path: str) -> bool:
        """Load trained CRNN model."""
        try:
            if self.model is not None:
                self.model.load_weights(model_path)
                logger.info(f"Loaded CRNN model from {model_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load CRNN model: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """Save CRNN model."""
        try:
            if self.model is not None:
                self.model.save_weights(model_path)
                logger.info(f"Saved CRNN model to {model_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to save CRNN model: {e}")
            return False
    
    def _create_keras_crnn(self, img_height: int, img_width: int, num_classes: int, hidden_size: int = 256):
        """Create CRNN architecture using Keras."""
        # Input layer
        input_layer = layers.Input(shape=(img_height, img_width, 1), name='image_input')
        
        # CNN feature extractor
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 1))(x)
        
        x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 1))(x)
        
        # Reshape for RNN
        new_shape = ((img_width // 4), (img_height // 8) * 512)
        x = layers.Reshape(target_shape=new_shape)(x)
        x = layers.Dense(hidden_size, activation='relu')(x)
        
        # RNN layers
        x = layers.Bidirectional(layers.LSTM(hidden_size, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(hidden_size, return_sequences=True))(x)
        
        # Output layer
        output = layers.Dense(num_classes, activation='softmax', name='ctc_output')(x)
        
        # Create model
        model = keras.Model(inputs=input_layer, outputs=output)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model



class EnsembleOCREngine(BaseOCREngine):
    """
    Ensemble OCR engine that combines multiple OCR approaches.
    Implements ensemble strategy for robust text extraction.
    """
    
    def __init__(self):
        self.engines = {}
        self.preprocessor = ImagePreprocessor()
        
        # Initialize available engines
        if KERAS_OCR_AVAILABLE:
            try:
                self.engines['keras'] = KerasOCREngine()
                logger.info("Added Keras-OCR to ensemble")
            except Exception as e:
                logger.warning(f"Failed to add Keras-OCR: {e}")
        
        if EASYOCR_AVAILABLE:
            try:
                self.engines['easy'] = EasyOCREngine()
                logger.info("Added EasyOCR to ensemble")
            except Exception as e:
                logger.warning(f"Failed to add EasyOCR: {e}")
        
        # Add custom CRNN if available
        try:
            self.engines['crnn'] = CustomCRNNEngine()
            logger.info("Added CRNN to ensemble")
        except Exception as e:
            logger.warning(f"Failed to add CRNN: {e}")
        
        logger.info(f"Ensemble OCR initialized with {len(self.engines)} engines")
    
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """Extract text using ensemble approach."""
        all_results = {}
        
        # Get results from all engines
        for name, engine in self.engines.items():
            try:
                results = engine.extract_text(image)
                all_results[name] = results
                logger.debug(f"{name} extracted {len(results)} text regions")
            except Exception as e:
                logger.warning(f"Text extraction failed for {name}: {e}")
                all_results[name] = []
        
        # Combine results using voting
        return self._combine_results(all_results)
    
    def extract_card_name(self, name_region: np.ndarray) -> str:
        """Extract card name using ensemble approach."""
        all_names = {}
        
        # Get names from all engines
        for name, engine in self.engines.items():
            try:
                card_name = engine.extract_card_name(name_region)
                if card_name.strip():
                    all_names[name] = card_name
                    logger.debug(f"{name} extracted name: '{card_name}'")
            except Exception as e:
                logger.warning(f"Name extraction failed for {name}: {e}")
        
        # Select best name using confidence and length heuristics
        return self._select_best_name(all_names)
    
    def _combine_results(self, all_results: Dict[str, List[Dict]]) -> List[Dict]:
        """Combine text extraction results from multiple engines."""
        if not all_results:
            return []
        
        # Simple approach: return results from the engine with most detections
        best_engine = max(all_results.keys(), key=lambda k: len(all_results[k]))
        return all_results[best_engine]
    
    def _select_best_name(self, all_names: Dict[str, str]) -> str:
        """Select the best card name from multiple engines."""
        if not all_names:
            return ""
        
        if len(all_names) == 1:
            return list(all_names.values())[0]
        
        # Prefer longer names (more complete)
        best_name = max(all_names.values(), key=len)
        
        # Could add more sophisticated voting here
        return best_name

# Factory function
def create_ocr_engine(engine_type: str = 'ensemble', **kwargs) -> BaseOCREngine:
    """
    Factory function to create OCR engines.
    
    Args:
        engine_type: Type of OCR engine ('keras', 'easy', 'crnn', 'ensemble')
        **kwargs: Additional arguments
        
    Returns:
        OCR engine instance
    """
    if engine_type.lower() == 'keras':
        return KerasOCREngine(**kwargs)
    elif engine_type.lower() == 'easy':
        return EasyOCREngine(**kwargs)
    elif engine_type.lower() == 'crnn':
        return CustomCRNNEngine(**kwargs)
    elif engine_type.lower() == 'ensemble':
        return EnsembleOCREngine(**kwargs)
    else:
        raise ValueError(f"Unknown OCR engine type: {engine_type}")