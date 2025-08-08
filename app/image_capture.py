import cv2
import numpy as np
from typing import Optional, Tuple, List
import os

class ImageCapture:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.is_capturing = False
        
    def initialize_camera(self) -> bool:
        """Initialize the camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False
            
            # Set camera properties for better quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_capturing = True
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get a single frame from the camera."""
        if not self.cap or not self.is_capturing:
            return None
            
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def release_camera(self):
        """Release the camera resources."""
        if self.cap:
            self.cap.release()
            self.is_capturing = False

class CardDetector:
    def __init__(self):
        self.min_contour_area = 10000
        self.max_contour_area = 500000
        
    def detect_cards(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Detect card regions in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_cards = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                # Approximate the contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) >= 4:  # Likely a rectangular shape
                    detected_cards.append((contour, approx))
        
        return detected_cards
    
    def extract_card_roi(self, frame: np.ndarray, contour: np.ndarray) -> Optional[np.ndarray]:
        """Extract and align a card region from the frame."""
        try:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) != 4:
                return None
            
            # Get perspective transform
            pts = approx.reshape(4, 2).astype(np.float32)
            rect = self._order_points(pts)
            
            # Calculate dimensions
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            
            maxWidth = max(int(widthA), int(widthB))
            maxHeight = max(int(heightA), int(heightB))
            
            # Standard card ratio is approximately 2.5:3.5
            if maxWidth > maxHeight:
                maxWidth, maxHeight = maxHeight, maxWidth
                
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
            
            return warped
            
        except Exception as e:
            print(f"Error extracting card ROI: {e}")
            return None
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points as top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference of coordinates
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left
        
        return rect

class CardProcessor:
    def __init__(self):
        pass
    
    def crop_name_region(self, card_image: np.ndarray, crop_ratio: Tuple[float, float] = (0.05, 0.25)) -> np.ndarray:
        """Crop the name region from a card image."""
        h, w = card_image.shape[:2]
        y1 = int(h * crop_ratio[0])
        y2 = int(h * crop_ratio[1])
        name_region = card_image[y1:y2, :]
        
        # Save debug image
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        import time
        debug_filename = f"name_region_{int(time.time())}.jpg"
        debug_path = os.path.join(debug_dir, debug_filename)
        cv2.imwrite(debug_path, name_region)
        print(f"Saved name region debug image: {debug_path}")
        
        return name_region
    
    def crop_text_region(self, card_image: np.ndarray, crop_ratio: Tuple[float, float] = (0.4, 0.85)) -> np.ndarray:
        """Crop the text region from a card image."""
        h, w = card_image.shape[:2]
        y1 = int(h * crop_ratio[0])
        y2 = int(h * crop_ratio[1])
        return card_image[y1:y2, :]
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply median blur to reduce noise
        blur = cv2.medianBlur(gray, 3)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Save debug image
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        import time
        debug_filename = f"preprocessed_{int(time.time())}.jpg"
        debug_path = os.path.join(debug_dir, debug_filename)
        cv2.imwrite(debug_path, cleaned)
        print(f"Saved preprocessed debug image: {debug_path}")
        
        return cleaned
    
    def save_card_image(self, card_image: np.ndarray, filename: str, output_dir: str = "captured_cards") -> str:
        """Save a card image to disk."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, card_image)
        return filepath