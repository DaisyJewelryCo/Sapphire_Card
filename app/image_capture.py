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
    def __init__(self, debug_mode=False):
        # Based on reference analysis: green box shows card should be ~72% of image
        # Adjust area thresholds to be more permissive for initial detection
        self.min_contour_area = 5000   # Reduced from 10000
        self.max_contour_area = 800000  # Increased from 500000
        self.debug_mode = debug_mode  # Debug visualization (disabled by default for performance)
        
    def detect_cards(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Detect card regions in the frame using improved algorithm based on reference analysis."""
        # Create debug image for visualization
        debug_frame = frame.copy() if self.debug_mode else None
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques to improve edge detection
        # 1. Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Try multiple Canny thresholds to catch different edge conditions
        edges_list = []
        canny_params = [(30, 100), (50, 150), (75, 200)]  # Multiple threshold pairs
        
        for low, high in canny_params:
            edges = cv2.Canny(blur, low, high)
            edges_list.append(edges)
        
        # Combine all edge images
        combined_edges = np.zeros_like(edges_list[0])
        for edges in edges_list:
            combined_edges = cv2.bitwise_or(combined_edges, edges)
        
        # Morphological operations to close gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_cards = []
        frame_h, frame_w = frame.shape[:2]
        
        # Debug: draw all contours in red
        if self.debug_mode and debug_frame is not None:
            cv2.drawContours(debug_frame, contours, -1, (0, 0, 255), 1)
        
        print(f"Found {len(contours)} total contours")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # More permissive area filtering
            if area < self.min_contour_area:
                continue
            if area > self.max_contour_area:
                continue
                
            # Get bounding rectangle for initial checks
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check minimum size requirements (based on reference: card should be significant portion of frame)
            min_size = min(frame_w, frame_h) * 0.10  # Reduced to 10% for more permissive detection
            if w < min_size or h < min_size:
                continue
            
            # Check aspect ratio - based on reference analysis: card aspect ratio is 0.997
            aspect_ratio = w / h if h > 0 else 0
            
            # More permissive aspect ratio range for initial detection
            if not (0.5 < aspect_ratio < 2.0):
                continue
            
            # Approximate the contour to check if it's roughly rectangular
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Accept contours with 4+ vertices (rectangular-ish shapes)
            if len(approx) >= 4:
                # Additional validation: check if contour is convex-ish
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Cards should be fairly solid shapes (not too concave)
                if solidity > 0.7:
                    detected_cards.append((contour, approx))
                    print(f"Card candidate {len(detected_cards)}: area={area:.0f}, aspect={aspect_ratio:.3f}, size={w}x{h}, solidity={solidity:.3f}")
                    
                    # Debug: draw accepted contours in green
                    if self.debug_mode and debug_frame is not None:
                        cv2.drawContours(debug_frame, [contour], -1, (0, 255, 0), 3)
                        cv2.putText(debug_frame, f"Card {len(detected_cards)}", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save debug image
        if self.debug_mode and debug_frame is not None:
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            import time
            debug_filename = f"card_detection_{int(time.time())}.jpg"
            debug_path = os.path.join(debug_dir, debug_filename)
            cv2.imwrite(debug_path, debug_frame)
            print(f"Saved card detection debug image: {debug_path}")
            
            # Also save the edge detection result
            edge_debug_path = os.path.join(debug_dir, f"edges_{int(time.time())}.jpg")
            cv2.imwrite(edge_debug_path, combined_edges)
            print(f"Saved edge detection debug image: {edge_debug_path}")
        
        print(f"Detected {len(detected_cards)} card candidates")
        return detected_cards
    
    def detect_cards_adaptive(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Adaptive card detection that adjusts to different camera conditions."""
        # First try the standard detection
        standard_results = self.detect_cards(frame)
        
        if standard_results:
            return standard_results
        
        # If no cards found, try with more permissive parameters
        print("Standard detection found no cards, trying adaptive parameters...")
        
        # Save original parameters
        orig_min_area = self.min_contour_area
        orig_max_area = self.max_contour_area
        
        # Try more permissive area thresholds
        self.min_contour_area = 3000   # Even more permissive
        self.max_contour_area = 1000000
        
        # Try detection with adjusted parameters
        adaptive_results = self.detect_cards(frame)
        
        # Restore original parameters
        self.min_contour_area = orig_min_area
        self.max_contour_area = orig_max_area
        
        if adaptive_results:
            print(f"Adaptive detection found {len(adaptive_results)} cards")
            return adaptive_results
        
        # If still no results, try different edge detection approach
        print("Trying alternative edge detection...")
        return self._detect_with_alternative_edges(frame)
    
    def _detect_with_alternative_edges(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Alternative edge detection for difficult lighting conditions."""
        debug_frame = frame.copy() if self.debug_mode else None
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try adaptive threshold approach for varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
        )
        
        # Use morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_cards = []
        frame_h, frame_w = frame.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000 or area > 800000:  # Use permissive thresholds
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Very permissive aspect ratio for difficult conditions
            if 0.5 < aspect_ratio < 2.0:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) >= 4:
                    detected_cards.append((contour, approx))
                    print(f"Alternative detection found card: area={area:.0f}, aspect={aspect_ratio:.3f}")
        
        if self.debug_mode and debug_frame is not None and detected_cards:
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            import time
            timestamp = int(time.time())
            
            for i, (contour, _) in enumerate(detected_cards):
                cv2.drawContours(debug_frame, [contour], -1, (0, 255, 255), 3)
            
            alt_debug_path = os.path.join(debug_dir, f"alternative_detection_{timestamp}.jpg")
            cv2.imwrite(alt_debug_path, debug_frame)
            print(f"Saved alternative detection debug: {alt_debug_path}")
        
        return detected_cards
    
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug visualization."""
        self.debug_mode = enabled
    
    def tune_detection_parameters(self, frame: np.ndarray, 
                                min_area: int = None, 
                                max_area: int = None,
                                min_aspect: float = None,
                                max_aspect: float = None):
        """Tune detection parameters and return results for testing."""
        if min_area is not None:
            self.min_contour_area = min_area
        if max_area is not None:
            self.max_contour_area = max_area
            
        # Store original values for aspect ratio
        original_min_aspect = 0.6
        original_max_aspect = 1.6
        
        # Temporarily modify aspect ratio check if provided
        if min_aspect is not None:
            original_min_aspect = min_aspect
        if max_aspect is not None:
            original_max_aspect = max_aspect
            
        print(f"Testing with parameters: area={self.min_contour_area}-{self.max_contour_area}, aspect={original_min_aspect}-{original_max_aspect}")
        
        # Run detection with current parameters
        results = self.detect_cards(frame)
        
        return results
    
    def detect_full_card_outline(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Alternative method to detect the complete card outline including borders."""
        debug_frame = frame.copy() if self.debug_mode else None
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Use adaptive threshold to find card boundaries
        # This works well for cards against different backgrounds
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert if needed (card should be darker than background typically)
        mean_val = np.mean(gray)
        if mean_val < 128:  # Dark background
            adaptive_thresh = cv2.bitwise_not(adaptive_thresh)
        
        # Method 2: Use morphological operations to clean up and connect card outline
        kernel = np.ones((7, 7), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Method 3: Find contours from the cleaned binary image
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_cards = []
        frame_h, frame_w = frame.shape[:2]
        
        if self.debug_mode and debug_frame is not None:
            cv2.drawContours(debug_frame, contours, -1, (255, 0, 0), 2)
        
        print(f"Full outline method found {len(contours)} contours")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Look for larger areas that could be full cards
            min_card_area = (frame_w * frame_h) * 0.1  # At least 10% of frame
            max_card_area = (frame_w * frame_h) * 0.8  # At most 80% of frame
            
            if area < min_card_area or area > max_card_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio for card-like shapes
            aspect_ratio = w / h if h > 0 else 0
            if not (0.6 < aspect_ratio < 1.6):
                continue
            
            # Check if contour is roughly rectangular
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) >= 4:
                # Check solidity (how "filled" the shape is)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                if solidity > 0.8:  # Should be a solid rectangular shape
                    detected_cards.append((contour, approx))
                    print(f"Full card outline {len(detected_cards)}: area={area:.0f}, aspect={aspect_ratio:.3f}, solidity={solidity:.3f}")
                    
                    if self.debug_mode and debug_frame is not None:
                        cv2.drawContours(debug_frame, [contour], -1, (0, 255, 0), 3)
                        cv2.putText(debug_frame, f"Full Card {len(detected_cards)}", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save debug images
        if self.debug_mode and debug_frame is not None:
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            import time
            timestamp = int(time.time())
            
            debug_path = os.path.join(debug_dir, f"full_outline_detection_{timestamp}.jpg")
            cv2.imwrite(debug_path, debug_frame)
            print(f"Saved full outline debug image: {debug_path}")
            
            # Also save the adaptive threshold result
            thresh_path = os.path.join(debug_dir, f"adaptive_thresh_{timestamp}.jpg")
            cv2.imwrite(thresh_path, adaptive_thresh)
            
            cleaned_path = os.path.join(debug_dir, f"cleaned_outline_{timestamp}.jpg")
            cv2.imwrite(cleaned_path, cleaned)
        
        return detected_cards
    
    def visualize_green_box_reference(self, frame: np.ndarray) -> np.ndarray:
        """Visualize the expected green box area based on reference analysis."""
        # Based on reference analysis: green box is at relative position (0.132, 0.133, 0.722, 0.724)
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate green box coordinates
        green_x = int(frame_w * 0.132)
        green_y = int(frame_h * 0.133) 
        green_w = int(frame_w * 0.722)
        green_h = int(frame_h * 0.724)
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw green box (expected card area)
        cv2.rectangle(vis_frame, (green_x, green_y), (green_x + green_w, green_y + green_h), (0, 255, 0), 3)
        cv2.putText(vis_frame, "Expected Card Area (Green Box)", 
                   (green_x, green_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Also draw the name region (purple box area)
        name_x = int(frame_w * 0.193)
        name_y = int(frame_h * 0.165)
        name_w = int(frame_w * 0.597)
        name_h = int(frame_h * 0.080)
        cv2.rectangle(vis_frame, (name_x, name_y), (name_x + name_w, name_y + name_h), (255, 0, 255), 2)
        cv2.putText(vis_frame, "Name Region", 
                   (name_x, name_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Draw text region (yellow box area)
        text_x = int(frame_w * 0.175)
        text_y = int(frame_h * 0.748)
        text_w = int(frame_w * 0.637)
        text_h = int(frame_h * 0.071)
        cv2.rectangle(vis_frame, (text_x, text_y), (text_x + text_w, text_y + text_h), (0, 255, 255), 2)
        cv2.putText(vis_frame, "Text Region", 
                   (text_x, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return vis_frame
    
    def extract_card_roi(self, frame: np.ndarray, contour: np.ndarray) -> Optional[np.ndarray]:
        """Extract and align a card region from the frame."""
        try:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If we don't get exactly 4 points, try different approximation values
            if len(approx) != 4:
                # Try more aggressive approximation
                for epsilon_factor in [0.03, 0.04, 0.05]:
                    approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
                    if len(approx) == 4:
                        break
                
                # If still not 4 points, use bounding rectangle as fallback
                if len(approx) != 4:
                    print(f"Could not approximate to 4 points (got {len(approx)}), using bounding rectangle")
                    x, y, w, h = cv2.boundingRect(contour)
                    approx = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
                    approx = approx.reshape(-1, 1, 2).astype(np.int32)
            
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
            
            # Ensure minimum dimensions
            if maxWidth < 50 or maxHeight < 50:
                print(f"Card dimensions too small: {maxWidth}x{maxHeight}")
                return None
            
            # Based on reference analysis: green box shows card aspect ratio should be 0.997 (nearly square)
            # Magic cards are typically square-ish when properly oriented
            card_aspect_ratio = maxWidth / maxHeight if maxHeight > 0 else 1.0
            
            print(f"Extracted card dimensions: {maxWidth}x{maxHeight}, aspect ratio: {card_aspect_ratio:.3f}")
            
            # If the card is significantly non-square, we might need to swap dimensions
            # The reference shows cards should be nearly square (aspect ~0.997)
            if card_aspect_ratio < 0.7 or card_aspect_ratio > 1.4:
                print(f"Card aspect ratio {card_aspect_ratio:.3f} is far from expected ~1.0, keeping as-is")
            
            # Keep the detected orientation - don't force width/height swap
                
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype=np.float32)
            
            try:
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
                
                if warped is None or warped.size == 0:
                    print("Perspective transform resulted in empty image")
                    return None
                
                print(f"Successfully extracted card ROI: {warped.shape}")
                return warped
            except Exception as transform_error:
                print(f"Error in perspective transform: {transform_error}")
                return None
            
        except Exception as e:
            print(f"Error extracting card ROI: {e}")
            # Fallback: use simple bounding rectangle
            try:
                print("Attempting fallback extraction using bounding rectangle")
                x, y, w, h = cv2.boundingRect(contour)
                if w > 50 and h > 50:  # Ensure minimum size
                    fallback_roi = frame[y:y+h, x:x+w]
                    print(f"Fallback extraction successful: {fallback_roi.shape}")
                    return fallback_roi
                else:
                    print(f"Fallback extraction failed: dimensions too small ({w}x{h})")
            except Exception as fallback_error:
                print(f"Fallback extraction also failed: {fallback_error}")
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
    
    def crop_name_region(self, card_image: np.ndarray, crop_ratio: Tuple[float, float] = (0.165, 0.245)) -> np.ndarray:
        """Crop the name region from a card image based on reference analysis."""
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
    
    def crop_text_region(self, card_image: np.ndarray, crop_ratio: Tuple[float, float] = (0.748, 0.819)) -> np.ndarray:
        """Crop the extra card information region from a card image based on reference analysis."""
        h, w = card_image.shape[:2]
        y1 = int(h * crop_ratio[0])
        y2 = int(h * crop_ratio[1])
        text_region = card_image[y1:y2, :]
        
        # Save debug image
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        import time
        debug_filename = f"text_region_{int(time.time())}.jpg"
        debug_path = os.path.join(debug_dir, debug_filename)
        cv2.imwrite(debug_path, text_region)
        print(f"Saved text region debug image: {debug_path}")
        
        return text_region
    
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