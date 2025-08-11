#!/usr/bin/env python3
"""
Interactive card detection parameter tuning script.
This script allows you to adjust detection parameters in real-time.
"""

import cv2
import numpy as np
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

from image_capture import ImageCapture

class InteractiveCardDetector:
    def __init__(self):
        # Adjustable parameters
        self.min_contour_area = 5000
        self.max_contour_area = 500000
        self.min_aspect_ratio = 0.4
        self.max_aspect_ratio = 2.5
        self.min_width = 50
        self.min_height = 70
        
        # Edge detection parameters
        self.canny_low = 50
        self.canny_high = 150
        self.blur_kernel = 9
        
    def create_trackbars(self):
        """Create trackbars for parameter adjustment."""
        cv2.namedWindow('Parameters')
        
        # Area thresholds (scaled down for trackbar)
        cv2.createTrackbar('Min Area (x100)', 'Parameters', self.min_contour_area // 100, 500, self.update_min_area)
        cv2.createTrackbar('Max Area (x1000)', 'Parameters', self.max_contour_area // 1000, 1000, self.update_max_area)
        
        # Aspect ratio (scaled by 100)
        cv2.createTrackbar('Min Aspect (x100)', 'Parameters', int(self.min_aspect_ratio * 100), 500, self.update_min_aspect)
        cv2.createTrackbar('Max Aspect (x100)', 'Parameters', int(self.max_aspect_ratio * 100), 500, self.update_max_aspect)
        
        # Size thresholds
        cv2.createTrackbar('Min Width', 'Parameters', self.min_width, 300, self.update_min_width)
        cv2.createTrackbar('Min Height', 'Parameters', self.min_height, 300, self.update_min_height)
        
        # Edge detection parameters
        cv2.createTrackbar('Canny Low', 'Parameters', self.canny_low, 255, self.update_canny_low)
        cv2.createTrackbar('Canny High', 'Parameters', self.canny_high, 255, self.update_canny_high)
        cv2.createTrackbar('Blur Kernel', 'Parameters', self.blur_kernel, 21, self.update_blur_kernel)
    
    def update_min_area(self, val):
        self.min_contour_area = val * 100
    
    def update_max_area(self, val):
        self.max_contour_area = val * 1000
    
    def update_min_aspect(self, val):
        self.min_aspect_ratio = val / 100.0
    
    def update_max_aspect(self, val):
        self.max_aspect_ratio = val / 100.0
    
    def update_min_width(self, val):
        self.min_width = val
    
    def update_min_height(self, val):
        self.min_height = val
    
    def update_canny_low(self, val):
        self.canny_low = val
    
    def update_canny_high(self, val):
        self.canny_high = val
    
    def update_blur_kernel(self, val):
        # Ensure odd kernel size
        self.blur_kernel = val if val % 2 == 1 else val + 1
        if self.blur_kernel < 3:
            self.blur_kernel = 3
    
    def detect_cards(self, frame):
        """Detect cards with current parameters."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply blurring
        blur = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Use adaptive threshold
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find edges
        edged = cv2.Canny(thresh, self.canny_low, self.canny_high)
        
        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_cards = []
        debug_info = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Check area threshold
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
            
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) >= 4:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Check size and aspect ratio
                if (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio and 
                    w > self.min_width and h > self.min_height):
                    detected_cards.append((contour, approx))
                    debug_info.append(f"Card {len(detected_cards)}: {w}x{h}, AR={aspect_ratio:.2f}, Area={area}")
        
        return detected_cards, debug_info, edged

def main():
    """Main function for interactive tuning."""
    print("Interactive Card Detection Parameter Tuning")
    print("Use trackbars to adjust parameters in real-time")
    print("Press 'q' to quit, 's' to save current parameters")
    
    # Initialize camera and detector
    capture = ImageCapture(0)
    detector = InteractiveCardDetector()
    
    if not capture.initialize_camera():
        print("ERROR: Could not initialize camera!")
        return
    
    # Create parameter trackbars
    detector.create_trackbars()
    
    try:
        while True:
            frame = capture.get_frame()
            if frame is None:
                continue
            
            # Detect cards with current parameters
            detected_cards, debug_info, edged = detector.detect_cards(frame)
            
            # Create display frame
            display_frame = frame.copy()
            
            # Draw detected cards
            for i, (contour, approx) in enumerate(detected_cards):
                # Draw contour in green
                cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
                
                # Draw bounding rectangle in blue
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Add label
                cv2.putText(display_frame, f"Card {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add parameter info to frame
            info_text = [
                f"Cards detected: {len(detected_cards)}",
                f"Min/Max Area: {detector.min_contour_area}/{detector.max_contour_area}",
                f"Aspect ratio: {detector.min_aspect_ratio:.2f}-{detector.max_aspect_ratio:.2f}",
                f"Min size: {detector.min_width}x{detector.min_height}",
                f"Canny: {detector.canny_low}-{detector.canny_high}",
                f"Blur kernel: {detector.blur_kernel}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(display_frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Show frames
            cv2.imshow('Card Detection', display_frame)
            cv2.imshow('Edge Detection', edged)
            
            # Print debug info
            if debug_info:
                print(f"\rDetected: {', '.join(debug_info)}", end="")
            else:
                print(f"\rNo cards detected", end="")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current parameters
                params = {
                    'min_contour_area': detector.min_contour_area,
                    'max_contour_area': detector.max_contour_area,
                    'min_aspect_ratio': detector.min_aspect_ratio,
                    'max_aspect_ratio': detector.max_aspect_ratio,
                    'min_width': detector.min_width,
                    'min_height': detector.min_height,
                    'canny_low': detector.canny_low,
                    'canny_high': detector.canny_high,
                    'blur_kernel': detector.blur_kernel
                }
                
                print(f"\nCurrent parameters:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
                print("Parameters saved to console. Copy these to your CardDetector class.")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        capture.release_camera()
        cv2.destroyAllWindows()
        print("\nCamera released and windows closed")

if __name__ == "__main__":
    main()