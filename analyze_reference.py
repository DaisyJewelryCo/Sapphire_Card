#!/usr/bin/env python3
"""
Analyze the reference image to understand card regions.
"""

import cv2
import numpy as np
import sys
import os

def analyze_reference_image():
    """Analyze the card_ref.jpg to understand the card structure."""
    
    # Load the reference image
    ref_path = "card_ref.jpg"
    if not os.path.exists(ref_path):
        print(f"Reference image {ref_path} not found!")
        return
    
    img = cv2.imread(ref_path)
    if img is None:
        print("Could not load reference image!")
        return
    
    print(f"Reference image loaded: {img.shape}")
    print("Analyzing card structure...")
    
    # Convert to different color spaces to find the colored rectangles
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for purple, green, and yellow rectangles
    color_ranges = {
        'purple': {
            'lower': np.array([120, 50, 50]),
            'upper': np.array([150, 255, 255]),
            'description': 'Name region'
        },
        'green': {
            'lower': np.array([40, 50, 50]),
            'upper': np.array([80, 255, 255]),
            'description': 'Whole card'
        },
        'yellow': {
            'lower': np.array([20, 50, 50]),
            'upper': np.array([30, 255, 255]),
            'description': 'Extra card information'
        }
    }
    
    regions = {}
    
    for color_name, color_info in color_ranges.items():
        # Create mask for this color
        mask = cv2.inRange(hsv, color_info['lower'], color_info['upper'])
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (should be the rectangle)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Calculate relative position (as percentage of image)
            img_h, img_w = img.shape[:2]
            rel_x = x / img_w
            rel_y = y / img_h
            rel_w = w / img_w
            rel_h = h / img_h
            
            regions[color_name] = {
                'absolute': (x, y, w, h),
                'relative': (rel_x, rel_y, rel_w, rel_h),
                'description': color_info['description']
            }
            
            print(f"\n{color_name.upper()} rectangle ({color_info['description']}):")
            print(f"  Absolute position: x={x}, y={y}, w={w}, h={h}")
            print(f"  Relative position: x={rel_x:.3f}, y={rel_y:.3f}, w={rel_w:.3f}, h={rel_h:.3f}")
            print(f"  Crop ratios: top={rel_y:.3f}, bottom={rel_y + rel_h:.3f}")
    
    # Generate code suggestions based on the analysis
    print("\n" + "="*60)
    print("CODE SUGGESTIONS:")
    print("="*60)
    
    if 'purple' in regions:
        purple = regions['purple']['relative']
        print(f"\nFor NAME REGION cropping (purple rectangle):")
        print(f"def crop_name_region(self, card_image, crop_ratio=({purple[1]:.3f}, {purple[1] + purple[3]:.3f})):")
        print(f"    # This will crop from {purple[1]*100:.1f}% to {(purple[1] + purple[3])*100:.1f}% of card height")
    
    if 'yellow' in regions:
        yellow = regions['yellow']['relative']
        print(f"\nFor TEXT REGION cropping (yellow rectangle):")
        print(f"def crop_text_region(self, card_image, crop_ratio=({yellow[1]:.3f}, {yellow[1] + yellow[3]:.3f})):")
        print(f"    # This will crop from {yellow[1]*100:.1f}% to {(yellow[1] + yellow[3])*100:.1f}% of card height")
    
    if 'green' in regions:
        green = regions['green']['relative']
        print(f"\nFor CARD DETECTION (green rectangle shows expected card bounds):")
        print(f"# Card aspect ratio: {green[2]/green[3]:.3f}")
        print(f"# Expected card size in image: {green[2]*100:.1f}% x {green[3]*100:.1f}%")
    
    # Create a visualization
    result_img = img.copy()
    colors_bgr = {
        'purple': (255, 0, 255),
        'green': (0, 255, 0),
        'yellow': (0, 255, 255)
    }
    
    for color_name, region_info in regions.items():
        x, y, w, h = region_info['absolute']
        color = colors_bgr[color_name]
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_img, f"{color_name} ({region_info['description']})", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Save the analysis result
    cv2.imwrite("reference_analysis.jpg", result_img)
    print(f"\nAnalysis visualization saved as 'reference_analysis.jpg'")
    
    return regions

if __name__ == "__main__":
    analyze_reference_image()