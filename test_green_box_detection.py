#!/usr/bin/env python3
"""
Test script to compare our card detection with the green box reference from analyze_reference.py
"""

import cv2
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from image_capture import CardDetector

def test_green_box_alignment():
    """Test how well our detection aligns with the green box reference."""
    
    # Test on the reference image
    test_image = "card_ref.jpg"
    if not os.path.exists(test_image):
        print(f"Reference image {test_image} not found")
        return
        
    img = cv2.imread(test_image)
    if img is None:
        print(f"Could not load reference image: {test_image}")
        return
        
    detector = CardDetector()
    detector.set_debug_mode(True)
    
    print(f"Testing green box alignment on: {test_image}")
    print(f"Image size: {img.shape}")
    
    # 1. Create visualization with expected green box areas
    green_box_vis = detector.visualize_green_box_reference(img)
    
    # Save the green box visualization
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    import time
    timestamp = int(time.time())
    
    green_box_path = os.path.join(debug_dir, f"green_box_reference_{timestamp}.jpg")
    cv2.imwrite(green_box_path, green_box_vis)
    print(f"Saved green box reference visualization: {green_box_path}")
    
    # 2. Run our detection
    detected_cards = detector.detect_cards(img)
    print(f"Our detection found: {len(detected_cards)} cards")
    
    # 3. Compare our detection with the green box
    if detected_cards:
        for i, (contour, approx) in enumerate(detected_cards):
            print(f"\nAnalyzing detected card {i+1}:")
            
            # Get bounding rectangle of our detection
            x, y, w, h = cv2.boundingRect(contour)
            our_aspect = w / h if h > 0 else 0
            
            # Calculate expected green box coordinates
            frame_h, frame_w = img.shape[:2]
            expected_x = int(frame_w * 0.132)
            expected_y = int(frame_h * 0.133)
            expected_w = int(frame_w * 0.722)
            expected_h = int(frame_h * 0.724)
            expected_aspect = expected_w / expected_h if expected_h > 0 else 0
            
            print(f"  Our detection: x={x}, y={y}, w={w}, h={h}, aspect={our_aspect:.3f}")
            print(f"  Expected (green box): x={expected_x}, y={expected_y}, w={expected_w}, h={expected_h}, aspect={expected_aspect:.3f}")
            
            # Calculate overlap/alignment metrics
            overlap_x = max(0, min(x + w, expected_x + expected_w) - max(x, expected_x))
            overlap_y = max(0, min(y + h, expected_y + expected_h) - max(y, expected_y))
            overlap_area = overlap_x * overlap_y
            
            our_area = w * h
            expected_area = expected_w * expected_h
            union_area = our_area + expected_area - overlap_area
            
            iou = overlap_area / union_area if union_area > 0 else 0
            
            print(f"  Intersection over Union (IoU): {iou:.3f}")
            print(f"  Position difference: dx={abs(x - expected_x)}, dy={abs(y - expected_y)}")
            print(f"  Size difference: dw={abs(w - expected_w)}, dh={abs(h - expected_h)}")
            
            # Create comparison visualization
            comparison_img = img.copy()
            
            # Draw expected green box
            cv2.rectangle(comparison_img, (expected_x, expected_y), 
                         (expected_x + expected_w, expected_y + expected_h), (0, 255, 0), 3)
            cv2.putText(comparison_img, "Expected (Green Box)", 
                       (expected_x, expected_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw our detection
            cv2.rectangle(comparison_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(comparison_img, "Our Detection", 
                       (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw overlap area
            overlap_x1 = max(x, expected_x)
            overlap_y1 = max(y, expected_y)
            overlap_x2 = min(x + w, expected_x + expected_w)
            overlap_y2 = min(y + h, expected_y + expected_h)
            
            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                cv2.rectangle(comparison_img, (overlap_x1, overlap_y1), 
                             (overlap_x2, overlap_y2), (0, 0, 255), 2)
                cv2.putText(comparison_img, f"Overlap (IoU: {iou:.3f})", 
                           (overlap_x1, overlap_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Save comparison
            comparison_path = os.path.join(debug_dir, f"detection_comparison_{i+1}_{timestamp}.jpg")
            cv2.imwrite(comparison_path, comparison_img)
            print(f"  Saved comparison visualization: {comparison_path}")
            
            # Extract and analyze the card ROI
            card_roi = detector.extract_card_roi(img, contour)
            if card_roi is not None:
                roi_path = os.path.join(debug_dir, f"extracted_card_roi_{i+1}_{timestamp}.jpg")
                cv2.imwrite(roi_path, card_roi)
                print(f"  Saved extracted card ROI: {roi_path}")
                print(f"  Extracted card size: {card_roi.shape}")

def test_detection_accuracy():
    """Test detection accuracy and suggest improvements."""
    
    test_image = "card_ref.jpg"
    if not os.path.exists(test_image):
        print(f"Reference image {test_image} not found")
        return
        
    img = cv2.imread(test_image)
    if img is None:
        return
        
    detector = CardDetector()
    
    print(f"\n{'='*60}")
    print("DETECTION ACCURACY ANALYSIS")
    print(f"{'='*60}")
    
    # Test with different parameters to see if we can get better alignment
    parameter_sets = [
        {"name": "Current (Permissive)", "min_area": 5000, "max_area": 800000, "min_aspect": 0.6, "max_aspect": 1.6},
        {"name": "Tighter Aspect", "min_area": 5000, "max_area": 800000, "min_aspect": 0.8, "max_aspect": 1.2},
        {"name": "Green Box Focused", "min_area": 200000, "max_area": 300000, "min_aspect": 0.9, "max_aspect": 1.1},
        {"name": "Very Permissive", "min_area": 1000, "max_area": 1000000, "min_aspect": 0.5, "max_aspect": 2.0},
    ]
    
    for params in parameter_sets:
        print(f"\nTesting: {params['name']}")
        detector.min_contour_area = params["min_area"]
        detector.max_contour_area = params["max_area"]
        
        detected_cards = detector.detect_cards(img)
        print(f"  Cards detected: {len(detected_cards)}")
        
        if detected_cards:
            contour, _ = detected_cards[0]  # Take first detection
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h if h > 0 else 0
            area = cv2.contourArea(contour)
            
            # Compare with expected green box
            frame_h, frame_w = img.shape[:2]
            expected_area = (frame_w * 0.722) * (frame_h * 0.724)
            
            print(f"  Detection: {w}x{h}, aspect={aspect:.3f}, area={area:.0f}")
            print(f"  Expected area: {expected_area:.0f}, ratio: {area/expected_area:.3f}")

if __name__ == "__main__":
    print("Testing green box alignment...")
    test_green_box_alignment()
    
    print("\n" + "="*80)
    test_detection_accuracy()
    
    print("\nTest complete! Check debug_images directory for visualizations.")