#!/usr/bin/env python3
"""
Test script to compare content-area detection vs full-card detection methods.
"""

import cv2
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from image_capture import CardDetector

def compare_detection_methods():
    """Compare the original method vs the full card outline method."""
    
    test_image = "card_ref.jpg"
    if not os.path.exists(test_image):
        print(f"Reference image {test_image} not found")
        return
        
    img = cv2.imread(test_image)
    if img is None:
        print(f"Could not load reference image: {test_image}")
        return
        
    detector = CardDetector(debug_mode=True)
    
    print(f"Comparing detection methods on: {test_image}")
    print(f"Image size: {img.shape}")
    
    # Method 1: Original edge-based detection (detects content area)
    print(f"\n{'='*60}")
    print("METHOD 1: EDGE-BASED DETECTION (Current)")
    print(f"{'='*60}")
    
    edge_cards = detector.detect_cards(img)
    print(f"Edge-based method detected: {len(edge_cards)} cards")
    
    if edge_cards:
        contour, _ = edge_cards[0]
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect = w / h if h > 0 else 0
        print(f"  Detection: x={x}, y={y}, w={w}, h={h}")
        print(f"  Area: {area:.0f}, Aspect: {aspect:.3f}")
    
    # Method 2: Full card outline detection
    print(f"\n{'='*60}")
    print("METHOD 2: FULL CARD OUTLINE DETECTION (New)")
    print(f"{'='*60}")
    
    outline_cards = detector.detect_full_card_outline(img)
    print(f"Full outline method detected: {len(outline_cards)} cards")
    
    if outline_cards:
        contour, _ = outline_cards[0]
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect = w / h if h > 0 else 0
        print(f"  Detection: x={x}, y={y}, w={w}, h={h}")
        print(f"  Area: {area:.0f}, Aspect: {aspect:.3f}")
    
    # Method 3: Compare with expected green box
    print(f"\n{'='*60}")
    print("COMPARISON WITH GREEN BOX REFERENCE")
    print(f"{'='*60}")
    
    frame_h, frame_w = img.shape[:2]
    expected_x = int(frame_w * 0.132)
    expected_y = int(frame_h * 0.133)
    expected_w = int(frame_w * 0.722)
    expected_h = int(frame_h * 0.724)
    expected_area = expected_w * expected_h
    expected_aspect = expected_w / expected_h
    
    print(f"Expected (Green Box): x={expected_x}, y={expected_y}, w={expected_w}, h={expected_h}")
    print(f"Expected Area: {expected_area:.0f}, Aspect: {expected_aspect:.3f}")
    
    # Calculate which method is closer to the green box
    if edge_cards and outline_cards:
        edge_contour, _ = edge_cards[0]
        edge_x, edge_y, edge_w, edge_h = cv2.boundingRect(edge_contour)
        
        outline_contour, _ = outline_cards[0]
        outline_x, outline_y, outline_w, outline_h = cv2.boundingRect(outline_contour)
        
        # Calculate IoU for both methods
        def calculate_iou(x1, y1, w1, h1, x2, y2, w2, h2):
            # Calculate intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        edge_iou = calculate_iou(edge_x, edge_y, edge_w, edge_h, 
                                expected_x, expected_y, expected_w, expected_h)
        outline_iou = calculate_iou(outline_x, outline_y, outline_w, outline_h,
                                   expected_x, expected_y, expected_w, expected_h)
        
        print(f"\nIoU Comparison:")
        print(f"  Edge-based method IoU: {edge_iou:.3f}")
        print(f"  Full outline method IoU: {outline_iou:.3f}")
        
        if outline_iou > edge_iou:
            print(f"  ✓ Full outline method is closer to green box reference!")
        else:
            print(f"  ✓ Edge-based method is closer to green box reference!")
    
    # Create a comparison visualization
    comparison_img = img.copy()
    
    # Draw expected green box
    cv2.rectangle(comparison_img, (expected_x, expected_y), 
                 (expected_x + expected_w, expected_y + expected_h), (0, 255, 0), 3)
    cv2.putText(comparison_img, "Expected (Green Box)", 
               (expected_x, expected_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw edge-based detection
    if edge_cards:
        edge_contour, _ = edge_cards[0]
        cv2.drawContours(comparison_img, [edge_contour], -1, (255, 0, 0), 3)
        edge_x, edge_y, edge_w, edge_h = cv2.boundingRect(edge_contour)
        cv2.putText(comparison_img, "Edge-based (Content Area)", 
                   (edge_x, edge_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Draw full outline detection
    if outline_cards:
        outline_contour, _ = outline_cards[0]
        cv2.drawContours(comparison_img, [outline_contour], -1, (0, 0, 255), 3)
        outline_x, outline_y, outline_w, outline_h = cv2.boundingRect(outline_contour)
        cv2.putText(comparison_img, "Full Outline", 
                   (outline_x, outline_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save comparison
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    import time
    timestamp = int(time.time())
    
    comparison_path = os.path.join(debug_dir, f"method_comparison_{timestamp}.jpg")
    cv2.imwrite(comparison_path, comparison_img)
    print(f"\nSaved method comparison: {comparison_path}")

def test_hybrid_approach():
    """Test a hybrid approach that combines both methods."""
    
    test_image = "card_ref.jpg"
    if not os.path.exists(test_image):
        return
        
    img = cv2.imread(test_image)
    if img is None:
        return
        
    detector = CardDetector(debug_mode=True)
    
    print(f"\n{'='*60}")
    print("HYBRID APPROACH TEST")
    print(f"{'='*60}")
    
    # Get results from both methods
    edge_cards = detector.detect_cards(img)
    outline_cards = detector.detect_full_card_outline(img)
    
    print(f"Edge method: {len(edge_cards)} cards")
    print(f"Outline method: {len(outline_cards)} cards")
    
    # Choose the best result based on size and position
    if edge_cards and outline_cards:
        edge_contour, _ = edge_cards[0]
        edge_area = cv2.contourArea(edge_contour)
        
        outline_contour, _ = outline_cards[0]
        outline_area = cv2.contourArea(outline_contour)
        
        print(f"Edge area: {edge_area:.0f}")
        print(f"Outline area: {outline_area:.0f}")
        
        # Prefer the larger detection (more likely to be the full card)
        if outline_area > edge_area * 1.2:  # At least 20% larger
            print("✓ Choosing full outline method (larger detection)")
            chosen_method = "outline"
        else:
            print("✓ Choosing edge method (similar or better size)")
            chosen_method = "edge"
    elif outline_cards:
        print("✓ Only outline method found cards")
        chosen_method = "outline"
    elif edge_cards:
        print("✓ Only edge method found cards")
        chosen_method = "edge"
    else:
        print("✗ No cards found by either method")
        chosen_method = None
    
    return chosen_method

if __name__ == "__main__":
    print("Testing full card detection vs content area detection...")
    
    compare_detection_methods()
    test_hybrid_approach()
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    print("1. Check the debug images to see which method better detects the full card")
    print("2. The full outline method should detect the entire card including borders")
    print("3. The edge-based method detects the content area (what you're currently seeing)")
    print("4. We can implement a hybrid approach or switch to the better method")
    print("\nCheck debug_images/ for visual comparisons!")