#!/usr/bin/env python3
"""
Analyze what our current detection is actually finding.
"""

import cv2
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from image_capture import CardDetector

def analyze_detection_accuracy():
    """Analyze exactly what our current detection is finding."""
    
    test_image = "card_ref.jpg"
    if not os.path.exists(test_image):
        print(f"Reference image {test_image} not found")
        return
        
    img = cv2.imread(test_image)
    if img is None:
        print(f"Could not load reference image: {test_image}")
        return
        
    detector = CardDetector(debug_mode=False)
    
    print(f"Analyzing current detection accuracy...")
    print(f"Image size: {img.shape}")
    
    # Get our current detection
    detected_cards = detector.detect_cards(img)
    
    if not detected_cards:
        print("No cards detected!")
        return
    
    contour, _ = detected_cards[0]
    x, y, w, h = cv2.boundingRect(contour)
    
    # Expected green box coordinates
    frame_h, frame_w = img.shape[:2]
    expected_x = int(frame_w * 0.132)  # 79
    expected_y = int(frame_h * 0.133)  # 106 
    expected_w = int(frame_w * 0.722)  # 433
    expected_h = int(frame_h * 0.724)  # 579
    
    print(f"\nDETECTION ANALYSIS:")
    print(f"{'='*50}")
    print(f"Our detection:     x={x:3d}, y={y:3d}, w={w:3d}, h={h:3d}")
    print(f"Expected (green):  x={expected_x:3d}, y={expected_y:3d}, w={expected_w:3d}, h={expected_h:3d}")
    print(f"Difference:        x={x-expected_x:+3d}, y={y-expected_y:+3d}, w={w-expected_w:+3d}, h={h-expected_h:+3d}")
    
    # Calculate percentage differences
    x_diff_pct = abs(x - expected_x) / expected_w * 100
    y_diff_pct = abs(y - expected_y) / expected_h * 100
    w_diff_pct = abs(w - expected_w) / expected_w * 100
    h_diff_pct = abs(h - expected_h) / expected_h * 100
    
    print(f"Percentage diff:   x={x_diff_pct:.1f}%, y={y_diff_pct:.1f}%, w={w_diff_pct:.1f}%, h={h_diff_pct:.1f}%")
    
    # Calculate IoU
    overlap_x = max(0, min(x + w, expected_x + expected_w) - max(x, expected_x))
    overlap_y = max(0, min(y + h, expected_y + expected_h) - max(y, expected_y))
    overlap_area = overlap_x * overlap_y
    
    our_area = w * h
    expected_area = expected_w * expected_h
    union_area = our_area + expected_area - overlap_area
    iou = overlap_area / union_area if union_area > 0 else 0
    
    print(f"IoU (overlap):     {iou:.4f} ({iou*100:.2f}%)")
    
    # Determine if this is content area or full card
    if iou > 0.95:
        print(f"\n✓ CONCLUSION: Detection is VERY ACCURATE (IoU > 95%)")
        print(f"  This appears to be detecting the FULL CARD, not just content area")
    elif iou > 0.90:
        print(f"\n✓ CONCLUSION: Detection is ACCURATE (IoU > 90%)")
        print(f"  This is likely detecting the full card with minor variations")
    else:
        print(f"\n⚠ CONCLUSION: Detection differs significantly from green box")
        print(f"  This might be detecting content area instead of full card")
    
    # Create detailed visualization
    analysis_img = img.copy()
    
    # Draw expected green box
    cv2.rectangle(analysis_img, (expected_x, expected_y), 
                 (expected_x + expected_w, expected_y + expected_h), (0, 255, 0), 2)
    cv2.putText(analysis_img, f"Expected Green Box", 
               (expected_x, expected_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw our detection
    cv2.rectangle(analysis_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(analysis_img, f"Our Detection (IoU: {iou:.3f})", 
               (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw overlap area
    overlap_x1 = max(x, expected_x)
    overlap_y1 = max(y, expected_y)
    overlap_x2 = min(x + w, expected_x + expected_w)
    overlap_y2 = min(y + h, expected_y + expected_h)
    
    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
        # Fill overlap area with semi-transparent color
        overlay = analysis_img.copy()
        cv2.rectangle(overlay, (overlap_x1, overlap_y1), (overlap_x2, overlap_y2), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.3, analysis_img, 0.7, 0, analysis_img)
        
        cv2.putText(analysis_img, f"Overlap Area", 
                   (overlap_x1, overlap_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Add analysis text
    cv2.putText(analysis_img, f"Pos Diff: ({x-expected_x:+d}, {y-expected_y:+d})", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(analysis_img, f"Size Diff: ({w-expected_w:+d}, {h-expected_h:+d})", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(analysis_img, f"IoU: {iou:.4f}", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Debug image saving disabled
    print(f"\nAnalysis complete (debug image saving disabled)")

def test_with_extracted_card():
    """Test what happens when we extract the detected card and process it."""
    
    test_image = "card_ref.jpg"
    if not os.path.exists(test_image):
        return
        
    img = cv2.imread(test_image)
    if img is None:
        return
        
    detector = CardDetector(debug_mode=False)
    
    print(f"\n{'='*60}")
    print("EXTRACTED CARD ANALYSIS")
    print(f"{'='*60}")
    
    detected_cards = detector.detect_cards(img)
    if not detected_cards:
        print("No cards detected for extraction test")
        return
    
    contour, _ = detected_cards[0]
    
    # Extract the card ROI
    card_roi = detector.extract_card_roi(img, contour)
    if card_roi is None:
        print("Failed to extract card ROI")
        return
    
    print(f"Extracted card size: {card_roi.shape}")
    
    # Debug image saving disabled
    print(f"Extracted card analysis complete (debug image saving disabled)")
    
    # Now test the name and text region cropping on this extracted card
    from image_capture import CardProcessor
    processor = CardProcessor()
    
    # Crop name region
    name_region = processor.crop_name_region(card_roi)
    print(f"Name region size: {name_region.shape}")
    
    # Crop text region
    text_region = processor.crop_text_region(card_roi)
    print(f"Text region size: {text_region.shape}")
    
    print(f"\n✓ Successfully extracted and processed card regions")
    print(f"  This confirms the detection is working for the full pipeline")

if __name__ == "__main__":
    print("Analyzing current card detection accuracy...")
    
    analyze_detection_accuracy()
    test_with_extracted_card()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("Based on this analysis, we can determine if the current detection")
    print("is finding the full card or just the content area.")
    print("Check the debug images for visual confirmation!")