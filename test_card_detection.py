#!/usr/bin/env python3
"""
Test script for improved card detection.
"""

import cv2
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from image_capture import CardDetector

def test_card_detection():
    """Test card detection on available test images."""
    
    # List of test images to try
    test_images = [
        "test_image.jpg",
        "card_ref.jpg",
        "reference_analysis.jpg"
    ]
    
    detector = CardDetector()
    detector.set_debug_mode(True)
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"Skipping {image_path} - file not found")
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing card detection on: {image_path}")
        print(f"{'='*60}")
        
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            continue
            
        print(f"Image loaded: {img.shape}")
        
        # Test detection
        detected_cards = detector.detect_cards(img)
        
        print(f"Detection result: {len(detected_cards)} cards found")
        
        # If cards were detected, try to extract ROI
        for i, (contour, approx) in enumerate(detected_cards):
            print(f"\nProcessing card {i+1}:")
            
            # Extract card ROI
            card_roi = detector.extract_card_roi(img, contour)
            
            if card_roi is not None:
                # Save the extracted card
                output_dir = "debug_images"
                os.makedirs(output_dir, exist_ok=True)
                
                import time
                card_filename = f"extracted_card_{i+1}_{int(time.time())}.jpg"
                card_path = os.path.join(output_dir, card_filename)
                cv2.imwrite(card_path, card_roi)
                print(f"  Saved extracted card: {card_path}")
                print(f"  Card ROI size: {card_roi.shape}")
            else:
                print(f"  Failed to extract ROI for card {i+1}")

def test_parameter_tuning():
    """Test different detection parameters."""
    
    test_image = "card_ref.jpg"  # Use the reference image
    if not os.path.exists(test_image):
        print(f"Reference image {test_image} not found for parameter tuning")
        return
        
    img = cv2.imread(test_image)
    if img is None:
        print(f"Could not load reference image: {test_image}")
        return
        
    detector = CardDetector()
    detector.set_debug_mode(True)
    
    print(f"\n{'='*60}")
    print(f"Parameter tuning on: {test_image}")
    print(f"{'='*60}")
    
    # Test different parameter combinations
    parameter_sets = [
        {"min_area": 3000, "max_area": 1000000, "min_aspect": 0.5, "max_aspect": 2.0},
        {"min_area": 5000, "max_area": 800000, "min_aspect": 0.6, "max_aspect": 1.6},
        {"min_area": 8000, "max_area": 600000, "min_aspect": 0.8, "max_aspect": 1.3},
        {"min_area": 10000, "max_area": 500000, "min_aspect": 0.9, "max_aspect": 1.1},
    ]
    
    for i, params in enumerate(parameter_sets):
        print(f"\n--- Parameter Set {i+1} ---")
        results = detector.tune_detection_parameters(img, **params)
        print(f"Results: {len(results)} cards detected")

if __name__ == "__main__":
    print("Testing improved card detection...")
    test_card_detection()
    
    print("\n" + "="*80)
    print("Testing parameter tuning...")
    test_parameter_tuning()
    
    print("\nTest complete! Check the debug_images directory for visualization results.")