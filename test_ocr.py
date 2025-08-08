#!/usr/bin/env python3
"""
Test script to verify OCR functionality
"""

import sys
import os
import numpy as np
import cv2

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from ocr import OCREngine, CardMatcher

def create_test_image():
    """Create a simple test image with text."""
    # Create a white image
    img = np.ones((100, 400, 3), dtype=np.uint8) * 255
    
    # Add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Lightning Bolt', (10, 50), font, 1, (0, 0, 0), 2)
    
    return img

def test_ocr_engine():
    """Test the OCR engine with a simple image."""
    print("Testing OCR Engine...")
    print("=" * 40)
    
    # Initialize OCR engine
    ocr_engine = OCREngine()
    
    # Create test image
    test_image = create_test_image()
    
    # Save test image for reference
    cv2.imwrite('test_image.jpg', test_image)
    print("Created test image: test_image.jpg")
    
    # Test OCR
    result = ocr_engine.extract_card_name(test_image)
    print(f"OCR Result: '{result}'")
    
    # Test card matching
    print("\nTesting Card Matcher...")
    card_matcher = CardMatcher()
    
    if result:
        match = card_matcher.match_card_name(result)
        if match:
            print(f"Match found: {match}")
        else:
            print("No match found")
    else:
        print("No text extracted to match")
    
    # Test with known card names
    print("\nTesting with known card names...")
    test_names = ["Lightning Bolt", "Pikachu", "Michael Jordan", "Forest"]
    
    for name in test_names:
        match = card_matcher.match_card_name(name)
        print(f"'{name}' -> {match}")

if __name__ == "__main__":
    test_ocr_engine()