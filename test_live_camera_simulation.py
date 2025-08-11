#!/usr/bin/env python3
"""
Test script to simulate live camera conditions and test adaptive detection.
"""

import cv2
import sys
import os
import numpy as np

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from image_capture import CardDetector

def simulate_camera_conditions():
    """Simulate different camera conditions that might affect detection."""
    
    test_image = "card_ref.jpg"
    if not os.path.exists(test_image):
        print(f"Reference image {test_image} not found")
        return
        
    img = cv2.imread(test_image)
    if img is None:
        print(f"Could not load reference image: {test_image}")
        return
        
    detector = CardDetector(debug_mode=True)
    
    print("Testing adaptive detection under simulated camera conditions...")
    
    # Test 1: Original image (should work perfectly)
    print(f"\n{'='*60}")
    print("TEST 1: ORIGINAL IMAGE (BASELINE)")
    print(f"{'='*60}")
    
    original_cards = detector.detect_cards_adaptive(img)
    print(f"Original image: {len(original_cards)} cards detected")
    
    # Test 2: Darker image (simulating poor lighting)
    print(f"\n{'='*60}")
    print("TEST 2: DARKER IMAGE (POOR LIGHTING)")
    print(f"{'='*60}")
    
    dark_img = cv2.convertScaleAbs(img, alpha=0.6, beta=-30)  # Darker
    dark_cards = detector.detect_cards_adaptive(dark_img)
    print(f"Dark image: {len(dark_cards)} cards detected")
    
    # Save dark image for reference
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    import time
    timestamp = int(time.time())
    
    dark_path = os.path.join(debug_dir, f"dark_simulation_{timestamp}.jpg")
    cv2.imwrite(dark_path, dark_img)
    print(f"Saved dark simulation: {dark_path}")
    
    # Test 3: Brighter image (simulating overexposure)
    print(f"\n{'='*60}")
    print("TEST 3: BRIGHTER IMAGE (OVEREXPOSURE)")
    print(f"{'='*60}")
    
    bright_img = cv2.convertScaleAbs(img, alpha=1.3, beta=20)  # Brighter
    bright_cards = detector.detect_cards_adaptive(bright_img)
    print(f"Bright image: {len(bright_cards)} cards detected")
    
    bright_path = os.path.join(debug_dir, f"bright_simulation_{timestamp}.jpg")
    cv2.imwrite(bright_path, bright_img)
    print(f"Saved bright simulation: {bright_path}")
    
    # Test 4: Blurry image (simulating camera focus issues)
    print(f"\n{'='*60}")
    print("TEST 4: BLURRY IMAGE (FOCUS ISSUES)")
    print(f"{'='*60}")
    
    blurry_img = cv2.GaussianBlur(img, (7, 7), 0)
    blurry_cards = detector.detect_cards_adaptive(blurry_img)
    print(f"Blurry image: {len(blurry_cards)} cards detected")
    
    blurry_path = os.path.join(debug_dir, f"blurry_simulation_{timestamp}.jpg")
    cv2.imwrite(blurry_path, blurry_img)
    print(f"Saved blurry simulation: {blurry_path}")
    
    # Test 5: Noisy image (simulating camera sensor noise)
    print(f"\n{'='*60}")
    print("TEST 5: NOISY IMAGE (SENSOR NOISE)")
    print(f"{'='*60}")
    
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    noisy_cards = detector.detect_cards_adaptive(noisy_img)
    print(f"Noisy image: {len(noisy_cards)} cards detected")
    
    noisy_path = os.path.join(debug_dir, f"noisy_simulation_{timestamp}.jpg")
    cv2.imwrite(noisy_path, noisy_img)
    print(f"Saved noisy simulation: {noisy_path}")
    
    # Test 6: Lower resolution (simulating camera resolution)
    print(f"\n{'='*60}")
    print("TEST 6: LOWER RESOLUTION (CAMERA QUALITY)")
    print(f"{'='*60}")
    
    # Resize to simulate lower camera resolution
    h, w = img.shape[:2]
    low_res_img = cv2.resize(img, (w//2, h//2))
    low_res_img = cv2.resize(low_res_img, (w, h))  # Scale back up
    
    low_res_cards = detector.detect_cards_adaptive(low_res_img)
    print(f"Low resolution image: {len(low_res_cards)} cards detected")
    
    low_res_path = os.path.join(debug_dir, f"low_res_simulation_{timestamp}.jpg")
    cv2.imwrite(low_res_path, low_res_img)
    print(f"Saved low resolution simulation: {low_res_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SIMULATION SUMMARY")
    print(f"{'='*60}")
    print(f"Original:      {len(original_cards)} cards")
    print(f"Dark:          {len(dark_cards)} cards")
    print(f"Bright:        {len(bright_cards)} cards")
    print(f"Blurry:        {len(blurry_cards)} cards")
    print(f"Noisy:         {len(noisy_cards)} cards")
    print(f"Low res:       {len(low_res_cards)} cards")
    
    total_tests = 6
    successful_tests = sum([
        len(original_cards) > 0,
        len(dark_cards) > 0,
        len(bright_cards) > 0,
        len(blurry_cards) > 0,
        len(noisy_cards) > 0,
        len(low_res_cards) > 0
    ])
    
    print(f"\nSuccess rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
    
    if successful_tests >= 5:
        print("✓ Adaptive detection is robust across different conditions")
    elif successful_tests >= 3:
        print("⚠ Adaptive detection works in most conditions but may need tuning")
    else:
        print("✗ Adaptive detection needs improvement for camera conditions")

def test_real_time_performance():
    """Test the performance of adaptive detection for real-time use."""
    
    test_image = "card_ref.jpg"
    if not os.path.exists(test_image):
        return
        
    img = cv2.imread(test_image)
    if img is None:
        return
        
    detector = CardDetector(debug_mode=False)  # Disable debug for performance test
    
    print(f"\n{'='*60}")
    print("REAL-TIME PERFORMANCE TEST")
    print(f"{'='*60}")
    
    import time
    
    # Test standard detection
    start_time = time.time()
    for _ in range(10):
        detector.detect_cards(img)
    standard_time = (time.time() - start_time) / 10
    
    # Test adaptive detection
    start_time = time.time()
    for _ in range(10):
        detector.detect_cards_adaptive(img)
    adaptive_time = (time.time() - start_time) / 10
    
    print(f"Standard detection: {standard_time*1000:.1f}ms avg ({1/standard_time:.1f} FPS)")
    print(f"Adaptive detection: {adaptive_time*1000:.1f}ms avg ({1/adaptive_time:.1f} FPS)")
    
    if adaptive_time < 0.033:  # 30 FPS threshold
        print("✓ Adaptive detection is fast enough for real-time use")
    else:
        print("⚠ Adaptive detection may be too slow for smooth real-time use")

if __name__ == "__main__":
    print("Testing adaptive card detection for live camera conditions...")
    
    simulate_camera_conditions()
    test_real_time_performance()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Run the GUI application and check if detection is improved")
    print("2. Look at debug_images/ to see how detection performs under different conditions")
    print("3. The adaptive detection should now handle varying camera conditions better")
    print("4. Debug mode is enabled in GUI to help diagnose live camera issues")
    print("\nTry running the main application now!")