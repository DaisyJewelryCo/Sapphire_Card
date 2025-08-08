#!/usr/bin/env python3
"""
Test script to verify Card Scanner installation and basic functionality.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from PyQt5.QtWidgets import QApplication
        print("✓ PyQt5 imported successfully")
    except ImportError as e:
        print(f"✗ PyQt5 import failed: {e}")
        return False
    
    try:
        import keras_ocr
        print("✓ Keras-OCR imported successfully")
    except ImportError as e:
        print(f"✗ Keras-OCR import failed: {e}")
        return False
    
    try:
        import tensorflow
        print("✓ TensorFlow imported successfully")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import requests
        print("✓ Requests imported successfully")
    except ImportError as e:
        print(f"✗ Requests import failed: {e}")
        return False
    
    try:
        from thefuzz import fuzz
        print("✓ TheFuzz imported successfully")
    except ImportError as e:
        print(f"✗ TheFuzz import failed: {e}")
        return False
    
    return True

def test_keras_ocr():
    """Test Keras-OCR availability."""
    print("\nTesting Keras-OCR...")
    
    try:
        import keras_ocr
        import tensorflow as tf
        
        # Test basic pipeline creation (this might take a while on first run)
        print("Creating Keras-OCR pipeline (this may take a moment)...")
        pipeline = keras_ocr.pipeline.Pipeline()
        print("✓ Keras-OCR pipeline created successfully")
        
        # Test with a simple image
        import numpy as np
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255  # White image
        predictions = pipeline.recognize([test_image])
        print("✓ Keras-OCR recognition test completed")
        
        return True
    except Exception as e:
        print(f"✗ Keras-OCR test failed: {e}")
        return False

def test_camera():
    """Test camera availability."""
    print("\nTesting camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera working - Frame size: {frame.shape}")
                cap.release()
                return True
            else:
                print("✗ Camera opened but couldn't read frame")
                cap.release()
                return False
        else:
            print("✗ Could not open camera")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_app_modules():
    """Test if app modules can be imported."""
    print("\nTesting app modules...")
    
    # Add app directory to path
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app')
    sys.path.insert(0, app_dir)
    
    try:
        from app.image_capture import ImageCapture, CardDetector, CardProcessor
        print("✓ Image capture modules imported")
    except ImportError as e:
        print(f"✗ Image capture import failed: {e}")
        return False
    
    try:
        from app.ocr import OCREngine, CardMatcher
        print("✓ OCR modules imported")
    except ImportError as e:
        print(f"✗ OCR import failed: {e}")
        return False
    
    try:
        from app.scryfall import ScryfallAPI, CardDataManager
        print("✓ API modules imported")
    except ImportError as e:
        print(f"✗ API import failed: {e}")
        return False
    
    try:
        from app.utils import DatabaseManager, ConfigManager
        print("✓ Utility modules imported")
    except ImportError as e:
        print(f"✗ Utility import failed: {e}")
        return False
    
    return True

def test_database():
    """Test database functionality."""
    print("\nTesting database...")
    
    try:
        # Add app directory to path
        app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app')
        sys.path.insert(0, app_dir)
        
        from app.utils import DatabaseManager
        
        # Create test database
        db = DatabaseManager("test_db.db")
        
        # Test adding a card
        test_card_data = {
            "name": "Test Card",
            "card_type": "MTG",
            "set_code": "TST",
            "set_name": "Test Set",
            "rarity": "common"
        }
        
        card_id = db.add_card(test_card_data)
        print(f"✓ Card added with ID: {card_id}")
        
        # Test retrieving cards
        cards = db.get_cards(limit=1)
        if cards and len(cards) > 0:
            print("✓ Card retrieved from database")
        else:
            print("✗ Could not retrieve card from database")
            return False
        
        # Clean up test database
        os.remove("test_db.db")
        print("✓ Database test completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Card Scanner Installation Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Keras-OCR Test", test_keras_ocr),
        ("Camera Test", test_camera),
        ("App Modules Test", test_app_modules),
        ("Database Test", test_database)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ ALL TESTS PASSED - Installation is ready!")
        print("\nYou can now run the application with:")
        print("python main.py")
    else:
        print("✗ SOME TESTS FAILED - Please check the errors above")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure TensorFlow and Keras-OCR are properly installed")
        print("3. Check camera permissions and connections")
        print("4. Make sure you have sufficient memory for TensorFlow models")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())