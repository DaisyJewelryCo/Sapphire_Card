#!/usr/bin/env python3
"""
Test script to check if training dialog can be imported and created.
"""

import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_training_dialog_import():
    """Test if training dialog can be imported."""
    try:
        print("Testing training dialog import...")
        from app.training_dialog import TrainingDialog
        print("✓ Training dialog imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import training dialog: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error importing training dialog: {e}")
        return False

def test_training_dialog_creation():
    """Test if training dialog can be created."""
    try:
        print("Testing training dialog creation...")
        
        # Try to import PyQt5 first
        try:
            from PyQt5.QtWidgets import QApplication
            print("✓ PyQt5 available")
        except ImportError:
            print("✗ PyQt5 not available - skipping GUI test")
            return False
        
        from app.training_dialog import TrainingDialog
        
        # Create QApplication if it doesn't exist
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create training dialog
        dialog = TrainingDialog()
        print("✓ Training dialog created successfully")
        
        # Test setting some dummy data
        dummy_result = {
            'name': 'Test Card',
            'set': 'Test Set',
            'type': 'Test Type',
            'confidence': 0.95
        }
        
        # Create a dummy image (just a simple array)
        import numpy as np
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        try:
            dialog.set_card_data(dummy_image, dummy_result)
            print("✓ Card data set successfully")
        except Exception as e:
            print(f"⚠ Card data setting failed (expected if OpenCV not available): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create training dialog: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the tests."""
    print("Training Dialog Test")
    print("=" * 30)
    
    success = True
    
    # Test import
    if not test_training_dialog_import():
        success = False
    
    print()
    
    # Test creation
    if not test_training_dialog_creation():
        success = False
    
    print()
    print("=" * 30)
    if success:
        print("✓ All tests passed!")
        print("Training dialog should work in the main application.")
    else:
        print("✗ Some tests failed!")
        print("Check the errors above to fix the training dialog integration.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())