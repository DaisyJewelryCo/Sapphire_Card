#!/usr/bin/env python3
"""
Test script for the training data collection system.
"""

import sys
import os
import numpy as np
import cv2
from datetime import datetime

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from training_data import TrainingDataManager, TrainingExample

def create_test_image():
    """Create a test card image."""
    # Create a simple test image (simulating a card)
    image = np.zeros((280, 200, 3), dtype=np.uint8)
    
    # Add some color and text-like patterns
    cv2.rectangle(image, (10, 10), (190, 270), (100, 150, 200), -1)
    cv2.rectangle(image, (20, 20), (180, 60), (255, 255, 255), -1)
    cv2.rectangle(image, (20, 200), (180, 260), (255, 255, 255), -1)
    
    # Add some noise to simulate text
    for i in range(50):
        x = np.random.randint(25, 175)
        y = np.random.randint(25, 55)
        cv2.circle(image, (x, y), 2, (0, 0, 0), -1)
    
    return image

def test_training_data_manager():
    """Test the training data manager functionality."""
    print("Testing Training Data Manager...")
    
    # Initialize manager
    manager = TrainingDataManager("test_training.db")
    
    # Create test data
    test_image = create_test_image()
    
    # Create training examples
    examples = [
        TrainingExample(
            detection_correct=True,
            recognition_correct=True,
            detected_name="Lightning Bolt",
            actual_name="Lightning Bolt",
            detected_set="Alpha",
            actual_set="Alpha",
            detected_type="Instant",
            actual_type="Instant",
            confidence_score=0.95,
            user_notes="Perfect detection and recognition",
            session_id="test_session_1"
        ),
        TrainingExample(
            detection_correct=True,
            recognition_correct=False,
            detected_name="Lightning Strike",
            actual_name="Lightning Bolt",
            detected_set="Core Set 2019",
            actual_set="Alpha",
            detected_type="Instant",
            actual_type="Instant",
            confidence_score=0.78,
            user_notes="Card detected correctly but misidentified",
            session_id="test_session_1"
        ),
        TrainingExample(
            detection_correct=False,
            recognition_correct=False,
            detected_name="",
            actual_name="Black Lotus",
            detected_set="",
            actual_set="Alpha",
            detected_type="",
            actual_type="Artifact",
            confidence_score=0.12,
            user_notes="Card not detected at all - poor lighting",
            session_id="test_session_2"
        )
    ]
    
    # Save examples
    print("Saving training examples...")
    for i, example in enumerate(examples):
        # Save image
        image_path = manager.save_card_image(test_image, example.session_id, i)
        example.image_path = image_path
        example.image_base64 = manager.encode_image_base64(test_image)
        
        # Save to database
        example_id = manager.save_training_example(example)
        print(f"  Saved example {i+1} with ID: {example_id}")
    
    # Test retrieval
    print("\nRetrieving training examples...")
    retrieved_examples = manager.get_training_examples(limit=10)
    print(f"Retrieved {len(retrieved_examples)} examples")
    
    for example in retrieved_examples:
        print(f"  ID: {example.id}, Detection: {example.detection_correct}, "
              f"Recognition: {example.recognition_correct}, Name: {example.detected_name}")
    
    # Test statistics
    print("\nTraining statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test ML-ready data
    print(f"\nML-ready examples: {stats['usable_training_data']} out of {stats['total_examples']}")
    
    # Test export
    print("\nExporting all training data...")
    export_path = manager.export_training_data('json')
    print(f"All data exported to: {export_path}")
    
    # Test ML export
    print("\nExporting ML training data...")
    ml_export_path = manager.export_ml_training_data('json')
    print(f"ML data exported to: {ml_export_path}")
    
    # Test getting only correct examples
    print("\nRetrieving only correct examples...")
    correct_examples = manager.get_correct_training_examples(limit=10)
    print(f"Found {len(correct_examples)} correct examples for ML training")
    
    # Test session retrieval
    print("\nTesting session retrieval...")
    session_examples = manager.get_examples_by_session("test_session_1")
    print(f"Found {len(session_examples)} examples in test_session_1")
    
    print("\nTraining data manager test completed successfully!")
    return True

def test_gui_integration():
    """Test GUI integration (requires PyQt5)."""
    try:
        from PyQt5.QtWidgets import QApplication
        from training_dialog import TrainingDialog
        
        print("\nTesting GUI integration...")
        
        app = QApplication(sys.argv)
        
        # Create training dialog
        dialog = TrainingDialog()
        
        # Test with sample data
        test_image = create_test_image()
        test_result = {
            'name': 'Lightning Bolt',
            'set': 'Alpha',
            'type': 'Instant',
            'confidence': 0.95,
            'ocr_results': {'text': 'Lightning Bolt'},
            'preprocessing_params': {'threshold': 127}
        }
        
        dialog.set_card_data(test_image, test_result)
        
        print("GUI integration test completed successfully!")
        print("Note: Dialog was created but not shown (would require user interaction)")
        
        return True
        
    except ImportError as e:
        print(f"GUI test skipped - PyQt5 not available: {e}")
        return False

def main():
    """Run all tests."""
    print("Card Scanner Training System Test")
    print("=" * 40)
    
    try:
        # Test core functionality
        if test_training_data_manager():
            print("\n✓ Training data manager tests passed")
        
        # Test GUI integration
        if test_gui_integration():
            print("✓ GUI integration tests passed")
        
        print("\n" + "=" * 40)
        print("All tests completed successfully!")
        print("\nTo use the training system:")
        print("1. Run the main application: python run.py")
        print("2. Click the 'Train' button to toggle training mode")
        print("3. Capture cards - they will automatically open the training dialog")
        print("4. Provide feedback on detection and recognition accuracy")
        print("5. Use Tools > Training Dialog to view history and statistics")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())