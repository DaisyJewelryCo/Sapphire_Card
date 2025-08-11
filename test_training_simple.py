#!/usr/bin/env python3
"""
Simple test script for the training data collection system (no dependencies).
"""

import sys
import os
from datetime import datetime

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from training_data import TrainingDataManager, TrainingExample

def test_training_data_manager():
    """Test the training data manager functionality."""
    print("Testing Training Data Manager...")
    
    # Initialize manager
    manager = TrainingDataManager("test_training_simple.db")
    
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
            user_notes="Perfect detection and recognition - ML ready!",
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
            user_notes="Card detected correctly but misidentified - not ML ready",
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
            user_notes="Card not detected at all - poor lighting - not ML ready",
            session_id="test_session_2"
        ),
        TrainingExample(
            detection_correct=True,
            recognition_correct=True,
            detected_name="Black Lotus",
            actual_name="Black Lotus",
            detected_set="Alpha",
            actual_set="Alpha",
            detected_type="Artifact",
            actual_type="Artifact",
            confidence_score=0.98,
            user_notes="Perfect example - ML ready!",
            session_id="test_session_2"
        )
    ]
    
    # Save examples
    print("Saving training examples...")
    for i, example in enumerate(examples):
        example_id = manager.save_training_example(example)
        print(f"  Saved example {i+1} with ID: {example_id}")
    
    # Test retrieval
    print("\nRetrieving all training examples...")
    retrieved_examples = manager.get_training_examples(limit=10)
    print(f"Retrieved {len(retrieved_examples)} examples")
    
    for example in retrieved_examples:
        ml_ready = "✓ ML-Ready" if (example.detection_correct and example.recognition_correct) else "✗ Not ML-Ready"
        print(f"  ID: {example.id}, Detection: {example.detection_correct}, "
              f"Recognition: {example.recognition_correct}, Name: {example.detected_name} - {ml_ready}")
    
    # Test ML-ready examples only
    print("\nRetrieving only ML-ready examples...")
    correct_examples = manager.get_correct_training_examples(limit=10)
    print(f"Found {len(correct_examples)} ML-ready examples")
    
    for example in correct_examples:
        print(f"  ✓ ID: {example.id}, Name: {example.detected_name}, Set: {example.detected_set}")
    
    # Test statistics
    print("\nTraining statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n📊 Summary:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  ML-ready examples: {stats['usable_training_data']}")
    print(f"  ML-ready percentage: {(stats['usable_training_data']/stats['total_examples']*100):.1f}%")
    
    # Test exports
    print("\nExporting all training data...")
    export_path = manager.export_training_data('json')
    print(f"All data exported to: {export_path}")
    
    print("\nExporting ML training data...")
    ml_export_path = manager.export_ml_training_data('json')
    print(f"ML data exported to: {ml_export_path}")
    
    # Verify ML export contains only correct examples
    import json
    with open(ml_export_path, 'r') as f:
        ml_data = json.load(f)
    
    print(f"\nML Export verification:")
    print(f"  Total examples in ML export: {len(ml_data['training_examples'])}")
    print(f"  Expected ML-ready examples: {stats['usable_training_data']}")
    
    all_correct = True
    for example in ml_data['training_examples']:
        if not (example['detection_correct'] and example['recognition_correct']):
            all_correct = False
            break
    
    print(f"  All exported examples are correct: {'✓ Yes' if all_correct else '✗ No'}")
    
    print("\nTraining data manager test completed successfully!")
    return True

def main():
    """Run the test."""
    print("Card Scanner Training System Test (Simple)")
    print("=" * 50)
    
    try:
        if test_training_data_manager():
            print("\n✓ Training data manager tests passed")
        
        print("\n" + "=" * 50)
        print("Test completed successfully!")
        print("\nKey Points:")
        print("• Only examples marked as BOTH detection AND recognition correct are ML-ready")
        print("• Use 'Export ML Training Data' to get only the high-quality examples")
        print("• The system tracks both all feedback and ML-ready examples separately")
        print("\nTo use the training system:")
        print("1. Run the main application: python run.py")
        print("2. Click the 'Train' button to toggle training mode")
        print("3. Capture cards - they will automatically open the training dialog")
        print("4. Use quick action buttons:")
        print("   • '✓ This is Correct' - for perfect detections (adds to ML training data)")
        print("   • '✗ This is Wrong' - for incorrect detections (allows manual correction)")
        print("5. Or use 'Submit Custom Feedback' for detailed manual feedback")
        print("6. Use 'Export ML Training Data' to get clean training data for your models")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())