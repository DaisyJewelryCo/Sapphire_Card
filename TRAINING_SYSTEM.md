# Card Scanner Training System

This document describes the training data collection system integrated into the Card Scanner application.

## Overview

The training system allows you to collect feedback on card detection and recognition accuracy to build a dataset for machine learning improvements. **Only examples you mark as fully correct (both detection and recognition) will be used for training ML models.** This ensures high-quality training data by filtering out incorrect detections and misidentifications.

## Features

### 1. Training Data Collection
- **Detection Feedback**: Mark whether the card was detected correctly in the image
- **Recognition Feedback**: Mark whether the card was identified correctly
- **Correction Data**: Provide the correct card name, set, and type when recognition fails
- **User Notes**: Add additional context about detection/recognition issues
- **Image Storage**: Automatically saves card images with training feedback
- **Session Tracking**: Groups related training examples together

### 2. Training Dialog Interface
- **Real-time Feedback**: Provide immediate feedback on captured cards
- **Training History**: View all previous training examples
- **Statistics Dashboard**: Monitor detection and recognition accuracy over time
- **ML-Ready Data Counter**: Shows how many examples are suitable for ML training
- **Smart Export**: Export all data or only ML-ready correct examples
- **Data Management**: Delete individual examples or clear all data

### 3. Training Mode
- **Toggle Training Mode**: Enable/disable automatic training dialog popup
- **Visual Indicators**: Button changes color to show training mode status
- **Seamless Integration**: Works alongside normal card capture workflow

## How to Use

### Basic Usage

1. **Start the Application**
   ```bash
   python run.py
   ```

2. **Enable Training Mode**
   - Click the blue "Train" button in the camera controls
   - Button turns green and shows "Training ON"
   - Status bar shows "Training mode enabled"

3. **Capture Cards**
   - Use "Capture Card" button or enable "Auto Capture"
   - Training dialog automatically opens after each capture
   - Provide feedback on detection and recognition accuracy

4. **Provide Feedback**
   - Check/uncheck "Card was detected correctly"
   - Check/uncheck "Card was recognized correctly"
   - **Important**: Only examples marked as BOTH detection and recognition correct will be used for ML training
   - Correct any wrong information in the "Actual" fields (this helps even if you mark it as incorrect)
   - Add notes about issues or observations
   - Click "Submit Training Feedback"

### Advanced Usage

#### Access Training Dialog Directly
- Use menu: **Tools > Training Dialog...**
- View training history and statistics
- Export training data
- Manage existing training examples

#### Training Data Export
- **Export Training Data**: Exports all feedback data (correct and incorrect)
- **Export ML Training Data**: Exports only examples marked as fully correct
- Data exported as JSON with metadata and statistics
- ML export includes quality metrics and filtering information

#### Training Statistics
- View accuracy rates for detection and recognition
- Track total number of training examples
- **ML-Ready Examples**: Count of examples suitable for machine learning
- Monitor training sessions over time

## Data Storage

### Database Schema
Training data is stored in SQLite database (`training_data.db`) with the following structure:

```sql
CREATE TABLE training_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    image_path TEXT,
    image_base64 TEXT,
    detection_correct BOOLEAN NOT NULL,
    recognition_correct BOOLEAN NOT NULL,
    detected_name TEXT,
    actual_name TEXT,
    detected_set TEXT,
    actual_set TEXT,
    detected_type TEXT,
    actual_type TEXT,
    confidence_score REAL,
    user_notes TEXT,
    card_region_coords TEXT,
    preprocessing_params TEXT,
    ocr_results TEXT,
    session_id TEXT
);
```

### File Structure
```
training_data/
├── images/           # Full-resolution card images
├── exports/          # Exported training data files
└── training_data.db  # SQLite database
```

## Training Data Format

### JSON Export Format
```json
{
  "metadata": {
    "export_timestamp": "2024-01-15T10:30:00",
    "total_examples": 150,
    "statistics": {
      "total_examples": 150,
      "correct_detections": 142,
      "correct_recognitions": 128,
      "detection_accuracy": 94.7,
      "recognition_accuracy": 85.3,
      "total_sessions": 12
    }
  },
  "examples": [
    {
      "id": 1,
      "timestamp": "2024-01-15T09:15:30",
      "detection_correct": true,
      "recognition_correct": false,
      "detected_name": "Lightning Strike",
      "actual_name": "Lightning Bolt",
      "detected_set": "Core Set 2019",
      "actual_set": "Alpha",
      "confidence_score": 0.78,
      "user_notes": "Similar card name caused confusion",
      "session_id": "session_001"
    }
  ]
}
```

## Machine Learning Integration

The training data can be used to improve the card recognition system:

### Detection Improvement
- Use incorrectly detected cards to improve card detection algorithms
- Analyze preprocessing parameters that led to detection failures
- Train better edge detection and contour finding

### Recognition Improvement
- Use misidentified cards to improve OCR accuracy
- Build better fuzzy matching algorithms
- Train custom models for card name recognition

### Data Analysis
- Identify common failure patterns
- Analyze confidence score thresholds
- Optimize preprocessing parameters

## API Reference

### TrainingDataManager Class

```python
from app.training_data import TrainingDataManager, TrainingExample

# Initialize manager
manager = TrainingDataManager("training_data.db")

# Save training example
example = TrainingExample(
    detection_correct=True,
    recognition_correct=False,
    detected_name="Wrong Name",
    actual_name="Correct Name",
    # ... other fields
)
example_id = manager.save_training_example(example)

# Get statistics
stats = manager.get_statistics()

# Export data
export_path = manager.export_training_data('json')
```

### TrainingDialog Class

```python
from app.training_dialog import TrainingDialog

# Create dialog
dialog = TrainingDialog(parent_window)

# Set card data for training
dialog.set_card_data(card_image, detection_result)

# Show dialog
dialog.show()
```

## Testing

Run the training system test:

```bash
python test_training.py
```

This will:
- Test database operations
- Create sample training data
- Test data export functionality
- Verify GUI integration (if PyQt5 available)

## Troubleshooting

### Common Issues

1. **Training dialog doesn't open**
   - Check if PyQt5 is installed
   - Verify training_dialog.py is in the app directory

2. **Database errors**
   - Check write permissions in the application directory
   - Ensure SQLite3 is available

3. **Image saving fails**
   - Check disk space
   - Verify training_data/images directory exists

4. **Export fails**
   - Check write permissions
   - Ensure training_data/exports directory exists

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export CARD_SCANNER_DEBUG=1
python run.py
```

## Future Enhancements

- **Batch Training**: Process multiple cards in training mode
- **Active Learning**: Suggest cards that need training feedback
- **Model Integration**: Direct integration with machine learning pipelines
- **Cloud Sync**: Synchronize training data across devices
- **Advanced Analytics**: More detailed statistics and visualizations

## Contributing

To contribute to the training system:

1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation
4. Ensure backward compatibility with existing training data

## License

This training system is part of the Card Scanner application and follows the same license terms.