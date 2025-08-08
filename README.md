# Card Scanner - Real-time Card Recognition Tool

A comprehensive application for real-time scanning and recognition of Magic: The Gathering, Pokemon, and Sports cards using computer vision and Keras-OCR technology.

## Features

### Real-time Card Detection
- Live camera feed with automatic card detection
- Visual bounding boxes around detected cards
- Confidence indicators for detection quality
- Support for multiple cards in frame

### Card Recognition
- Keras-OCR based text extraction from card names
- Deep learning powered text recognition
- Fuzzy matching against card databases
- Support for MTG, Pokemon, and Sports cards
- Automatic card type detection

### Data Management
- SQLite database for storing card information
- Batch organization for grouping related cards
- Export functionality (JSON/CSV formats)
- Image storage with captured card photos

### User Interface
- Modern PyQt5-based GUI
- Dual-panel layout (camera feed + card info)
- Real-time processing with background threads
- Comprehensive card information display
- Batch management tools

### API Integration
- Scryfall API for MTG card data
- Pokemon TCG API support
- Automatic price information retrieval
- Card image downloading

## Installation

### Prerequisites

1. **Python 3.7+**
2. **TensorFlow and Keras-OCR** (installed automatically with requirements)
3. **Sufficient RAM** (at least 4GB recommended for TensorFlow models)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python main.py --check-deps
```

## Usage

### Starting the Application

```bash
python main.py
```

### Command Line Options

```bash
python main.py --help
```

Available options:
- `--cli`: Run in command-line mode for testing
- `--check-deps`: Check dependencies and exit
- `--setup`: Setup directories and exit

### Basic Workflow

1. **Start the application** and ensure your camera is connected
2. **Position cards** in front of the camera - they will be automatically detected
3. **Click "Capture Card"** or enable auto-capture mode
4. **Review the recognized card** information in the right panel
5. **Select or create a batch** to organize your cards
6. **Add cards to batches** for later export
7. **Export batches** when ready to upload to auction platforms

### Batch Management

1. Click "Select Batch" to choose an existing batch or create a new one
2. Captured cards will automatically be added to the selected batch
3. Use the "Export Batch" button to save batch data for auction uploads

## Configuration

The application creates a `config.json` file with customizable settings:

```json
{
  "camera_index": 0,
  "ocr_confidence_threshold": 80,
  "auto_capture_enabled": false,
  "auto_capture_interval": 2.0,
  "image_save_enabled": true,
  "image_save_directory": "captured_cards",
  "cache_directory": "card_cache",
  "database_path": "card_database.db",
  "keras_ocr_models_path": "keras_ocr_models"
}
```

## Directory Structure

```
CameraCapture/
├── app/
│   ├── __init__.py
│   ├── image_capture.py    # Camera and image processing
│   ├── ocr.py             # OCR and text recognition
│   ├── scryfall.py        # API integrations
│   ├── utils.py           # Database and utilities
│   ├── gui.py             # User interface
│   └── main.py            # Application entry point
├── captured_cards/        # Saved card images
├── card_images/          # Downloaded reference images
├── card_cache/           # API response cache
├── exports/              # Exported batch files
├── keras_ocr_models/     # Keras-OCR model cache
├── requirements.txt
├── main.py              # Main entry point
└── README.md
```

## Database Schema

### Cards Table
- `id`: Unique identifier
- `name`: Card name
- `card_type`: MTG/Pokemon/Sports
- `set_code`: Set abbreviation
- `set_name`: Full set name
- `rarity`: Card rarity
- `condition`: Card condition
- `quantity`: Number of copies
- `estimated_value`: Current market value
- `capture_timestamp`: When card was scanned
- `image_path`: Path to captured image
- `data_json`: Full card data from APIs
- `uploaded`: Upload status flag
- `notes`: User notes

### Batches Table
- `id`: Unique identifier
- `name`: Batch name
- `created_at`: Creation timestamp
- `description`: Batch description
- `total_cards`: Number of cards in batch
- `total_value`: Combined estimated value
- `uploaded`: Upload status flag

## API Usage

### Scryfall API (MTG Cards)
- Automatic rate limiting (100ms between requests)
- Fuzzy name matching
- Comprehensive card data retrieval
- Image URL extraction

### Pokemon TCG API
- Card name searching
- Set information
- Price data integration

## Troubleshooting

### Camera Issues
- Ensure camera permissions are granted
- Try different camera indices (0, 1, 2, etc.)
- Check camera resolution settings

### OCR Issues
- Ensure TensorFlow and Keras-OCR are properly installed
- Check lighting conditions for optimal text recognition
- Verify card positioning and focus
- Clean card surfaces for better text recognition
- Allow time for model downloads on first run

### Performance Issues
- Close other camera applications
- Reduce camera resolution in config
- Disable auto-capture for manual control
- Ensure sufficient RAM for TensorFlow models
- Consider using GPU acceleration if available

### Database Issues
- Check file permissions in application directory
- Verify SQLite installation
- Use CLI mode to check database status

## Development

### Adding New Card Types
1. Extend `CardMatcher` class in `ocr.py`
2. Add API integration in `scryfall.py`
3. Update database schema if needed
4. Add UI elements for new card type

### Extending API Support
1. Create new API class in `scryfall.py`
2. Implement standardized data parsing
3. Add to `CardDataManager`
4. Update configuration options

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Scryfall API for MTG card data
- Pokemon TCG API for Pokemon card data
- Keras-OCR for deep learning text recognition
- TensorFlow for machine learning framework
- OpenCV for computer vision
- PyQt5 for the user interface

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include system information and error messages