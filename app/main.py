#!/usr/bin/env python3
"""
Card Scanner - Real-time MTG/Pokemon/Sports Card Recognition Tool
Main entry point for the application.
"""

import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication, QMessageBox

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui import main as gui_main
from utils import ConfigManager, DatabaseManager

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import easyocr
    except ImportError:
        missing_deps.append("easyocr")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        from PyQt5 import QtWidgets
    except ImportError:
        missing_deps.append("PyQt5")
    
    try:
        import requests
    except ImportError:
        missing_deps.append("requests")
    
    try:
        from thefuzz import fuzz
    except ImportError:
        missing_deps.append("thefuzz")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        print("Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies using:")
        print("pip install " + " ".join(missing_deps))
        return False
    
    return True

def check_easyocr():
    """Check if EasyOCR is available."""
    try:
        import easyocr
        import torch
        print("✓ EasyOCR and PyTorch are available")
        return True
    except Exception as e:
        print(f"EasyOCR or PyTorch not found or not properly configured: {e}")
        print("\nPlease ensure PyTorch and EasyOCR are installed:")
        print("  pip install torch torchvision easyocr")
        return False

def setup_directories():
    """Create necessary directories."""
    directories = [
        "captured_cards",
        "card_images",
        "card_cache",
        "exports",
        "easyocr_models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def cli_mode():
    """Run in command-line mode for testing."""
    print("Card Scanner CLI Mode")
    print("This mode is for testing and development.")
    
    # Initialize components
    config_manager = ConfigManager()
    db_manager = DatabaseManager()
    
    # Show statistics
    stats = db_manager.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"Total Cards: {stats['total_cards']}")
    print(f"Total Value: ${stats['total_value']:.2f}")
    print(f"Total Batches: {stats['total_batches']}")
    
    if stats['cards_by_type']:
        print("\nCards by Type:")
        for card_type, count in stats['cards_by_type'].items():
            print(f"  {card_type}: {count}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Card Scanner - Real-time Card Recognition")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies and exit")
    parser.add_argument("--setup", action="store_true", help="Setup directories and exit")
    
    args = parser.parse_args()
    
    if args.check_deps:
        print("Checking dependencies...")
        deps_ok = check_dependencies()
        tesseract_ok = check_tesseract()
        
        if deps_ok and tesseract_ok:
            print("All dependencies are available!")
            return 0
        else:
            return 1
    
    if args.setup:
        print("Setting up directories...")
        setup_directories()
        print("Setup complete!")
        return 0
    
    # Check dependencies before starting
    if not check_dependencies():
        return 1
    
    if not check_easyocr():
        print("Warning: EasyOCR not found. OCR functionality will be limited.")
    
    # Setup directories
    setup_directories()
    
    if args.cli:
        cli_mode()
        return 0
    
    # Run GUI application
    try:
        gui_main()
        return 0
    except Exception as e:
        print(f"Error starting GUI application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())