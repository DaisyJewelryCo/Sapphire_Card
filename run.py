#!/usr/bin/env python3
"""
Simple launcher script for the Card Scanner application.
This script handles basic setup and launches the GUI.
"""

import sys
import os
import subprocess

def check_virtual_env():
    """Check if we're in a virtual environment."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def install_dependencies():
    """Install dependencies if needed."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies. Please run:")
        print("pip install -r requirements.txt")
        return False

def check_easyocr():
    """Check if EasyOCR is available."""
    try:
        import easyocr
        import torch
        print("✓ EasyOCR and PyTorch are available")
        return True
    except ImportError as e:
        print(f"✗ EasyOCR or PyTorch not found: {e}")
        return False

def main():
    """Main launcher function."""
    print("Card Scanner Launcher")
    print("=" * 30)
    
    # Check if we're in the right directory
    if not os.path.exists('app'):
        print("Error: Please run this script from the CameraCapture directory")
        return 1
    
    # Check virtual environment
    if not check_virtual_env():
        print("Warning: Not running in a virtual environment")
        print("Consider creating one with: python3 -m venv venv && source venv/bin/activate")
    
    # Check dependencies
    try:
        import cv2
        import PyQt5
        import easyocr
        import torch
        import requests
        import numpy
        from thefuzz import fuzz
        print("✓ All Python dependencies are available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        if input("Install dependencies now? (y/n): ").lower() == 'y':
            if not install_dependencies():
                return 1
        else:
            return 1
    
    # Check EasyOCR
    if not check_easyocr():
        print("Warning: EasyOCR not properly configured. OCR functionality will be limited.")
        if input("Continue anyway? (y/n): ").lower() != 'y':
            return 1
    
    # Create necessary directories
    directories = ['captured_cards', 'card_images', 'card_cache', 'exports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("\nStarting Card Scanner...")
    
    # Add app directory to Python path
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app')
    sys.path.insert(0, app_dir)
    
    try:
        # Import and run the GUI
        from app.gui import main as gui_main
        gui_main()
        return 0
    except Exception as e:
        print(f"Error starting application: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed")
        print("2. Check that your camera is not being used by another application")
        print("3. Verify EasyOCR and PyTorch are properly installed")
        print("4. Ensure you have sufficient memory for PyTorch models")
        return 1

if __name__ == "__main__":
    sys.exit(main())