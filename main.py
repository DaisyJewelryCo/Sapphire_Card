#!/usr/bin/env python3
"""
Card Scanner Application Entry Point
"""

import sys
import os

# Add the app directory to Python path
app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app')
sys.path.insert(0, app_dir)

from app.main import main

if __name__ == "__main__":
    sys.exit(main())