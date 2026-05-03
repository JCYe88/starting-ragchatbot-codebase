"""
Shared fixtures and path setup for all tests.
Tests are run from backend/ so imports resolve the same way the app does.
"""
import sys
import os

# Ensure the backend package root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
