"""
tests/conftest.py
=================
Shared pytest fixtures and configuration.
"""
import sys, os
# Make the project root importable from any test file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
