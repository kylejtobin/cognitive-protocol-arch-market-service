#!/usr/bin/env python3
"""Run the Market Service dashboard."""

from src.ui.dashboard import run_dashboard

if __name__ == "__main__":
    print("Starting Market Service...")
    print("AI-Powered Trading Analysis Dashboard")
    print("Press 'q' to quit, 's' to switch symbols")
    print("-" * 50)
    run_dashboard()
