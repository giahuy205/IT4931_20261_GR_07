# -*- coding: utf-8 -*-
"""
Main Dashboard Entrypoint - Group 7 (IT4931)
Runs the unified interactive Streamlit application.
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    week4_app = current_dir / "app_week4.py"
    runpy.run_path(str(week4_app), run_name="__main__")

