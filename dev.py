#!/usr/bin/env python3
"""
Development server with hot reload for the Exo Proxy Server.
This script provides better hot reload configuration for development.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the development server with optimized hot reload."""
    project_root = Path(__file__).parent

    # Ensure we're in the project root
    os.chdir(project_root)

    # Check if virtual environment exists
    venv_path = project_root / "venv" / "bin" / "activate"
    if not venv_path.exists():
        print("❌ Virtual environment not found. Please run setup first.")
        sys.exit(1)

    # Command to run uvicorn with optimized hot reload
    cmd = [
        "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--reload-delay", "0.1",  # Faster reload response
        "--reload-include", "*.py",
        "--reload-include", "*.html",
        "--reload-include", "*.css",
        "--reload-include", "*.js",
        "--reload-include", "*.jinja2",
        "--reload-exclude", "__pycache__",
        "--reload-exclude", "*.pyc",
        "--reload-exclude", ".git",
        "--log-level", "info"
    ]

    print("🚀 Starting Exo Proxy Server with Hot Reload")
    print("=" * 50)
    print("📁 Watching files: *.py, *.html, *.css, *.js, *.jinja2")
    print("🌐 Server: http://localhost:8000")
    print("🔄 Hot reload: ENABLED (0.1s delay)")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)

    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()


