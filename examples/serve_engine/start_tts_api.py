#!/usr/bin/env python3
"""Start script for the Higgs Audio Multi-Speaker TTS API service."""

import uvicorn
import os
import sys

# Add the current directory to the path so we can import the service module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Change to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run the API service
    uvicorn.run("tts_api_service:app", host="0.0.0.0", port=8001, reload=True)