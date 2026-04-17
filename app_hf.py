# app_hf.py
# Entry point for Hugging Face Spaces deployment

import os
from dotenv import load_dotenv

load_dotenv()

# Import and run the main app
from app import build_interface

demo = build_interface()

demo.launch(
    server_name = "0.0.0.0",
    server_port = 7860,
    share       = False,
    inbrowser   = False,
)