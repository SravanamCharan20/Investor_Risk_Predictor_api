#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies with specific versions
pip install -r requirements.txt

# Make sure the script is executable
chmod +x build.sh 