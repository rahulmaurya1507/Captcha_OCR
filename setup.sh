#!/bin/bash

# Create a virtual environment
python3 -m venv virtvenv

# Activate the virtual environment
virtvenv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Set up Flask app
export FLASK_APP=app.py
export FLASK_ENV=development

# Run Flask app
flask run
