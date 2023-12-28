@echo off

REM Create a virtual environment
python -m venv virtvenv

REM Activate the virtual environment
.\venv\Scripts\activate

REM Upgrade pip
pip install --upgrade pip

REM Install dependencies from requirements.txt
pip install -r requirements.txt

REM Set up Flask app
set FLASK_APP=app.py
set FLASK_ENV=development

REM Run Flask app
flask run
