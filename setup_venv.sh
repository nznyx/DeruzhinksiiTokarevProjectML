#!/bin/bash
set -e

# Define python version
PYTHON_CMD="python3.12"

# Check if python3.12 is installed
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "$PYTHON_CMD could not be found. Please install Python 3.12."
    exit 1
fi

echo "Creating virtual environment using $PYTHON_CMD..."
rm -rf venv
$PYTHON_CMD -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
fi

echo "Setup complete. To activate run: source venv/bin/activate"

