#!/bin/bash
set -e

# Ensure we are in project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./setup_venv.sh first."
    exit 1
fi

source venv/bin/activate

echo "Running data preparation..."
# Run the python script. 
# PYTHONPATH=. ensures that 'src' module can be imported even if we are running a file inside it.
PYTHONPATH=. python src/data/prepare_dataset.py

