#!/bin/bash

# Build the virtual environment for the project
python -m venv ./particles_cdssm
source ./particles_cdssm/bin/activate
pip install --upgrade pip
pip install -r ../requirements.txt

# Add the project itself to the virtual environment using a .pth file
# This version is portable across Python versions
PROJECT_ROOT=$(cd .. && pwd)
SITE_PACKAGES=$(python -c 'import site; print([p for p in site.getsitepackages() if "site-packages" in p][0])')

echo "$PROJECT_ROOT" > "$SITE_PACKAGES/particles_cdssm.pth"