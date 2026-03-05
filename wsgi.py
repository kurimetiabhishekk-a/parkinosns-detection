# This file is auto-created by PythonAnywhere.
# Replace 'yourusername' with your actual PythonAnywhere username.

import sys
import os

# Path to your project on PythonAnywhere
path = '/home/yourusername/parkisense'
if path not in sys.path:
    sys.path.insert(0, path)

# Set working directory so Flask can find templates/static/models
os.chdir(path)

# Import Flask app
from main import app as application  # noqa
