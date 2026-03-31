

import sys
import os

path = '/home/yourusername/parkisense'
if path not in sys.path:
    sys.path.insert(0, path)

os.chdir(path)

from main import app as application  # noqa
