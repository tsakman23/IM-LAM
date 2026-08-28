import os
import sys

# Flat modules (utils.py, detector.py, episode.py) live one level up, mirroring
# the authors' masklam-extract-masks-main layout. Put that dir on sys.path so the
# tests can `import utils` etc. regardless of the hyphenated package directory name.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
