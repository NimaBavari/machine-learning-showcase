import os
import sys

this_dir = sys.path[0]
data_dir = os.path.join(this_dir, "data")

__all__ = ["data_dir"]
