# -*- coding: utf-8 -*-
"""
Mô-đun VAR Coffee - Chuyển tiếp tới utils.var_model.
"""
import os
import sys

_cur_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_cur_dir, "..", "..", ".."))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from utils.var_model import *
