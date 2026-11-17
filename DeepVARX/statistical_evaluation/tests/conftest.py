import sys
from pathlib import Path

# Add statistical_evaluation directory to sys.path
_test_dir = Path(__file__).resolve().parent
_stat_eval_dir = _test_dir.parent
if str(_stat_eval_dir) not in sys.path:
    sys.path.insert(0, str(_stat_eval_dir))
