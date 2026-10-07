from pathlib import Path
import sys
import math
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from correct_f1 import f1_counts


def test_no_alarm_with_missed_events_has_zero_f1():
    assert f1_counts(0,0,63)==0
    assert f1_counts(0,7,0)==0
    assert f1_counts(0,7,63)==0


def test_only_empty_confusion_is_undefined():
    assert math.isnan(f1_counts(0,0,0))
    assert f1_counts(33,21,108)==66/195
