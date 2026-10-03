from pathlib import Path
from unittest.mock import patch,MagicMock
import numpy as np
import pandas as pd
import pytest
from phase_f.metrics import _cost
from phase_f.models.foundation import require_finetune_mode
from phase_f.run import complete_prediction_cache


def test_shared_inference_cost_not_multiplied_by_horizon():
    frame=pd.DataFrame({'model':['a']*4,'horizon':[4,4,5,5],'fold':[0]*4,
        'inference_seconds':[.2,.3,.2,.3],
        'inference_seconds_run_id':['origin1','origin2','origin1','origin2']})
    assert _cost(frame,'inference_seconds')==.5


def test_lora_cannot_silently_fallback_to_full():
    with patch('phase_f.models.foundation.importlib.util.find_spec',return_value=None):
        with pytest.raises(RuntimeError,match='fallback is forbidden'):
            require_finetune_mode('lora')


def test_orphaned_prediction_is_preserved_for_recompute():
    path=MagicMock(spec=Path)
    path.exists.return_value=True;path.stem='exp';path.suffix='.parquet'
    orphan=MagicMock(spec=Path);orphan.exists.return_value=False
    path.with_name.return_value=orphan
    audit=MagicMock(spec=Path);audit.exists.return_value=False
    with patch('phase_f.run.sha256',return_value='a'*64):
        assert complete_prediction_cache(path,audit) is False
    path.rename.assert_called_once_with(orphan)
