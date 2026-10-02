"""Cross-module frozen-contract checks without reading experiment outcomes."""
import ast
import json
from pathlib import Path

from phase_c.data import CORE_FEATURES,GROUPS,HORIZONS
from phase_c.evaluation import MAIN,SIMPLICITY

ROOT=Path(__file__).resolve().parents[1]


def test_exact_supplied_phase_b_feature_manifest_and_preregistered_candidates():
    tree=ast.parse((ROOT/"scripts/phase_b_reference/run_phase_b_temp_probe.py").read_text(encoding="utf-8"))
    probe_groups=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=="GROUPS")
    assert list(CORE_FEATURES)==sum([probe_groups[k] for k in ("G0","G1","G2")],[])
    cfg=json.loads((ROOT/"configs/phase_c.json").read_text())
    assert cfg["horizons"]==list(HORIZONS)==list(range(4,17))
    assert cfg["complexity_order"]==list(SIMPLICITY)
    assert "R1" not in MAIN
    assert cfg["peak_weight"]==2.0 and cfg["ridge"]["alpha"]==100
    assert len(cfg["lgbm"]["candidates"])==3 and len(cfg["tcn"]["candidates"])==4


def test_retrospective_gap_is_rejected_before_any_feature_fit():
    import numpy as np
    import pandas as pd
    import pytest
    from phase_c.data import _prepare_history,build_contexts
    index=pd.date_range("2021-01-01 00:15",periods=96*50,freq="15min")
    frame=pd.DataFrame({"power":100.0,"headcount":10.0,"time_repaired":False},index=index)
    frame.loc[index[96:100],"power"]=0.0
    frame.loc[index[96:100],"headcount"]=np.nan
    clean,_=_prepare_history(frame)
    with pytest.raises(ValueError,match="Retrospective gap"):
        build_contexts(clean)
