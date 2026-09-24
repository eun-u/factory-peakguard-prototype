from scripts.run_session_0924 import compare_decisions


def test_selection_record_accepts_csv_roundoff_and_tuple_storage_only():
    original = {"point_model": "baseline", "n": 100, "ci": (1.2, 3.4), "adopted": False}
    roundtrip = {"point_model": "baseline", "n": 100, "ci": [1.2+1e-14, 3.4], "adopted": False}
    result = compare_decisions(original, roundtrip)
    assert result["equal"]
    assert 0 < result["max_abs_difference"] < 1e-10
    for key, value in (("point_model", "new_model"), ("n", 101), ("adopted", True), ("ci", [1.3, 3.4])):
        changed = {**roundtrip, key: value}
        assert not compare_decisions(original, changed)["equal"]
