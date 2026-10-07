"""Boundary test uses poisoned synthetic bytes, never the real holdout."""
import io
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from src.session_data import _safe_rows


def test_boundary_reader_does_not_consume_poisoned_fields(monkeypatch):
    safe=('날짜,시간,15분,30분,45분,60분\n20210101,0,1,2,').encode('utf-8')
    class GuardedBytes(io.BytesIO):
        def read(self,n=-1):
            assert n>=0 and self.tell()+n<=len(safe), 'forbidden tail was touched'
            return super().read(n)
        def readline(self,n=-1):
            assert self.tell()==0, 'full boundary row was read'
            return super().readline(n)
    monkeypatch.setattr(Path,'open',lambda *a,**k:GuardedBytes(safe+b'NON_NUMERIC_POISON'))
    rows,_=_safe_rows(Path('synthetic.csv'),pd.Timestamp('2021-01-01 00:45'))
    assert rows[0]['_safe_minutes']==(15,30)
    assert rows[0]['45분'] is None and rows[0]['60분'] is None
