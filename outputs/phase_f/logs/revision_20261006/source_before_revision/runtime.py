"""Activate only explicitly installed Phase F optional dependencies."""
import os
import sys
from pathlib import Path


def activate_optional_dependencies(root):
    root=Path(root)
    site=root/'outputs/phase_f/optional_envs/numba/site'
    if site.is_dir():
        if str(site) not in sys.path:sys.path.insert(0,str(site))
        paths=os.environ.get('PYTHONPATH','').split(os.pathsep)
        os.environ['PYTHONPATH']=os.pathsep.join(dict.fromkeys([str(root),str(site),*[p for p in paths if p]]))
        cache=root/'outputs/phase_f/cache/numba'
        cache.mkdir(parents=True,exist_ok=True)
        os.environ['NUMBA_CACHE_DIR']=str(cache)
