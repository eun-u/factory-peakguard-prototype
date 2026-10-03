"""Small synthetic GPU readiness checks; never counted as search trials."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import torch
from phase_f.models import other_foundation as models
from phase_f.registry import config_hash, write_json, now


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('An actual CUDA device is required for GPU readiness checks')
    root=Path.cwd(); prepared=SimpleNamespace(root=root,out=root/'outputs/phase_f')
    report={'started_at':now(),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'synthetic_only':True,'counts_as_search_trials':False,'checks':[],
            'holdout_read':False,'historical_final_artifact_read':False}
    output=prepared.out/'logs/gpu_other_synthetic_smoke_v2.json'
    matrix=np.stack([50+5*np.sin(np.arange(512)/12),40+4*np.cos(np.arange(512)/18)]).astype('float32')
    origins=pd.date_range('2021-01-10',periods=2,freq='15min')
    for kind in models.MODELS:
        spec={'id':f'gpu-smoke-v2-{kind}','kind':kind,'context_length':512,'batch_size':2,
              'synthetic_only':True,'actual_cuda_required':True}
        snapshot,lock=models._snapshot(prepared,kind,spec)
        work=prepared.out/'models'/spec['id']; work.mkdir(parents=True,exist_ok=True)
        request=work/'inputs.npz'
        digest=models._write_request(request,matrix,origins,matrix[0].copy())
        identity=config_hash({'spec':spec,'input_sha256':digest,'snapshot_hash':lock['snapshot_hash']})
        try:
            quantiles,mean,levels=models._worker(prepared,spec,snapshot,request,identity,models._optional_site(prepared,kind))
            assert quantiles.shape==(2,16,9) and np.isfinite(quantiles).all()
            manifest=json.loads((work/'worker/manifest.json').read_text(encoding='utf-8'))
            report['checks'].append({'kind':kind,'status':'passed','shape':list(quantiles.shape),
                                    'inference_seconds':sum(c['forecast_seconds'] for c in manifest['chunks']),
                                    'model_device':manifest['model_device'],'quantile_levels':levels,
                                    'model_revision':lock['revision']})
        except Exception as exc:
            report['checks'].append({'kind':kind,'status':'failed','error_type':type(exc).__name__,
                                    'error':str(exc).replace(str(root),'<repo>')})
        write_json(output,report)
    report['completed_at']=now();report['passed']=all(c['status']=='passed' for c in report['checks'])
    write_json(output,report);print(json.dumps(report,indent=2),flush=True)
    if not report['passed']:raise RuntimeError('Optional GPU smoke failed; inspect per-worker traceback')


if __name__=='__main__':main()
