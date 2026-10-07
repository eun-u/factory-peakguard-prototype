"""Create portable evidence hashes from the recorded, immutable base commit."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "a1ddbbd", "verification"], cwd=ROOT, text=True).splitlines()
result = {}
for name in files:
    data = subprocess.check_output(["git", "show", "a1ddbbd:"+name], cwd=ROOT)
    if Path(name).suffix in {".py", ".md", ".json", ".csv", ".txt"}:
        data = data.replace(b"\r\n", b"\n")
    result[name] = hashlib.sha256(data).hexdigest()
(ROOT/"configs/verification_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
