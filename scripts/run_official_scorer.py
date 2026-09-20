"""Run an organizer-supplied evaluator with its documented CLI, preserving evidence.

Arguments after -- are passed unchanged. No inferred scorer flags or substitute metrics.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vreid.artifacts import sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--script',required=True)
    ap.add_argument('--source-url',required=True)
    ap.add_argument('--expected-sha256',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('scorer_args',nargs=argparse.REMAINDER)
    a=ap.parse_args();script=Path(a.script).resolve();out=Path(a.out)
    if not script.is_file():ap.error('official evaluate.py missing; obtain from organizer')
    actual=sha256(script)
    if actual!=a.expected_sha256:ap.error('official script SHA-256 does not match recorded source')
    args=a.scorer_args[1:] if a.scorer_args[:1]==['--'] else a.scorer_args
    start=time.perf_counter()
    import os
    result=subprocess.run([sys.executable,str(script),*args],capture_output=True,text=True,
                          encoding='utf-8',errors='replace',env={**os.environ,'PYTHONUTF8':'1'})
    out.mkdir(parents=True,exist_ok=True)
    (out/'stdout.txt').write_text(result.stdout,encoding='utf-8')
    (out/'stderr.txt').write_text(result.stderr,encoding='utf-8')
    report=dict(source_url=a.source_url,script_sha256=actual,args=args,exit_code=result.returncode,
        seconds=time.perf_counter()-start,score='see organizer output; not inferred from exit code')
    (out/'run.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(result.stdout);print(result.stderr,file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
