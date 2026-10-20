"""Run the independently ID-checked expanded verifier in its own service store."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))

if __name__ == '__main__':
    os.environ['VREID_RERANK_PROFILE'] = 'current-query-v1'
    os.environ['VREID_REFUSAL_PROFILE'] = 'expanded-source-v1'
    from run_service import main
    sys.argv = [sys.argv[0], '--backend', 'robust', '--device', 'cuda',
                '--data-dir', str(ROOT/'var/service-expanded-source'), '--threads', '2', *sys.argv[1:]]
    raise SystemExit(main())
