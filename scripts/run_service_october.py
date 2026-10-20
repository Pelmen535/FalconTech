"""Explicit October service; same launcher and runtime as normal service entry."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_service import main
if __name__=='__main__':
    sys.argv[1:1]=['--backend','october']
    raise SystemExit(main())
