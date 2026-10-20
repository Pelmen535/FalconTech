"""Explicit expanded-source verifier; earlier CLI profiles remain available."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'var/systematic-ranking-20261017/predict_expanded.py'),
                   run_name='__main__')
