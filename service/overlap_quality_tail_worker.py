"""Overlap exact crop descriptors after existing resource admission checks."""
from pathlib import Path
def main():
 from .overlap_quality_tail import verified_selection
 verified_selection(Path(__file__).resolve().parents[1])
 from vreid.overlap_quality import enable
 enable()
 from .quality_tail_worker import main as run
 run()
if __name__=='__main__':main()
