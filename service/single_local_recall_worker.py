def main():
 from pathlib import Path
 from vreid.single_local_recall import SingleTailPolicy,verified_selection
 root=Path(__file__).resolve().parents[1];verified_selection(root)
 from .single_local_resources import install_windows
 install_windows(root)
 import vreid.refusal_policy_vric_road as base
 base.VricRoadRefusalPolicy=SingleTailPolicy
 from .vric_road_worker import main as run
 run()
if __name__=='__main__':main()
