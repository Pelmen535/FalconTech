"""Guarded existing worker with original global encoders and extra patch encoder."""
def main():
 from vreid.local_recall import LocalTailPolicy
 import vreid.refusal_policy_vric_road as base
 base.VricRoadRefusalPolicy=LocalTailPolicy
 from .vric_road_worker import main as run
 run()
if __name__=='__main__':main()
