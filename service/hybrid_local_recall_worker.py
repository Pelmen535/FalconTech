def main():
 from vreid.hybrid_local_recall import HybridTailPolicy
 import vreid.refusal_policy_vric_road as base
 base.VricRoadRefusalPolicy=HybridTailPolicy
 from .vric_road_worker import main as run
 run()
if __name__=='__main__':main()
