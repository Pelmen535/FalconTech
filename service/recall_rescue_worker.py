"""Install the optional native rescue only within this worker process; keep Linux RPC isolation."""
import os
def main():
 from vreid.recall_rescue import RecallRescuePolicy
 import vreid.refusal_policy_vric_road as base
 base.VricRoadRefusalPolicy=RecallRescuePolicy
 if os.name=='nt':
  from .vric_road_worker import main as run
 else:
  from .vric_road_linux_worker import main as run
 run()
if __name__=='__main__':main()
