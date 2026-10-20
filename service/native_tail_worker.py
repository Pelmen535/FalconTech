"""Install the optional tail only within this worker process; keep Linux RPC isolation."""
import os
def main():
 from vreid.native_tail import NativeTailPolicy
 import vreid.refusal_policy_vric_road as base
 base.VricRoadRefusalPolicy=NativeTailPolicy
 if os.name=='nt':
  from .vric_road_worker import main as run
 else:
  from .vric_road_linux_worker import main as run
 run()
if __name__=='__main__':main()
