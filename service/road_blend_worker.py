"""Process-local policy replacement; preserve the original worker and profiles."""
def main():
    from vreid.refusal_policy_road_blend import RoadBlendRefusalPolicy
    import vreid.refusal_policy_expanded as expanded
    expanded.ExpandedRefusalPolicy=RoadBlendRefusalPolicy
    from service.robust_worker import main as run
    run()

if __name__=='__main__':main()
