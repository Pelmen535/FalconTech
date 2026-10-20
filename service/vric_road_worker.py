"""Process-local policy replacement; preserve the original worker and profiles."""
def main():
    from vreid.refusal_policy_vric_road import VricRoadRefusalPolicy
    import vreid.refusal_policy_expanded as expanded
    expanded.ExpandedRefusalPolicy=VricRoadRefusalPolicy
    from service.robust_worker import main as run
    run()

if __name__=='__main__':main()
