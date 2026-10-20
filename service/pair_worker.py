"""Process-local policy replacement; preserve the original worker and profiles."""
def main():
    from vreid.refusal_policy_pair import PairRefusalPolicy
    import vreid.refusal_policy_expanded as expanded
    expanded.ExpandedRefusalPolicy=PairRefusalPolicy
    from service.robust_worker import main as run
    run()

if __name__=='__main__':main()
