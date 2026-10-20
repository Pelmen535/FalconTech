"""Process-local policy replacement; preserve the original worker and profiles."""
def main():
    from vreid.refusal_policy_recall import RecallRefusalPolicy
    import vreid.refusal_policy_expanded as expanded
    expanded.ExpandedRefusalPolicy=RecallRefusalPolicy
    from service.robust_worker import main as run
    run()

if __name__=='__main__':main()
