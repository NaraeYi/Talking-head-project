"""Compatibility wrapper for the updated metrics runner."""
from run_all_metrics import *  # noqa: F401,F403

if __name__ == '__main__':
    from run_all_metrics import main
    main()
