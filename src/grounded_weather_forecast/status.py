"""Exit-status protocol shared by CLI commands and installed job wrappers."""

EXIT_NO_FOLDS = 3
# Zero is a maintenance command with expected pending work; one is legacy data.
PENDING_EXIT_CODES = (0, 1, EXIT_NO_FOLDS)
