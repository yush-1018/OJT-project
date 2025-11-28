# OJT-project
Probability-Guided Number Guessing Simulator

The Probability-Guided Number Guessing Simulator is a Python CLI game that demonstrates production-grade scripting practices. The game selects a hidden number within a configurable range, and after each incorrect guess it recalculates and displays the remaining possible range—showing how probability narrows the solution space over time.

Key Features

Probability Guidance: The game updates the min/max possible values after every guess, illustrating how search space shrinks.

CLI with Config: Uses argparse and a central INI/JSON config file for --min, --max, and --attempts.

Robust Error Handling: Graceful input validation and a top-level exception handler ensure safe execution and proper exit codes.

Structured Logging: Logs guesses, results, remaining attempts, and exception tracebacks for auditability.

Unit Testing: unittest ensures correct range logic and win/loss conditions.
