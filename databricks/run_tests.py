"""
Databricks notebook script to run insurance-dynamics tests.
Run this as a Python script in a Databricks notebook cell.
"""

# COMMAND ----------
# %pip install insurance-dynamics ruptures jinja2 pytest pytest-cov

# COMMAND ----------
import subprocess
import sys

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "insurance-dynamics", "ruptures", "jinja2", "pytest"], check=True)

# COMMAND ----------
# Run tests
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--tb=short"],
    capture_output=True, text=True, cwd="/path/to/insurance-dynamics"
)
print(result.stdout)
print(result.stderr)
