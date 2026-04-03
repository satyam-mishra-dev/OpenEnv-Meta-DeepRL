# Prevent pytest from importing the repo-root __init__.py as a test module.
# The package is installed via setuptools, so shopOps imports still work.
collect_ignore = ["__init__.py"]
