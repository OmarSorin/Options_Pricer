from hypothesis import settings

# Default: 300 random examples per property test (fast enough for every run).
# Thorough: `pytest --hypothesis-profile=thorough` for a deeper search before a release.
settings.register_profile("default", max_examples=300, deadline=None)
settings.register_profile("thorough", max_examples=5000, deadline=None)
settings.load_profile("default")
