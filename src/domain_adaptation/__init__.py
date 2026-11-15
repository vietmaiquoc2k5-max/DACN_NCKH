"""Domain Adaptation module for Fire and Smoke Detection (Week 5 - MT4).

This module provides tools for:
- Creating experience replay datasets mixing D-Fire and Vietnam camera data.
- Fine-tuning models under 3 adaptation strategies (Full, Freeze Backbone, Freeze + Replay).
- Evaluating domain adaptation performance and catastrophic forgetting against MT4 goals.
"""

__version__ = "1.0.0"

