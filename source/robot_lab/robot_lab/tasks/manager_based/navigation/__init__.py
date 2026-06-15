# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Navigation tasks for robot_lab.

This module contains hierarchical navigation tasks where a high-level NavRL policy
outputs velocity commands that are tracked by a frozen low-level locomotion policy.
"""

from .config import *  # noqa: F401, F403
