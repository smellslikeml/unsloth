# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inference runtime config; every value is env-overridable."""

from __future__ import annotations

import os

# Compile user-authored corrections from chat history into tool-call deny-rules
# (TRACE), enforced at the same gate as the command blocklist. On by default so
# the "tell me once" behavior ships; set to "0" for the raw tool path. Bypass
# Permissions in the chat also skips the gate, like it skips the static blocklist.
CORRECTION_RULES_ENABLED = os.environ.get("UNSLOTH_STUDIO_CORRECTION_RULES_ENABLED", "1") == "1"
