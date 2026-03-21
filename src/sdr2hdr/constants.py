"""Shared constants for color science calculations."""

from __future__ import annotations

# ITU-R BT.709 luma coefficients (used for Rec.2020 luminance calculation)
LUMA_R = 0.2627
LUMA_G = 0.6780
LUMA_B = 0.0593

# ITU-R BT.601 grayscale coefficients (used by cv2.cvtColor BGR2GRAY)
GRAY_R = 0.2990
GRAY_G = 0.5870
GRAY_B = 0.1140

# SMPTE ST 2084 (PQ) transfer function constants
PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 32.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 128.0
PQ_C3 = 2392.0 / 128.0
