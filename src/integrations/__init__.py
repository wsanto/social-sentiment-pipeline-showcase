"""
Integration modules for external APIs

This package contains clients for:
- LunarCrush API: Social media data aggregation
- Kaiko EQ+ SDK: Emotion analysis
"""

from src.integrations.lunarcrush_client import LunarCrushClient
from src.integrations.kaiko_client import KaikoClient

__all__ = ["LunarCrushClient", "KaikoClient"]
