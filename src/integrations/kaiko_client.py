"""
Kaiko V2 EQ+ Emotion Analysis Client

Analyzes text for emotions using Kaiko's Synapse V2 API.
Supports advanced EQ features: Intensity, Wonder, Discovery, Growth.
Maps V2 emotions to Plutchik's 8 emotions for backward compatibility.
"""

import os
import asyncio
import time
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple
import aiohttp
import structlog
from datetime import datetime

logger = structlog.get_logger(__name__)

MAX_RETRIES = 2
BASE_BACKOFF_SECONDS = 2

# Kaiko API rate: 60 calls/min = 3600/hour
DEFAULT_CALLS_PER_MINUTE = 60  # ~3600/hour, ~86K/day
MIN_CALLS_PER_MINUTE = 5       # Floor when heavily throttled
RATE_LIMIT_DECAY = 0.8         # Reduce rate by 20% on each 429
RATE_LIMIT_RECOVERY = 1.1      # Increase rate by 10% on success streaks
RECOVERY_STREAK_THRESHOLD = 20 # Consecutive successes before increasing rate


class RateLimiter:
    """Token-bucket rate limiter that adapts to 429 responses.

    Tracks calls per minute. When a 429 is received, the rate is reduced.
    After a streak of successes, the rate slowly recovers toward the default.
    Shared across all KaikoClient instances via module-level singleton.
    """

    def __init__(self, calls_per_minute: float = DEFAULT_CALLS_PER_MINUTE):
        self.calls_per_minute = calls_per_minute
        self._min_interval = 60.0 / calls_per_minute
        self._last_call_time = 0.0
        self._success_streak = 0
        self._total_429s = 0
        self._lock: Optional[asyncio.Lock] = None  # Created lazily per event loop

    async def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self):
        """Wait until we're allowed to make the next API call."""
        lock = await self._get_lock()
        async with lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                logger.debug("rate_limiter_wait", wait_seconds=round(wait_time, 2))
                await asyncio.sleep(wait_time)
            self._last_call_time = time.monotonic()

    def record_success(self):
        """Record a successful API call. Recover rate after sustained success."""
        self._success_streak += 1
        if (self._success_streak >= RECOVERY_STREAK_THRESHOLD
                and self.calls_per_minute < DEFAULT_CALLS_PER_MINUTE):
            self.calls_per_minute = min(
                DEFAULT_CALLS_PER_MINUTE,
                self.calls_per_minute * RATE_LIMIT_RECOVERY
            )
            self._min_interval = 60.0 / self.calls_per_minute
            self._success_streak = 0
            logger.info("rate_limiter_recovered", new_rpm=round(self.calls_per_minute, 1))

    def record_429(self):
        """Record a rate limit hit. Immediately reduce throughput."""
        self._success_streak = 0
        self._total_429s += 1
        self.calls_per_minute = max(
            MIN_CALLS_PER_MINUTE,
            self.calls_per_minute * RATE_LIMIT_DECAY
        )
        self._min_interval = 60.0 / self.calls_per_minute
        logger.warning(
            "rate_limiter_throttled",
            new_rpm=round(self.calls_per_minute, 1),
            total_429s=self._total_429s,
        )

    def get_recommended_batch_size(self) -> int:
        """Return a batch size that fits within the current rate and the 4-min enrichment window."""
        # With 240s soft time limit, budget 200s for actual API work
        available_calls = int(self.calls_per_minute * (200 / 60))
        return max(5, min(50, available_calls))

    def get_stats(self) -> Dict[str, Any]:
        return {
            "calls_per_minute": round(self.calls_per_minute, 1),
            "min_interval_seconds": round(self._min_interval, 2),
            "success_streak": self._success_streak,
            "total_429s": self._total_429s,
        }


# Module-level singleton — shared across all KaikoClient instances in the same process
_rate_limiter = RateLimiter()


class KaikoClient:
    """
    Async client for Kaiko Synapse V2 API

    Analyzes text and returns:
    - V2 Metrics: Intensity, Wonder Index, Discovery Level, Emotional Complexity
    - Legacy Compatibility: Valence, Arousal, Plutchik's 8 emotions
    - Support for Persistent Context (Growth & Trajectory tracking)
    """

    BASE_URL = "https://api.kaikostudios.xyz/v2"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Kaiko client

        Args:
            api_key: Kaiko API key (defaults to env var KAIKO_API_KEY)
        """
        self.api_key = api_key or os.getenv("KAIKO_API_KEY")
        if not self.api_key:
            raise ValueError("Kaiko API key not provided")

        self.session: Optional[aiohttp.ClientSession] = None
        # API uses x-api-key header (not Bearer token)
        self._headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(headers=self._headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _ensure_session(self):
        """Ensure aiohttp session exists and is valid for current event loop"""
        try:
            if self.session is not None and not self.session.closed:
                current_loop = asyncio.get_running_loop()
                if hasattr(self.session, '_loop') and self.session._loop is current_loop:
                    return
                await self.session.close()
        except Exception:
            pass
        self.session = aiohttp.ClientSession(headers=self._headers)

    def _map_emotions_to_plutchik(
        self,
        kaiko_emotions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Map Kaiko V2 GoEmotions to Plutchik's 8 emotions (Legacy Support).

        GoEmotions are softmax probabilities that concentrate mass on 1-2
        dominant emotions, leaving most others near zero (0.001-0.005).
        We aggregate related GoEmotions into each Plutchik category and
        normalize so the 8 values are on a meaningful 0-1 scale.

        Args:
            kaiko_emotions: Dict with raw V2 GoEmotions scores

        Returns:
            Dict with Plutchik's 8 emotions, normalized to 0-1
        """
        # Aggregate related GoEmotions into Plutchik categories
        joy = sum(kaiko_emotions.get(e, 0) or 0 for e in
                  ("joy", "amusement", "excitement", "optimism", "relief"))
        trust = sum(kaiko_emotions.get(e, 0) or 0 for e in
                    ("love", "caring", "gratitude", "admiration", "approval"))
        fear = sum(kaiko_emotions.get(e, 0) or 0 for e in
                   ("fear", "nervousness"))
        surprise = sum(kaiko_emotions.get(e, 0) or 0 for e in
                       ("surprise", "realization", "confusion"))
        sadness = sum(kaiko_emotions.get(e, 0) or 0 for e in
                      ("sadness", "grief", "disappointment", "remorse"))
        disgust = sum(kaiko_emotions.get(e, 0) or 0 for e in
                      ("disgust", "disapproval"))
        anger = sum(kaiko_emotions.get(e, 0) or 0 for e in
                    ("anger", "annoyance"))
        anticipation = sum(kaiko_emotions.get(e, 0) or 0 for e in
                           ("curiosity", "desire", "anticipation"))

        raw = {
            "joy": joy, "trust": trust, "fear": fear, "surprise": surprise,
            "sadness": sadness, "disgust": disgust, "anger": anger,
            "anticipation": anticipation,
        }

        # Normalize: scale so the max Plutchik category maps to ~1.0
        max_val = max(raw.values()) or 1.0
        return {k: round(min(1.0, v / max_val), 4) for k, v in raw.items()}

    def _parse_v2_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a Kaiko V2 API response into standardized emotion metrics."""
        # Log the full response structure to diagnose default/baseline values
        top_keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        emotions_section = data.get("emotions", {}) if isinstance(data, dict) else {}
        emotion_roles = list(emotions_section.keys()) if isinstance(emotions_section, dict) else []
        # Check for per-message results that we might be missing
        has_messages = "messages" in data or "results" in data or "items" in data
        logger.info(
            "kaiko_raw_response_structure",
            top_keys=top_keys,
            emotion_roles=emotion_roles,
            has_per_message_results=has_messages,
            response_preview={k: type(v).__name__ for k, v in data.items()} if isinstance(data, dict) else str(data)[:200],
        )

        user_emotions = data.get("emotions", {}).get("user", {})
        if not user_emotions:
            user_emotions = data.get("emotions", {}).get("_default", {})
            if user_emotions:
                logger.warning("kaiko_using_default_role", message="No 'user' role found, using '_default'")

        raw_emotions = user_emotions.get("raw") or {}
        if not raw_emotions:
            logger.warning(
                "kaiko_empty_raw_emotions",
                user_emotions_keys=list(user_emotions.keys()) if isinstance(user_emotions, dict) else "not_dict",
            )

        # Coalesce None → 0.0 for numeric fields (API may return explicit nulls)
        intensity = user_emotions.get("intensity") or 0.0
        intensity_level = user_emotions.get("intensityLevel") or "minimal"
        complexity = user_emotions.get("complexity") or "simple"
        wonder_index = user_emotions.get("wonderIndex") or 0.0
        discovery_level = user_emotions.get("discoveryLevel") or "routine"

        v2_valence = user_emotions.get("valence") or 0.0
        v2_arousal = user_emotions.get("arousal") or 0.0

        plutchik_emotions = self._map_emotions_to_plutchik(raw_emotions)

        # Toxicity: derive from negative GoEmotions (not the broken (1-trust)/3 formula)
        # The GoEmotions taxonomy has no "toxicity"/"hostility" labels, so use the
        # negative-sentiment GoEmotions as a proxy for toxic content.
        toxicity_score = min(1.0, sum(
            raw_emotions.get(e, 0) for e in
            ("anger", "annoyance", "disgust", "disapproval")
        ))

        # Dominant emotion from SDK; also extract the top non-neutral emotion
        dominant_emotion = user_emotions.get("category", "neutral")
        if dominant_emotion == "neutral" and raw_emotions:
            # Find the highest-scoring non-neutral GoEmotion as an alternative
            non_neutral = {k: v for k, v in raw_emotions.items()
                          if k != "neutral" and isinstance(v, (int, float)) and v > 0.05}
            if non_neutral:
                dominant_emotion = max(non_neutral, key=non_neutral.get)

        # --- P0: Safety & Hostility fields from SDK V2 ---
        safety_concern_score = user_emotions.get("safetyConcernScore") or 0.0

        hostility_state = user_emotions.get("hostilityState") or {}
        hostility_level = hostility_state.get("level", "none")
        hostility_score = hostility_state.get("score") or 0.0
        hostility_escalation_count = hostility_state.get("escalationCount") or 0
        de_escalation_detected = hostility_state.get("deEscalationDetected", False)

        is_breakthrough = user_emotions.get("isBreakthrough", False)

        # --- P1: Empathy, Meta-Emotional, Labels, Vector fields ---
        empathic_concern = user_emotions.get("empathicConcern") or 0.0
        personal_distress = user_emotions.get("personalDistress") or 0.0
        meta_emotional_score = user_emotions.get("metaEmotionalScore") or 0.0
        emotional_vector = user_emotions.get("emotionalVector") or []
        emotional_tags = user_emotions.get("emotionalTags") or []
        emotional_signature = user_emotions.get("emotionalSignature") or ""
        pattern_emotion_score = user_emotions.get("patternEmotionScore") or 0.0
        dimensional_source = user_emotions.get("dimensionalSource") or ""
        is_fallback = user_emotions.get("isFallback", False)

        # Store the full 27 GoEmotions as structured dict (not opaque blob)
        raw_27_emotions = {}
        for key, value in raw_emotions.items():
            if isinstance(value, (int, float)):
                raw_27_emotions[key] = round(float(value), 4)

        # Active labels: use SDK value if present, otherwise derive from raw scores
        # Threshold 0.1 = emotions with >10% probability are considered "active"
        active_labels = user_emotions.get("activeLabels") or []
        if not active_labels and raw_27_emotions:
            active_labels = sorted(
                [k for k, v in raw_27_emotions.items() if v > 0.1 and k != "neutral"],
                key=lambda k: raw_27_emotions[k],
                reverse=True
            )

        return {
            "valence": round(v2_valence, 4),
            "arousal": round(v2_arousal, 4),
            "emotions": {k: round(v, 4) for k, v in plutchik_emotions.items()},
            "dominant_emotion": dominant_emotion,
            "toxicity_score": round(toxicity_score, 4),
            "intensity": round(intensity, 4),
            "intensity_level": intensity_level,
            "emotional_complexity": complexity,
            "wonder_index": round(wonder_index, 4),
            "discovery_level": discovery_level,
            # P0: Safety & Hostility
            "safety_concern_score": round(safety_concern_score, 4),
            "hostility_level": hostility_level,
            "hostility_score": round(hostility_score, 4),
            "hostility_escalation_count": hostility_escalation_count,
            "de_escalation_detected": de_escalation_detected,
            "is_breakthrough": is_breakthrough,
            # P1: Empathy & Meta-Emotional
            "empathic_concern": round(empathic_concern, 4),
            "personal_distress": round(personal_distress, 4),
            "meta_emotional_score": round(meta_emotional_score, 4),
            "active_labels": active_labels,
            "emotional_vector": emotional_vector,
            "emotional_tags": emotional_tags,
            # P2: Additional fields (extracted now for forward compat)
            "emotional_signature": emotional_signature,
            "pattern_emotion_score": round(pattern_emotion_score, 4),
            "dimensional_source": dimensional_source,
            "is_fallback": is_fallback,
            # P2: Optional SDK sections (present when context_id is used)
            "trajectory": data.get("trajectory") or {},
            "growth": data.get("growth") or {},
            "patterns": data.get("patterns") or {},
            "beliefs": data.get("beliefs") or {},
            "conversation_mode": data.get("conversationMode", {}).get("mode", "") if isinstance(data.get("conversationMode"), dict) else str(data.get("conversationMode") or ""),
            "personality_mode": data.get("conversationMode", {}).get("personalityMode", "") if isinstance(data.get("conversationMode"), dict) else "",
            "data_quality": data.get("dataQuality") or dimensional_source or "",
            # Raw data
            "raw_kaiko_response": data,
            "raw_v2_emotions": raw_emotions,
            "raw_27_emotions": raw_27_emotions,
        }

    async def _make_api_call(
        self, url: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make an API call with rate limiting, retry logic, and exponential backoff."""
        await self._ensure_session()

        for attempt in range(MAX_RETRIES + 1):
            # Wait for rate limiter before each attempt
            await _rate_limiter.acquire()

            try:
                async with self.session.post(url, json=payload) as response:
                    if response.status == 429:
                        _rate_limiter.record_429()
                        if attempt < MAX_RETRIES:
                            backoff = BASE_BACKOFF_SECONDS * (2 ** attempt)
                            logger.warning("kaiko_rate_limited", retry=attempt + 1, backoff_seconds=backoff)
                            await asyncio.sleep(backoff)
                            continue
                    response.raise_for_status()
                    _rate_limiter.record_success()
                    return await response.json()
            except aiohttp.ClientResponseError as e:
                if e.status == 429:
                    _rate_limiter.record_429()
                if e.status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    backoff = BASE_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning("kaiko_retry", status=e.status, retry=attempt + 1, backoff=backoff)
                    await asyncio.sleep(backoff)
                    continue
                logger.error("kaiko_api_error", status=e.status, message=str(e), url=url)
                raise
            except Exception as e:
                if attempt < MAX_RETRIES:
                    backoff = BASE_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning("kaiko_retry_error", error=str(e), retry=attempt + 1, backoff=backoff)
                    await asyncio.sleep(backoff)
                    continue
                logger.error("kaiko_unexpected_error", error=str(e))
                raise

    async def analyze_text(
        self,
        text: str,
        model: str = "emotion-1",
        context_id: Optional[str] = None,
        external_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze emotions in text using Synapse V2

        Args:
            text: Text to analyze
            model: Kaiko model to use (default: emotion-v2)
            context_id: Unique session/user ID for stateful tracking (Growth/Trajectory)
            external_id: Unique message ID

        Returns:
            Dict with V2 metrics + legacy fields
        """
        message_obj = {"content": {"text": text}}
        if external_id:
            message_obj["external_id"] = external_id

        payload = {
            "model": model,
            "params": {"granularity": "detailed"},
            "messages": [message_obj]
        }

        if context_id:
            url = f"{self.BASE_URL}/emotions/{context_id}/analysis"
        else:
            url = f"{self.BASE_URL}/emotions/analysis"

        logger.info(
            "kaiko_v2_analyze_request",
            text_length=len(text),
            model=model,
            has_context=bool(context_id)
        )

        data = await self._make_api_call(url, payload)
        result = self._parse_v2_response(data)

        logger.info(
            "kaiko_v2_analyze_success",
            dominant=result["dominant_emotion"],
            intensity=result["intensity"],
            wonder=result["wonder_index"]
        )

        return result

    async def analyze_texts_batch(
        self,
        items: List[Dict[str, str]],
        model: str = "emotion-1",
    ) -> List[Dict[str, Any]]:
        """
        Batch analyze texts by sending individual requests per item.

        The V2 context API returns aggregated per-role results, not per-message
        results, so multi-message payloads produce one result applied to all
        items. Individual calls ensure each text gets its own analysis.

        Args:
            items: List of dicts with 'text', optional 'context_id' and 'external_id'
            model: Kaiko model to use

        Returns:
            List of dicts with 'external_id' and either 'result' or 'error'
        """
        results = []
        for item in items:
            try:
                r = await self.analyze_text(
                    text=item["text"],
                    model=model,
                    context_id=item.get("context_id"),
                    external_id=item.get("external_id"),
                )
                results.append({"external_id": item.get("external_id"), "result": r})
            except Exception as e:
                logger.warning(
                    "kaiko_batch_item_error",
                    error=str(e),
                    external_id=item.get("external_id"),
                )
                results.append({"external_id": item.get("external_id"), "error": str(e)})

        # Warn if all results are identical (symptom of fallback/upstream issue)
        scored = [r["result"].get("valence") for r in results if "result" in r]
        if len(scored) > 2 and len(set(scored)) == 1:
            logger.warning(
                "kaiko_batch_uniform_scores",
                count=len(scored),
                valence=scored[0],
            )

        return results


# Convenience function for testing
async def test_client():
    """Test the Kaiko V2 client"""
    import asyncio

    # Note: Ensure KAIKO_API_KEY is set to Staging key
    async with KaikoClient() as client:
        # Test Stateless
        text = "I am absolutely fascinated by this discovery! It opens up so many new possibilities."
        print(f"Analyzing: {text}")
        result = await client.analyze_text(text)

        print("\n--- Results ---")
        print(f"Dominant: {result['dominant_emotion']}")
        print(f"Wonder Index: {result['wonder_index']}")
        print(f"Discovery Level: {result['discovery_level']}")
        print(f"Intensity: {result['intensity']} ({result['intensity_level']})")
        print(f"Emotions: {result['emotions']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_client())
