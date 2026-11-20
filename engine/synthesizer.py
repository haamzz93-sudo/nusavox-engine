"""
NusaVox Core Neural Synthesizer
Handles speech synthesis, profile mapping, and content-addressable audio caching.
"""

import os
import hashlib
import asyncio
from typing import Optional, Tuple
import edge_tts

from .normalizer import normalize_indonesian_speech

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "storage", "cache")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
os.makedirs(CACHE_DIR, exist_ok=True)

VOICE_CONFIG = {
    "bupati-madiun": {
        "voice": "id-ID-ArdiNeural",
        "pitch": "-3Hz",
        "rate": "-2%",
        "gender": "male"
    },
    "civic-formal-male": {
        "voice": "id-ID-ArdiNeural",
        "pitch": "-1Hz",
        "rate": "-1%",
        "gender": "male"
    },
    "civic-friendly-female": {
        "voice": "id-ID-GadisNeural",
        "pitch": "+0Hz",
        "rate": "+0%",
        "gender": "female"
    }
}

class NusaVoxSynthesizer:
    def __init__(self):
        self.cache_dir = CACHE_DIR

    def compute_cache_key(self, text: str, profile_id: str, pitch: Optional[str] = None, rate: Optional[str] = None) -> str:
        cfg = VOICE_CONFIG.get(profile_id, VOICE_CONFIG["bupati-madiun"])
        p = pitch or cfg["pitch"]
        r = rate or cfg["rate"]
        data = f"{profile_id}:{p}:{r}:{text.strip()}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    async def synthesize(
        self,
        raw_text: str,
        profile_id: str = "bupati-madiun",
        pitch: Optional[str] = None,
        rate: Optional[str] = None
    ) -> Tuple[bytes, bool, str]:
        """
        Synthesize speech text to MP3 bytes.
        Returns: (audio_bytes, is_cached, normalized_text)
        """
        normalized = normalize_indonesian_speech(raw_text)
        if not normalized.strip():
            raise ValueError("Input text cannot be empty after normalization.")

        cfg = VOICE_CONFIG.get(profile_id, VOICE_CONFIG["bupati-madiun"])
        selected_voice = cfg["voice"]
        selected_pitch = pitch or cfg["pitch"]
        selected_rate = rate or cfg["rate"]

        cache_key = self.compute_cache_key(normalized, profile_id, selected_pitch, selected_rate)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.mp3")

        # 1. Return from disk cache if already synthesized
        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 500:
            with open(cache_file, "rb") as f:
                return f.read(), True, normalized

        # 2. Synthesize via neural acoustic generator
        communicate = edge_tts.Communicate(
            normalized,
            voice=selected_voice,
            rate=selected_rate,
            pitch=selected_pitch
        )

        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        audio_bytes = b"".join(audio_chunks)

        if not audio_bytes:
            raise RuntimeError("Speech synthesis failed to produce audio data.")

        # 3. Store in permanent content-addressable cache
        with open(cache_file, "wb") as f:
            f.write(audio_bytes)

        return audio_bytes, False, normalized
