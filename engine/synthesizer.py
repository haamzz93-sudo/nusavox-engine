"""
NusaVox Autonomous Neural Voice Engine — Core Synthesizer
Hybrid Zero-Shot Neural Cloning (CUDA / Qwen3-TTS) + Edge Acoustic Engine + Cache
"""

import os
import io
import json
import hashlib
import asyncio
import logging
from typing import Optional, Tuple

logger = logging.getLogger("nusavox.synthesizer")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "storage", "cache")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)

from .normalizer import normalize_indonesian_speech

class NusaVoxSynthesizer:
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.profiles_dir = PROFILES_DIR
        self.neural_model = None
        self.prompt_cache = {}
        self.has_neural_cuda = False

        # Detect PyTorch CUDA & Qwen-TTS availability
        try:
            import torch
            if torch.cuda.is_available():
                self.has_neural_cuda = True
                self.device_name = torch.cuda.get_device_name(0)
                logger.info(f"NusaVox Neural CUDA Engine Ready: {self.device_name}")
            else:
                self.device_name = "CPU"
                logger.info("NusaVox Neural Engine: CPU mode (Acoustic Fallback)")
        except Exception as e:
            self.device_name = "None"
            logger.info(f"NusaVox Neural Engine: Acoustic mode ({e})")

    def get_profiles(self) -> list:
        json_path = os.path.join(self.profiles_dir, "profiles.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("profiles", [])
            except Exception as e:
                logger.error(f"Error reading profiles.json: {e}")
        return []

    def get_profile(self, profile_id: str) -> Optional[dict]:
        for p in self.get_profiles():
            if p.get("id") == profile_id:
                return p
        return None

    def _get_neural_model(self):
        if self.neural_model is None and self.has_neural_cuda:
            import torch
            from qwen_tts import Qwen3TTSModel
            logger.info("Loading Qwen3-TTS 0.6B on CUDA...")
            self.neural_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map="cuda",
                dtype=torch.bfloat16
            )
            logger.info("Qwen3-TTS Loaded Successfully on GPU!")
        return self.neural_model

    def _get_voice_clone_prompt(self, ref_audio_path: str):
        if ref_audio_path not in self.prompt_cache:
            model = self._get_neural_model()
            logger.info(f"Extracting vocal embedding from: {ref_audio_path}")
            prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text="",
                x_vector_only_mode=True
            )
            self.prompt_cache[ref_audio_path] = prompt
        return self.prompt_cache[ref_audio_path]

    def compute_cache_key(self, text: str, profile_id: str, mode: str) -> str:
        data = f"{mode}:{profile_id}:{text.strip()}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    async def synthesize(
        self,
        raw_text: str,
        profile_id: str = "bupati-madiun",
        pitch: Optional[str] = None,
        rate: Optional[str] = None
    ) -> Tuple[bytes, bool, str, str]:
        """
        Synthesize speech text.
        Returns: (audio_bytes, is_cached, normalized_text, media_type)
        """
        normalized = normalize_indonesian_speech(raw_text)
        if not normalized.strip():
            raise ValueError("Input text cannot be empty after normalization.")

        profile = self.get_profile(profile_id)
        sample_file = profile.get("sample_file") if profile else None
        ref_audio_path = None

        if sample_file:
            candidate = os.path.join(self.profiles_dir, sample_file)
            if os.path.exists(candidate) and os.path.getsize(candidate) > 1000:
                ref_audio_path = candidate

        # Check if we can run real neural cloning on CUDA
        use_neural = (ref_audio_path is not None) and self.has_neural_cuda

        mode = "neural_clone" if use_neural else "acoustic"
        cache_key = self.compute_cache_key(normalized, profile_id, mode)
        ext = "wav" if use_neural else "mp3"
        media_type = "audio/wav" if use_neural else "audio/mpeg"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.{ext}")

        # 1. Return from disk cache if already synthesized
        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
            with open(cache_file, "rb") as f:
                return f.read(), True, normalized, media_type

        # 2A. Real Zero-Shot Neural Voice Cloning on CUDA
        if use_neural:
            try:
                loop = asyncio.get_running_loop()
                audio_bytes = await loop.run_in_executor(
                    None,
                    self._generate_neural_sync,
                    normalized,
                    ref_audio_path
                )
                with open(cache_file, "wb") as f:
                    f.write(audio_bytes)
                return audio_bytes, False, normalized, media_type
            except Exception as e:
                logger.warning(f"Neural cloning encountered error, falling back to acoustic: {e}")

        # 2B. Acoustic Generator (Edge-TTS Fallback for 24/7 VPS compatibility)
        import edge_tts
        cfg_voice = "id-ID-ArdiNeural"
        cfg_pitch = pitch or (profile.get("default_pitch", "-3Hz") if profile else "-3Hz")
        cfg_rate = rate or (profile.get("default_rate", "-2%") if profile else "-2%")
        if profile and profile.get("gender") == "female":
            cfg_voice = "id-ID-GadisNeural"
            cfg_pitch = "+0Hz"
            cfg_rate = "+0%"

        communicate = edge_tts.Communicate(
            normalized,
            voice=cfg_voice,
            rate=cfg_rate,
            pitch=cfg_pitch
        )
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        audio_bytes = b"".join(audio_chunks)
        if not audio_bytes:
            raise RuntimeError("Speech synthesis failed to produce audio data.")

        # Cache acoustic output
        with open(cache_file, "wb") as f:
            f.write(audio_bytes)

        return audio_bytes, False, normalized, "audio/mpeg"

    def _generate_neural_sync(self, text: str, ref_audio_path: str) -> bytes:
        import torch
        import soundfile as sf
        prompt = self._get_voice_clone_prompt(ref_audio_path)
        model = self._get_neural_model()

        with torch.inference_mode():
            wavs, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt,
                language="auto"
            )

        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        return buf.getvalue()
