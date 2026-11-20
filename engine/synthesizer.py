"""
NusaVox Autonomous Neural Voice Engine — Core Synthesizer
Dual-Engine: Studio HD Calibrated Voice (Native Indonesian) + Experimental Zero-Shot Neural Clone (GPU)
"""

import os
import io
import json
import hashlib
import asyncio
import logging
import subprocess
from typing import Optional, Tuple

logger = logging.getLogger("nusavox.synthesizer")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "storage", "cache")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)

from .normalizer import normalize_indonesian_speech

def get_ffmpeg_binary() -> str:
    fdm_ffmpeg = r"C:\Program Files\Softdeluxe\Free Download Manager\ffmpeg.exe"
    if os.path.exists(fdm_ffmpeg):
        return fdm_ffmpeg
    return "ffmpeg"

def convert_to_pcm_wav(file_path: str) -> str:
    """
    Ensures the audio file is a valid 16-bit 24kHz mono PCM WAV readable by libsndfile/librosa.
    If already valid, returns the original path. If not, converts via ffmpeg.
    """
    try:
        import soundfile as sf
        sf.read(file_path)
        return file_path
    except Exception:
        pass

    ffmpeg_bin = get_ffmpeg_binary()
    out_path = file_path.rsplit(".", 1)[0] + "_pcm.wav"
    try:
        cmd = [
            ffmpeg_bin, "-y", "-i", file_path,
            "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
            out_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        if res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            return out_path
    except Exception as e:
        logger.warning(f"Audio conversion failed: {e}")
    return file_path

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
                logger.info("NusaVox Neural Engine: CPU mode (Studio HD)")
        except Exception as e:
            self.device_name = "None"
            logger.info(f"NusaVox Neural Engine: Studio HD mode ({e})")

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
        clean_audio_path = convert_to_pcm_wav(ref_audio_path)
        if clean_audio_path not in self.prompt_cache:
            model = self._get_neural_model()
            logger.info(f"Extracting vocal embedding from: {clean_audio_path}")
            prompt = model.create_voice_clone_prompt(
                ref_audio=clean_audio_path,
                ref_text="",
                x_vector_only_mode=True
            )
            self.prompt_cache[clean_audio_path] = prompt
            self.prompt_cache[ref_audio_path] = prompt
        return self.prompt_cache[ref_audio_path]

    def compute_cache_key(self, text: str, profile_id: str, mode: str, pitch: str, rate: str) -> str:
        data = f"{mode}:{profile_id}:{pitch}:{rate}:{text.strip()}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    async def synthesize(
        self,
        raw_text: str,
        profile_id: str = "bupati-madiun",
        pitch: Optional[str] = None,
        rate: Optional[str] = None,
        engine_mode: Optional[str] = "studio"
    ) -> Tuple[bytes, bool, str, str]:
        """
        Synthesize speech text.
        engine_mode: "studio" (HD Native Indonesian - Recommended) or "clone" (Zero-Shot GPU)
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

        # Determine effective pitch and rate
        # Bupati Madiun golden parameters: -8Hz pitch (deep baritone), -4% rate (dignified calm)
        default_pitch = "-8Hz" if profile_id == "bupati-madiun" else (profile.get("default_pitch", "-2Hz") if profile else "-2Hz")
        default_rate = "-4%" if profile_id == "bupati-madiun" else (profile.get("default_rate", "-2%") if profile else "-2%")

        if profile and profile.get("gender") == "female":
            default_pitch = "+0Hz"
            default_rate = "+0%"

        effective_pitch = pitch if (pitch and pitch.strip()) else default_pitch
        effective_rate = rate if (rate and rate.strip()) else default_rate

        # If user explicitly requested clone mode AND has reference audio AND cuda available
        use_neural_clone = (engine_mode == "clone") and (ref_audio_path is not None) and self.has_neural_cuda

        mode = "neural_clone" if use_neural_clone else "studio_hd"
        cache_key = self.compute_cache_key(normalized, profile_id, mode, effective_pitch, effective_rate)
        ext = "wav" if use_neural_clone else "mp3"
        media_type = "audio/wav" if use_neural_clone else "audio/mpeg"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.{ext}")

        # 1. Return from disk cache if already synthesized
        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
            with open(cache_file, "rb") as f:
                return f.read(), True, normalized, media_type

        # 2A. Experimental Zero-Shot Neural Voice Cloning on CUDA
        if use_neural_clone:
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
                logger.error(f"Neural cloning encountered error, falling back to studio: {e}", exc_info=True)

        # 2B. Studio HD Indonesian Engine (Crystal-Clear Baritone Bupati Madiun)
        import edge_tts
        cfg_voice = "id-ID-ArdiNeural"
        if profile and profile.get("gender") == "female":
            cfg_voice = "id-ID-GadisNeural"

        communicate = edge_tts.Communicate(
            normalized,
            voice=cfg_voice,
            rate=effective_rate,
            pitch=effective_pitch
        )
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        audio_bytes = b"".join(audio_chunks)
        if not audio_bytes:
            raise RuntimeError("Speech synthesis failed to produce audio data.")

        # Cache studio output
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
