"""
NusaVox Autonomous Neural Voice Engine
Production FastAPI Microservice for Civic & Enterprise AI Voice Cloning and Synthesis.
"""

import os
import re
import time
import json
import shutil
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field

from engine.synthesizer import NusaVoxSynthesizer, convert_to_pcm_wav

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
PROFILES_JSON = os.path.join(PROFILES_DIR, "profiles.json")
os.makedirs(PROFILES_DIR, exist_ok=True)

app = FastAPI(
    title="NusaVox Neural Engine",
    description="High-Performance Autonomous Neural Voice Cloning & Speech Synthesis Microservice",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

synthesizer = NusaVoxSynthesizer()
START_TIME = time.time()

# Request Schemas
class OpenAISpeechRequest(BaseModel):
    model: Optional[str] = "nusavox-civic-1"
    input: str = Field(..., description="The text to synthesize into spoken audio")
    voice: Optional[str] = Field("bupati-madiun", description="Voice profile ID (e.g. bupati-madiun)")
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0

class DirectSynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice: Optional[str] = Field("bupati-madiun", description="Voice profile identifier")
    pitch: Optional[str] = None
    rate: Optional[str] = None

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

@app.get("/health")
def health_check():
    cache_dir = synthesizer.cache_dir
    cached_count = len(os.listdir(cache_dir)) if os.path.exists(cache_dir) else 0
    uptime_sec = round(time.time() - START_TIME, 2)
    profiles = synthesizer.get_profiles()
    return {
        "status": "online",
        "engine": "NusaVox Neural Engine",
        "version": "1.1.0",
        "cuda_available": synthesizer.has_neural_cuda,
        "device_name": synthesizer.device_name,
        "uptime_seconds": uptime_sec,
        "total_voices": len(profiles),
        "cached_audio_files": cached_count,
        "default_voice": "bupati-madiun"
    }

@app.get("/voices")
def list_voices():
    profiles = synthesizer.get_profiles()
    # Add full sample audio URL for each profile if file exists
    for p in profiles:
        sample_file = p.get("sample_file")
        if sample_file and os.path.exists(os.path.join(PROFILES_DIR, sample_file)):
            p["sample_url"] = f"/profiles/audio/{sample_file}"
        else:
            p["sample_url"] = None
    return {
        "engine_name": "NusaVox Neural Engine",
        "version": "1.1.0",
        "cuda_available": synthesizer.has_neural_cuda,
        "device_name": synthesizer.device_name,
        "profiles": profiles
    }

@app.get("/profiles/audio/{filename}")
def serve_profile_audio(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(PROFILES_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    media_type = "audio/wav" if safe_name.endswith(".wav") else "audio/mpeg"
    return FileResponse(path, media_type=media_type)

@app.post("/voices/upload")
async def upload_voice_profile(
    file: UploadFile = File(...),
    name: str = Form(...),
    voice_id: Optional[str] = Form(None),
    gender: str = Form("male"),
    description: Optional[str] = Form("")
):
    """
    Upload a new voice sample to clone and register as a voice profile.
    """
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Voice name is required.")

    slug_id = slugify(voice_id) if (voice_id and voice_id.strip()) else slugify(clean_name)
    if not slug_id:
        slug_id = f"custom-voice-{int(time.time())}"

    # Determine file extension
    orig_ext = os.path.splitext(file.filename or "")[1].lower()
    if orig_ext not in [".wav", ".mp3", ".ogg", ".m4a", ".webm"]:
        orig_ext = ".wav"

    temp_raw_filename = f"{slug_id}_raw{orig_ext}"
    temp_raw_path = os.path.join(PROFILES_DIR, temp_raw_filename)
    target_filename = f"{slug_id}.wav"
    target_path = os.path.join(PROFILES_DIR, target_filename)

    # Save raw uploaded file
    with open(temp_raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Standardize to 16-bit 24kHz PCM WAV via ffmpeg
    clean_path = convert_to_pcm_wav(temp_raw_path)
    if os.path.exists(clean_path):
        if clean_path != target_path:
            shutil.copyfile(clean_path, target_path)
            if clean_path != temp_raw_path:
                try: os.remove(clean_path)
                except Exception: pass
        if temp_raw_path != target_path:
            try: os.remove(temp_raw_path)
            except Exception: pass

    # Read profiles.json
    profiles_data = {"engine_name": "NusaVox Neural Engine", "version": "1.1.0", "profiles": []}
    if os.path.exists(PROFILES_JSON):
        try:
            with open(PROFILES_JSON, "r", encoding="utf-8") as f:
                profiles_data = json.load(f)
        except Exception:
            pass

    profiles = profiles_data.get("profiles", [])

    # Check if voice_id already exists; update it or append
    existing = False
    new_profile = {
        "id": slug_id,
        "name": clean_name,
        "description": description.strip() or f"Kloning suara kustom: {clean_name}",
        "gender": gender if gender in ["male", "female"] else "male",
        "language": "id-ID",
        "sample_file": target_filename,
        "default_pitch": "-2Hz" if gender == "male" else "+0Hz",
        "default_rate": "-1%",
        "category": "custom-clone",
        "is_custom": True,
        "created_at": int(time.time())
    }

    for i, p in enumerate(profiles):
        if p.get("id") == slug_id:
            profiles[i] = new_profile
            existing = True
            break

    if not existing:
        profiles.append(new_profile)

    profiles_data["profiles"] = profiles

    with open(PROFILES_JSON, "w", encoding="utf-8") as f:
        json.dump(profiles_data, f, indent=2, ensure_ascii=False)

    # Invalidate synthesizer prompt cache for this voice if previously cached
    if target_path in synthesizer.prompt_cache:
        del synthesizer.prompt_cache[target_path]

    new_profile["sample_url"] = f"/profiles/audio/{target_filename}"
    return {
        "status": "success",
        "message": f"Profil suara '{clean_name}' berhasil didaftarkan!",
        "profile": new_profile
    }

@app.delete("/voices/{voice_id}")
def delete_voice_profile(voice_id: str):
    """
    Delete a custom voice profile and its sample audio file.
    """
    if not os.path.exists(PROFILES_JSON):
        raise HTTPException(status_code=404, detail="No profiles found.")

    with open(PROFILES_JSON, "r", encoding="utf-8") as f:
        profiles_data = json.load(f)

    profiles = profiles_data.get("profiles", [])
    target = None
    target_idx = -1

    for i, p in enumerate(profiles):
        if p.get("id") == voice_id:
            target = p
            target_idx = i
            break

    if not target:
        raise HTTPException(status_code=404, detail="Voice profile not found.")

    if not target.get("is_custom", False):
        raise HTTPException(status_code=403, detail="Suara sistem default tidak dapat dihapus.")

    # Remove audio file
    sample_file = target.get("sample_file")
    if sample_file:
        file_path = os.path.join(PROFILES_DIR, sample_file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                pass

    # Remove from list
    profiles.pop(target_idx)
    profiles_data["profiles"] = profiles

    with open(PROFILES_JSON, "w", encoding="utf-8") as f:
        json.dump(profiles_data, f, indent=2, ensure_ascii=False)

    return {"status": "success", "message": f"Suara '{voice_id}' berhasil dihapus."}

@app.post("/v1/audio/speech")
async def openai_speech_endpoint(req: OpenAISpeechRequest):
    """OpenAI API Compatible Speech Endpoint"""
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        audio_bytes, is_cached, normalized_text, media_type = await synthesizer.synthesize(
            raw_text=req.input,
            profile_id=req.voice or "bupati-madiun"
        )
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "X-NusaVox-Cached": "true" if is_cached else "false",
                "X-NusaVox-Voice": req.voice or "bupati-madiun",
                "X-NusaVox-Engine": "neural_clone" if media_type == "audio/wav" else "acoustic_fallback",
                "Content-Disposition": f'inline; filename="speech.{"wav" if media_type == "audio/wav" else "mp3"}"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
@app.post("/synthesize")
async def direct_synthesize(req: DirectSynthesizeRequest):
    """Direct JSON Synthesis Endpoint"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' is required.")

    try:
        audio_bytes, is_cached, normalized_text, media_type = await synthesizer.synthesize(
            raw_text=req.text,
            profile_id=req.voice or "bupati-madiun",
            pitch=req.pitch,
            rate=req.rate
        )
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "X-NusaVox-Cached": "true" if is_cached else "false",
                "X-NusaVox-Voice": req.voice or "bupati-madiun",
                "X-NusaVox-Engine": "neural_clone" if media_type == "audio/wav" else "acoustic_fallback",
                "Content-Disposition": f'inline; filename="speech.{"wav" if media_type == "audio/wav" else "mp3"}"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def index_web_studio():
    """Interactive NusaVox Studio & Voice Cloning CRUD Manager"""
    return """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NusaVox Studio — Autonomous Voice Cloning & Neural Engine</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Plus Jakarta Sans', sans-serif; }
    .glass { background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .gold-gradient { background: linear-gradient(135deg, #F59E0B 0%, #D97706 50%, #B45309 100%); }
    .btn-gradient { background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%); }
    .pulse-record { animation: pulseRecord 1.5s infinite; }
    @keyframes pulseRecord {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
      70% { transform: scale(1); box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col justify-between p-4 md:p-8">

  <!-- Header -->
  <header class="w-full max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between py-4 border-b border-slate-800/80 mb-6 gap-4">
    <div class="flex items-center gap-3">
      <div class="w-11 h-11 rounded-2xl gold-gradient flex items-center justify-center font-extrabold text-slate-950 text-2xl shadow-lg shadow-amber-500/20">
        N
      </div>
      <div>
        <div class="flex items-center gap-2">
          <h1 class="text-xl font-bold tracking-tight">NusaVox Studio</h1>
          <span class="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">v1.1 Neural</span>
        </div>
        <p class="text-xs text-slate-400">Autonomous Neural Voice Cloning & Speech Microservice</p>
      </div>
    </div>

    <!-- Mode Badge & Nav Tabs -->
    <div class="flex items-center gap-3">
      <div id="gpuBadge" class="hidden sm:inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span id="gpuText">Checking Engine...</span>
      </div>
      <div class="flex bg-slate-900 p-1 rounded-xl border border-slate-800">
        <button id="tabBtnStudio" onclick="switchTab('studio')" class="px-4 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-slate-950 shadow transition-all">
          🎙️ Studio Generator
        </button>
        <button id="tabBtnClone" onclick="switchTab('clone')" class="px-4 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-100 transition-all">
          🧬 Kloning Suara (CRUD)
        </button>
      </div>
    </div>
  </header>

  <!-- CONTENT TABS -->
  <main class="w-full max-w-5xl mx-auto flex-1 flex flex-col justify-center">

    <!-- TAB 1: STUDIO SINTESIS (TEXT TO SPEECH) -->
    <section id="tabStudio" class="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">
      <!-- Left: Form Input -->
      <div class="glass p-6 rounded-2xl shadow-2xl flex flex-col justify-between">
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <h2 class="text-base font-semibold flex items-center gap-2">
              <svg class="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z"></path></svg>
              Sintesis Suara (TTS)
            </h2>
            <span id="activeEngineType" class="text-[11px] text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20 font-mono">Zero-Shot Neural</span>
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-300 mb-1.5">Pilih Profil Karakter Suara</label>
            <select id="voiceSelect" onchange="onVoiceSelected()" class="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-sm focus:outline-none focus:border-amber-500 text-slate-100">
              <option value="bupati-madiun">Memuat suara...</option>
            </select>
          </div>

          <!-- Quick Samples -->
          <div>
            <div class="flex items-center justify-between mb-1.5">
              <label class="text-xs font-medium text-slate-300">Naskah Teks (Bahasa Indonesia)</label>
              <div class="flex gap-1.5 text-[10px]">
                <button type="button" onclick="setSampleText('bupati')" class="text-amber-400 hover:underline">Sapaan Bupati</button>
                <span class="text-slate-600">|</span>
                <button type="button" onclick="setSampleText('sop')" class="text-amber-400 hover:underline">SOP KIA</button>
                <span class="text-slate-600">|</span>
                <button type="button" onclick="setSampleText('singkat')" class="text-amber-400 hover:underline">Singkat</button>
              </div>
            </div>
            <textarea id="textInput" rows="4" class="w-full bg-slate-900 border border-slate-700/80 rounded-xl p-3.5 text-sm focus:outline-none focus:border-amber-500 text-slate-100 placeholder-slate-500">Halo masyarakat Kabupaten Madiun. Selamat datang di Mal Pelayanan Publik. Silakan sampaikan layanan kependudukan atau perizinan yang Bapak dan Ibu butuhkan.</textarea>
          </div>

          <button id="btnSynthesize" onclick="generateAudio()" class="w-full gold-gradient hover:opacity-95 text-slate-950 font-bold py-3 px-4 rounded-xl shadow-lg shadow-amber-500/25 transition-all flex items-center justify-center gap-2">
            <svg id="playIcon" class="w-5 h-5 fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
            <span id="btnText">Generate & Putar Suara</span>
          </button>
        </div>

        <!-- Audio Player Container -->
        <div id="audioWrapper" class="mt-4 hidden pt-4 border-t border-slate-800">
          <div class="flex items-center justify-between text-xs text-slate-400 mb-2">
            <span>Hasil Suara Output:</span>
            <span id="cacheTag" class="font-mono text-[10px] px-2 py-0.5 rounded">Cached</span>
          </div>
          <audio id="audioPlayer" controls class="w-full rounded-lg"></audio>
        </div>
      </div>

      <!-- Right: Voice Details & API Docs -->
      <div class="glass p-6 rounded-2xl shadow-2xl flex flex-col justify-between">
        <div>
          <div class="flex items-center justify-between mb-3">
            <h2 class="text-base font-semibold flex items-center gap-2">
              <svg class="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
              Detail Karakter Aktif
            </h2>
            <span id="voiceBadge" class="text-[10px] font-semibold px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">System Voice</span>
          </div>

          <div id="voiceDetailCard" class="bg-slate-900/80 rounded-xl p-4 border border-slate-800 mb-4 space-y-2">
            <div class="flex items-center justify-between">
              <h3 id="detailName" class="font-bold text-sm text-slate-100">Bapak Bupati Madiun</h3>
              <span id="detailGender" class="text-xs text-slate-400">Pria</span>
            </div>
            <p id="detailDesc" class="text-xs text-slate-400 leading-relaxed">Suara resmi berwibawa, ramah, dan santun khas birokrasi Jawa Timur.</p>
            <div id="referenceAudioContainer" class="hidden pt-2 border-t border-slate-800/80">
              <p class="text-[11px] text-amber-400 mb-1 flex items-center gap-1">
                <span>🎙️ Sampel Rekaman Asli:</span>
              </p>
              <audio id="refAudioPlayer" controls class="w-full h-8"></audio>
            </div>
          </div>

          <h3 class="text-xs font-semibold text-slate-300 mb-2">Integrasi REST API (OpenAI Compatible)</h3>
          <div class="bg-slate-950 rounded-xl p-3 border border-slate-800 text-[11px] font-mono text-slate-300 overflow-x-auto space-y-1.5">
            <div class="text-amber-400">POST /v1/audio/speech</div>
            <div class="text-slate-400">{</div>
            <div class="pl-3 text-emerald-300">"model": <span class="text-slate-100">"nusavox-civic-1"</span>,</div>
            <div class="pl-3 text-emerald-300">"voice": <span id="apiVoiceSnippet" class="text-slate-100">"bupati-madiun"</span>,</div>
            <div class="pl-3 text-emerald-300">"input": <span class="text-slate-100">"Teks panduan pelayanan..."</span></div>
            <div class="text-slate-400">}</div>
          </div>
        </div>

        <div class="mt-4 pt-3 border-t border-slate-800 text-xs text-slate-400 flex items-center justify-between">
          <span>⚡ Fast Cache SHA-256</span>
          <span class="text-amber-400 cursor-pointer hover:underline" onclick="switchTab('clone')">+ Buat Kloning Baru &rarr;</span>
        </div>
      </div>
    </section>

    <!-- TAB 2: KLONING SUARA SENDIRI (VOICE CRUD MANAGER) -->
    <section id="tabClone" class="hidden grid grid-cols-1 md:grid-cols-2 gap-6 w-full">
      <!-- Left: Create Voice (Upload & Live Record) -->
      <div class="glass p-6 rounded-2xl shadow-2xl flex flex-col justify-between">
        <div>
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-base font-semibold flex items-center gap-2">
              <svg class="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
              Daftarkan Suara Baru (Kloning)
            </h2>
            <span class="text-xs text-slate-400">Upload / Rekam Mic</span>
          </div>

          <!-- Toggle: File Upload vs Mic Record -->
          <div class="flex bg-slate-900 p-1 rounded-xl border border-slate-800 mb-4">
            <button id="btnInputUpload" type="button" onclick="setInputMode('upload')" class="flex-1 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-slate-950 transition-all">
              📁 Upload File Audio
            </button>
            <button id="btnInputMic" type="button" onclick="setInputMode('mic')" class="flex-1 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-100 transition-all">
              🎤 Rekam Langsung (Mic)
            </button>
          </div>

          <form id="voiceForm" onsubmit="submitNewVoice(event)" class="space-y-3.5">
            <!-- Mode 1: File Input -->
            <div id="fileUploadContainer">
              <label class="block text-xs font-medium text-slate-300 mb-1">File Rekaman Audio (.wav, .mp3, .m4a)</label>
              <input type="file" id="voiceFile" accept="audio/*" class="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-amber-500 text-slate-200 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-amber-500/20 file:text-amber-400 hover:file:bg-amber-500/30">
              <p class="text-[11px] text-slate-500 mt-1">Disarankan durasi rekaman 10-30 detik dengan suara jelas tanpa noise.</p>
            </div>

            <!-- Mode 2: Live Mic Recording -->
            <div id="micRecordContainer" class="hidden bg-slate-900/90 border border-slate-800 rounded-xl p-4 text-center space-y-3">
              <div id="recordStatusText" class="text-xs text-slate-400">Tekan tombol merah untuk merekam suara Anda melalui mikrofon</div>
              
              <div class="flex justify-center items-center gap-3">
                <button type="button" id="btnRecordToggle" onclick="toggleRecording()" class="w-12 h-12 rounded-full bg-red-600 hover:bg-red-500 text-white flex items-center justify-center shadow-lg transition-all">
                  <svg class="w-6 h-6 fill-current" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/></svg>
                </button>
                <div id="recordTimer" class="hidden text-base font-mono font-bold text-red-400">00:00</div>
              </div>

              <div id="recordedPreviewContainer" class="hidden pt-2">
                <p class="text-[11px] text-emerald-400 mb-1">Dengarkan Rekaman Sebelum Disimpan:</p>
                <audio id="recordedAudioPreview" controls class="w-full h-8"></audio>
              </div>
            </div>

            <!-- Profile Details -->
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Nama Karakter / Tokoh</label>
              <input type="text" id="voiceName" required placeholder="Contoh: Suara Saya (Hamzah) / Pak Camat" class="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3.5 py-2 text-sm focus:outline-none focus:border-amber-500 text-slate-100 placeholder-slate-500">
            </div>

            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">ID Karakter (Slug)</label>
                <input type="text" id="voiceId" placeholder="contoh: suara-hamzah" class="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-amber-500 text-slate-100 font-mono">
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Gender</label>
                <select id="voiceGender" class="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-amber-500 text-slate-100">
                  <option value="male">Pria</option>
                  <option value="female">Wanita</option>
                </select>
              </div>
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Deskripsi / Catatan</label>
              <input type="text" id="voiceDesc" placeholder="Contoh: Nada santai, artikulasi jelas" class="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3.5 py-2 text-xs focus:outline-none focus:border-amber-500 text-slate-100 placeholder-slate-500">
            </div>

            <button type="submit" id="btnSubmitVoice" class="w-full gold-gradient hover:opacity-95 text-slate-950 font-bold py-2.5 px-4 rounded-xl shadow-lg shadow-amber-500/25 transition-all text-xs flex items-center justify-center gap-2 mt-2">
              <svg class="w-4 h-4 fill-current" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>
              <span>Simpan & Kloning Suara</span>
            </button>
          </form>
        </div>
      </div>

      <!-- Right: List of Registered Voices (Read & Delete) -->
      <div class="glass p-6 rounded-2xl shadow-2xl flex flex-col justify-between">
        <div>
          <div class="flex items-center justify-between mb-3">
            <h2 class="text-base font-semibold flex items-center gap-2">
              <svg class="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 10h16M4 14h16M4 18h16"></path></svg>
              Daftar Profil Suara Terdaftar
            </h2>
            <button onclick="loadVoices()" class="text-xs text-amber-400 hover:underline">Refresh</button>
          </div>
          <p class="text-xs text-slate-400 mb-3">Setiap profil dapat langsung digunakan di Studio atau dihapus jika tidak diperlukan.</p>

          <!-- Voice Cards Scrollable List -->
          <div id="voicesListContainer" class="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
            <div class="text-center py-8 text-xs text-slate-500">Memuat profil suara...</div>
          </div>
        </div>

        <div class="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-500 flex items-center justify-between">
          <span>Total: <strong id="totalVoicesCount" class="text-slate-300">0</strong> Suara</span>
          <span>Zero-Shot Real-Time Cloning</span>
        </div>
      </div>
    </section>

  </main>

  <!-- Footer -->
  <footer class="w-full max-w-5xl mx-auto text-center text-xs text-slate-500 py-6 border-t border-slate-800/80 mt-8">
    NusaVox Neural Engine &copy; 2026. Microservice & Studio Suara AI untuk Mal Pelayanan Publik & Aplikasi Otonom.
  </footer>

  <script>
    let voicesData = [];
    let currentInputMode = 'upload';
    let mediaRecorder = null;
    let recordedChunks = [];
    let recordedBlob = null;
    let recordInterval = null;
    let recordSeconds = 0;

    // Switch Tabs
    function switchTab(tab) {
      const tabStudio = document.getElementById('tabStudio');
      const tabClone = document.getElementById('tabClone');
      const btnStudio = document.getElementById('tabBtnStudio');
      const btnClone = document.getElementById('tabBtnClone');

      if (tab === 'studio') {
        tabStudio.classList.remove('hidden');
        tabClone.classList.add('hidden');
        btnStudio.className = 'px-4 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-slate-950 shadow transition-all';
        btnClone.className = 'px-4 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-100 transition-all';
      } else {
        tabStudio.classList.add('hidden');
        tabClone.classList.remove('hidden');
        btnClone.className = 'px-4 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-slate-950 shadow transition-all';
        btnStudio.className = 'px-4 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-100 transition-all';
      }
    }

    // Set sample text helper
    function setSampleText(type) {
      const input = document.getElementById('textInput');
      if (type === 'bupati') {
        input.value = "Halo masyarakat Kabupaten Madiun. Selamat datang di Mal Pelayanan Publik. Silakan sampaikan layanan kependudukan atau perizinan yang Bapak dan Ibu butuhkan.";
      } else if (type === 'sop') {
        input.value = "Layanan penerbitan KIA Disdukcapil Kabupaten Madiun memerlukan KTP, KK, Akta Kelahiran, dan pas foto. Proses memakan waktu 10 hingga 15 menit tanpa biaya atau gratis.";
      } else {
        input.value = "Selamat pagi! Ada yang bisa kami bantu hari ini di Mal Pelayanan Publik Kabupaten Madiun?";
      }
    }

    // Load voices from API
    async function loadVoices() {
      try {
        const res = await fetch('/voices');
        const data = await res.json();
        voicesData = data.profiles || [];

        // Update GPU badge
        const gpuBadge = document.getElementById('gpuBadge');
        const gpuText = document.getElementById('gpuText');
        if (data.cuda_available) {
          gpuText.textContent = `⚡ GPU: ${data.device_name}`;
          gpuBadge.className = 'hidden sm:inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
        } else {
          gpuText.textContent = '☁️ Engine: Acoustic Mode';
          gpuBadge.className = 'hidden sm:inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20';
        }

        // Render select dropdown
        const select = document.getElementById('voiceSelect');
        const currentVal = select.value;
        select.innerHTML = '';
        voicesData.forEach(v => {
          const opt = document.createElement('option');
          opt.value = v.id;
          const tag = v.is_custom ? '✨ [Custom]' : (v.id === 'bupati-madiun' ? '👑 [Bupati]' : '🏛️ [Sistem]');
          opt.textContent = `${tag} ${v.name}`;
          select.appendChild(opt);
        });

        if (currentVal && voicesData.some(v => v.id === currentVal)) {
          select.value = currentVal;
        } else if (voicesData.length > 0) {
          select.value = voicesData[0].id;
        }

        onVoiceSelected();
        renderVoicesList();
      } catch (err) {
        console.error('Failed to load voices:', err);
      }
    }

    // Update voice detail card
    function onVoiceSelected() {
      const vid = document.getElementById('voiceSelect').value;
      const v = voicesData.find(x => x.id === vid);
      if (!v) return;

      document.getElementById('detailName').textContent = v.name;
      document.getElementById('detailDesc').textContent = v.description || 'Tanpa keterangan.';
      document.getElementById('detailGender').textContent = v.gender === 'female' ? 'Wanita' : 'Pria';
      document.getElementById('apiVoiceSnippet').textContent = `"${v.id}"`;

      const badge = document.getElementById('voiceBadge');
      if (v.is_custom) {
        badge.textContent = 'Custom Cloned';
        badge.className = 'text-[10px] font-semibold px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20';
      } else {
        badge.textContent = 'Official Profile';
        badge.className = 'text-[10px] font-semibold px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20';
      }

      const refContainer = document.getElementById('referenceAudioContainer');
      const refPlayer = document.getElementById('refAudioPlayer');
      if (v.sample_url) {
        refPlayer.src = v.sample_url;
        refContainer.classList.remove('hidden');
      } else {
        refContainer.classList.add('hidden');
        refPlayer.pause();
      }
    }

    // Render CRUD List in Tab 2
    function renderVoicesList() {
      const container = document.getElementById('voicesListContainer');
      document.getElementById('totalVoicesCount').textContent = voicesData.length;

      if (!voicesData || voicesData.length === 0) {
        container.innerHTML = '<div class="text-center py-8 text-xs text-slate-500">Belum ada suara terdaftar.</div>';
        return;
      }

      container.innerHTML = '';
      voicesData.forEach(v => {
        const card = document.createElement('div');
        card.className = 'bg-slate-900/90 border border-slate-800 rounded-xl p-3 flex flex-col gap-2 hover:border-slate-700 transition-all';

        const isCustom = v.is_custom;
        const tag = isCustom
          ? '<span class="text-[10px] bg-amber-500/10 text-amber-400 border border-amber-500/20 px-1.5 py-0.5 rounded">Custom</span>'
          : '<span class="text-[10px] bg-blue-500/10 text-blue-400 border border-blue-500/20 px-1.5 py-0.5 rounded">Sistem</span>';

        card.innerHTML = `
          <div class="flex items-center justify-between">
            <div>
              <div class="flex items-center gap-1.5">
                <span class="text-sm font-semibold text-slate-100">${v.name}</span>
                ${tag}
              </div>
              <p class="text-[11px] text-slate-400 font-mono">${v.id} • ${v.gender === 'female' ? 'Wanita' : 'Pria'}</p>
            </div>
            <div class="flex items-center gap-1.5">
              <button onclick="useVoiceInStudio('${v.id}')" class="px-2.5 py-1 rounded-lg bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 text-xs font-semibold transition-all">
                🎙️ Gunakan
              </button>
              ${isCustom ? `
                <button onclick="deleteVoice('${v.id}')" class="px-2.5 py-1 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 text-xs font-semibold transition-all">
                  🗑️
                </button>
              ` : ''}
            </div>
          </div>
          ${v.sample_url ? `
            <div class="pt-1.5 border-t border-slate-800/60">
              <audio controls src="${v.sample_url}" class="w-full h-7 rounded"></audio>
            </div>
          ` : ''}
        `;
        container.appendChild(card);
      });
    }

    // Switch to Studio with voice selected
    function useVoiceInStudio(vid) {
      switchTab('studio');
      const select = document.getElementById('voiceSelect');
      select.value = vid;
      onVoiceSelected();
    }

    // Delete Voice Profile
    async function deleteVoice(vid) {
      if (!confirm(`Hapus profil suara "${vid}"? File rekaman akan dihapus dari server.`)) return;

      try {
        const res = await fetch(`/voices/${vid}`, { method: 'DELETE' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Gagal menghapus suara.');
        alert(data.message || 'Suara berhasil dihapus.');
        loadVoices();
      } catch (err) {
        alert(err.message);
      }
    }

    // Toggle Input Mode (File vs Mic)
    function setInputMode(mode) {
      currentInputMode = mode;
      const fileContainer = document.getElementById('fileUploadContainer');
      const micContainer = document.getElementById('micRecordContainer');
      const btnUpload = document.getElementById('btnInputUpload');
      const btnMic = document.getElementById('btnInputMic');

      if (mode === 'upload') {
        fileContainer.classList.remove('hidden');
        micContainer.classList.add('hidden');
        btnUpload.className = 'flex-1 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-slate-950 transition-all';
        btnMic.className = 'flex-1 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-100 transition-all';
      } else {
        fileContainer.classList.add('hidden');
        micContainer.classList.remove('hidden');
        btnMic.className = 'flex-1 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-slate-950 transition-all';
        btnUpload.className = 'flex-1 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-100 transition-all';
      }
    }

    // Microphone Recording Logic
    async function toggleRecording() {
      const btn = document.getElementById('btnRecordToggle');
      const statusText = document.getElementById('recordStatusText');
      const timer = document.getElementById('recordTimer');
      const preview = document.getElementById('recordedPreviewContainer');

      if (mediaRecorder && mediaRecorder.state === 'recording') {
        // Stop recording
        mediaRecorder.stop();
        btn.classList.remove('pulse-record');
        statusText.textContent = 'Rekaman selesai! Dengarkan preview di bawah.';
        clearInterval(recordInterval);
      } else {
        // Start recording
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          recordedChunks = [];
          mediaRecorder = new MediaRecorder(stream);

          mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) recordedChunks.push(e.data);
          };

          mediaRecorder.onstop = () => {
            recordedBlob = new Blob(recordedChunks, { type: 'audio/wav' });
            const url = URL.createObjectURL(recordedBlob);
            document.getElementById('recordedAudioPreview').src = url;
            preview.classList.remove('hidden');
            stream.getTracks().forEach(t => t.stop());
          };

          mediaRecorder.start();
          btn.classList.add('pulse-record');
          statusText.textContent = 'Merekam... Bicaralah selama 10-20 detik!';
          preview.classList.add('hidden');
          timer.classList.remove('hidden');

          recordSeconds = 0;
          timer.textContent = '00:00';
          recordInterval = setInterval(() => {
            recordSeconds++;
            const mins = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
            const secs = String(recordSeconds % 60).padStart(2, '0');
            timer.textContent = `${mins}:${secs}`;
          }, 1000);

        } catch (err) {
          alert('Tidak dapat mengakses mikrofon: ' + err.message);
        }
      }
    }

    // Form Submit (Upload/Record)
    async function submitNewVoice(e) {
      e.preventDefault();
      const name = document.getElementById('voiceName').value.trim();
      const voiceId = document.getElementById('voiceId').value.trim();
      const gender = document.getElementById('voiceGender').value;
      const desc = document.getElementById('voiceDesc').value.trim();
      const fileInput = document.getElementById('voiceFile');
      const btn = document.getElementById('btnSubmitVoice');

      const formData = new FormData();
      formData.append('name', name);
      if (voiceId) formData.append('voice_id', voiceId);
      formData.append('gender', gender);
      formData.append('description', desc);

      if (currentInputMode === 'upload') {
        if (!fileInput.files || fileInput.files.length === 0) {
          alert('Silakan pilih file rekaman audio terlebih dahulu.');
          return;
        }
        formData.append('file', fileInput.files[0]);
      } else {
        if (!recordedBlob) {
          alert('Silakan rekam suara Anda terlebih dahulu dengan mikrofon.');
          return;
        }
        formData.append('file', recordedBlob, `${voiceId || 'my-voice'}.wav`);
      }

      btn.disabled = true;
      btn.innerHTML = '<span>Mengunggah & Memproses...</span>';

      try {
        const res = await fetch('/voices/upload', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Gagal mendaftarkan suara.');

        alert(`Sukses! Profil suara "${data.profile.name}" berhasil didaftarkan dan siap digunakan!`);
        document.getElementById('voiceForm').reset();
        document.getElementById('recordedPreviewContainer').classList.add('hidden');
        recordedBlob = null;
        loadVoices();
        useVoiceInStudio(data.profile.id);
      } catch (err) {
        alert(err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<svg class="w-4 h-4 fill-current" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg><span>Simpan & Kloning Suara</span>';
      }
    }

    // Synthesize Audio (Tab 1)
    async function generateAudio() {
      const text = document.getElementById('textInput').value.trim();
      const voice = document.getElementById('voiceSelect').value;
      const btn = document.getElementById('btnSynthesize');
      const btnText = document.getElementById('btnText');
      const audioWrapper = document.getElementById('audioWrapper');
      const audioPlayer = document.getElementById('audioPlayer');
      const cacheTag = document.getElementById('cacheTag');

      if (!text) { alert('Silakan masukkan teks terlebih dahulu.'); return; }

      btn.disabled = true;
      btnText.textContent = 'Memproses Kloning Suara...';

      try {
        const t0 = performance.now();
        const res = await fetch('/v1/audio/speech', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ input: text, voice: voice })
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || 'Gagal memproses suara.');
        }

        const isCached = res.headers.get('X-NusaVox-Cached') === 'true';
        const engineType = res.headers.get('X-NusaVox-Engine');
        const elapsed = Math.round(performance.now() - t0);

        if (isCached) {
          cacheTag.textContent = `⚡ Cached (${elapsed} ms)`;
          cacheTag.className = 'font-mono text-[10px] px-2 py-0.5 rounded text-emerald-400 bg-emerald-500/10 border border-emerald-500/20';
        } else if (engineType === 'neural_clone') {
          cacheTag.textContent = `🧠 Neural Clone (${elapsed} ms)`;
          cacheTag.className = 'font-mono text-[10px] px-2 py-0.5 rounded text-amber-400 bg-amber-500/10 border border-amber-500/20';
        } else {
          cacheTag.textContent = `☁️ Acoustic (${elapsed} ms)`;
          cacheTag.className = 'font-mono text-[10px] px-2 py-0.5 rounded text-blue-400 bg-blue-500/10 border border-blue-500/20';
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        audioPlayer.src = url;
        audioWrapper.classList.remove('hidden');
        audioPlayer.play();
      } catch (err) {
        alert(err.message);
      } finally {
        btn.disabled = false;
        btnText.textContent = 'Generate & Putar Suara';
      }
    }

    // Initial load
    loadVoices();
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=17494)
