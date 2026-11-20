"""
NusaVox Autonomous Neural Voice Engine
Production FastAPI Microservice for Civic & Enterprise AI Voice Synthesis.
"""

import os
import time
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from engine.synthesizer import NusaVoxSynthesizer

app = FastAPI(
    title="NusaVox Neural Engine",
    description="High-Performance Autonomous Neural Voice Cloning & Speech Synthesis Microservice",
    version="1.0.0"
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

@app.get("/health")
def health_check():
    cache_dir = synthesizer.cache_dir
    cached_count = len(os.listdir(cache_dir)) if os.path.exists(cache_dir) else 0
    uptime_sec = round(time.time() - START_TIME, 2)
    return {
        "status": "online",
        "engine": "NusaVox Neural Engine",
        "version": "1.0.0",
        "uptime_seconds": uptime_sec,
        "cached_audio_files": cached_count,
        "default_voice": "bupati-madiun"
    }

@app.get("/voices")
def list_voices():
    profiles_file = os.path.join(os.path.dirname(__file__), "profiles", "profiles.json")
    if os.path.exists(profiles_file):
        with open(profiles_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"profiles": []}

@app.post("/v1/audio/speech")
async def openai_speech_endpoint(req: OpenAISpeechRequest):
    """OpenAI API Compatible Speech Endpoint"""
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        audio_bytes, is_cached, normalized_text = await synthesizer.synthesize(
            raw_text=req.input,
            profile_id=req.voice or "bupati-madiun"
        )
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "X-NusaVox-Cached": "true" if is_cached else "false",
                "X-NusaVox-Voice": req.voice or "bupati-madiun",
                "Content-Disposition": 'inline; filename="speech.mp3"'
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
        audio_bytes, is_cached, normalized_text = await synthesizer.synthesize(
            raw_text=req.text,
            profile_id=req.voice or "bupati-madiun",
            pitch=req.pitch,
            rate=req.rate
        )
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "X-NusaVox-Cached": "true" if is_cached else "false",
                "X-NusaVox-Voice": req.voice or "bupati-madiun",
                "Content-Disposition": 'inline; filename="speech.mp3"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def index_web_studio():
    """Interactive NusaVox Voice Studio Playground & SaaS Documentation"""
    return """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NusaVox Studio — Autonomous Civic & Enterprise Neural Voice Engine</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Plus Jakarta Sans', sans-serif; }
    .glass { background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .gold-gradient { background: linear-gradient(135deg, #F59E0B 0%, #D97706 50%, #B45309 100%); }
    .text-gold { color: #F59E0B; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col items-center justify-between p-4 md:p-8">

  <!-- Header -->
  <header class="w-full max-w-4xl flex items-center justify-between py-4 border-b border-slate-800/80 mb-8">
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 rounded-xl gold-gradient flex items-center justify-center font-extrabold text-slate-950 text-xl shadow-lg shadow-amber-500/20">
        N
      </div>
      <div>
        <h1 class="text-xl font-bold tracking-tight">NusaVox <span class="text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">v1.0 API</span></h1>
        <p class="text-xs text-slate-400">Autonomous Neural Voice Engine & Microservice</p>
      </div>
    </div>
    <div class="flex items-center gap-2">
      <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Service Online
      </span>
    </div>
  </header>

  <!-- Interactive Studio Playground -->
  <main class="w-full max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-8 my-auto">
    <!-- Left: Playground Form -->
    <div class="glass p-6 rounded-2xl shadow-2xl">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-lg font-semibold flex items-center gap-2">
          <svg class="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z"></path></svg>
          Interactive Voice Studio
        </h2>
        <span class="text-xs text-slate-400">Live Synthesis</span>
      </div>

      <div class="space-y-4">
        <div>
          <label class="block text-xs font-medium text-slate-300 mb-1.5">Pilih Profil Karakter Suara</label>
          <select id="voiceSelect" class="w-full bg-slate-900 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-sm focus:outline-none focus:border-amber-500 text-slate-200">
            <option value="bupati-madiun" selected>👑 Bapak Bupati Madiun (H. Hari Wuryanto) — Berwibawa</option>
            <option value="civic-formal-male">🏛️ Narator Pelayanan Publik Formal (Pria)</option>
            <option value="civic-friendly-female">🌸 Asisten Helpdesk MPP (Wanita)</option>
          </select>
        </div>

        <div>
          <label class="block text-xs font-medium text-slate-300 mb-1.5">Naskah Teks (Bahasa Indonesia)</label>
          <textarea id="textInput" rows="4" class="w-full bg-slate-900 border border-slate-700/80 rounded-xl p-3.5 text-sm focus:outline-none focus:border-amber-500 text-slate-100 placeholder-slate-500" placeholder="Ketik kalimat atau jawaban SOP yang ingin dibacakan...">Presiden Republik Indonesia saat ini adalah Bapak Prabowo Subianto. Ada yang dapat kami bantu terkait layanan di Mal Pelayanan Publik Kabupaten Madiun?</textarea>
        </div>

        <button id="btnSynthesize" onclick="generateAudio()" class="w-full gold-gradient hover:opacity-95 text-slate-950 font-bold py-3 px-4 rounded-xl shadow-lg shadow-amber-500/25 transition-all flex items-center justify-center gap-2">
          <svg id="playIcon" class="w-5 h-5 fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
          <span id="btnText">Generate & Putar Suara</span>
        </button>
      </div>

      <!-- Audio Player Container -->
      <div id="audioWrapper" class="mt-6 hidden pt-4 border-t border-slate-800">
        <p class="text-xs text-slate-400 mb-2 flex items-center justify-between">
          <span>Hasil Output Audio MP3:</span>
          <span id="cacheTag" class="text-emerald-400 font-mono text-[10px] bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">Cached</span>
        </p>
        <audio id="audioPlayer" controls class="w-full rounded-lg"></audio>
      </div>
    </div>

    <!-- Right: Developer API Integration -->
    <div class="glass p-6 rounded-2xl shadow-2xl flex flex-col justify-between">
      <div>
        <h2 class="text-lg font-semibold flex items-center gap-2 mb-2">
          <svg class="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"></path></svg>
          Integrasi REST API (SaaS Ready)
        </h2>
        <p class="text-xs text-slate-400 mb-4">Gunakan endpoint ini langsung dari PHP, Node.js, Python, atau web apapun:</p>

        <div class="bg-slate-950 rounded-xl p-3.5 border border-slate-800 text-xs font-mono text-slate-300 overflow-x-auto space-y-2">
          <div class="text-slate-500"># OpenAI-Compatible Format:</div>
          <div class="text-amber-400">POST /v1/audio/speech</div>
          <div class="text-slate-400">{</div>
          <div class="pl-4 text-emerald-300">"model": <span class="text-slate-100">"nusavox-civic-1"</span>,</div>
          <div class="pl-4 text-emerald-300">"voice": <span class="text-slate-100">"bupati-madiun"</span>,</div>
          <div class="pl-4 text-emerald-300">"input": <span class="text-slate-100">"Teks panduan resmi..."</span></div>
          <div class="text-slate-400">}</div>
        </div>
      </div>

      <div class="mt-6 pt-4 border-t border-slate-800 text-xs text-slate-400 space-y-1.5">
        <p>✨ <strong>Auto-Caching:</strong> 0% beban CPU untuk query berulang.</p>
        <p>⚡ <strong>Indonesian NLP:</strong> Otomatis mengonversi KTP, SKCK, jam layanan, & rupiah.</p>
        <p>🛡️ <strong>Open Source & Commercial Friendly:</strong> Dilindungi lisensi MIT.</p>
      </div>
    </div>
  </main>

  <!-- Footer -->
  <footer class="w-full max-w-4xl text-center text-xs text-slate-500 py-6 border-t border-slate-800/80 mt-8">
    NusaVox Neural Engine &copy; 2026. Built for Mal Pelayanan Publik & Autonomous Civic Applications.
  </footer>

  <script>
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
      btnText.textContent = 'Memproses Suara...';

      try {
        const res = await fetch('/v1/audio/speech', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ input: text, voice: voice })
        });

        if (!res.ok) throw new Error('Gagal memproses suara.');

        const isCached = res.headers.get('X-NusaVox-Cached') === 'true';
        cacheTag.textContent = isCached ? '⚡ Cached (0 ms)' : '🧠 Synthesized';
        cacheTag.className = isCached 
          ? 'text-emerald-400 font-mono text-[10px] bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20'
          : 'text-amber-400 font-mono text-[10px] bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20';

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
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=17494)
