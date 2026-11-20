# NusaVox Neural Engine

NusaVox is a high-performance autonomous neural text-to-speech (TTS) and voice synthesis microservice designed for Indonesian civic and public service applications.

## Features

- **OpenAI Compatible Endpoint**: Drop-in replacement via `/v1/audio/speech`.
- **Intelligent Speech Preprocessing**: Normalizes Indonesian acronyms (KTP, SKCK, OSS, NIB), currency, and operational schedules.
- **Smart Disk Cache**: Identical queries are cached permanently on disk, eliminating redundant CPU inference.
- **Interactive Web Studio**: Integrated web playground with instant live playback.

## Quick Start

### Installation

```bash
git clone https://github.com/haamzz93-sudo/nusavox-engine.git
cd nusavox-engine
pip install -r requirements.txt
python main.py
```

The service will start on port `17494`.

### API Usage

#### OpenAI-Compatible Request

```bash
curl -X POST http://127.0.0.1:17494/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Halo, selamat datang di layanan informasi perizinan resmi.",
    "voice": "bupati-madiun"
  }' \
  --output speech.mp3
```

#### Health Check

```bash
curl http://127.0.0.1:17494/health
```

## License

MIT License.
