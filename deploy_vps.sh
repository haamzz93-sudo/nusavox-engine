#!/usr/bin/env bash
# ==============================================================================
# NusaVox Neural Engine — Production VPS Deployment Script
# ==============================================================================
set -e

echo "🚀 [NusaVox] Deploying NusaVox Neural Engine on VPS..."

# Ensure python3 and pip are available
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Installing python3 and pip..."
    sudo apt-get update && sudo apt-get install -y python3 python3-pip ffmpeg
fi

# Install python requirements
echo "📦 [NusaVox] Installing dependencies..."
pip3 install -r requirements.txt

# Start or restart via PM2
if command -v pm2 &> /dev/null; then
    echo "⚡ [NusaVox] Managing process with PM2..."
    pm2 stop nusavox-engine 2>/dev/null || true
    pm2 delete nusavox-engine 2>/dev/null || true
    pm2 start main.py --name "nusavox-engine" --interpreter python3
    pm2 save
    echo "✅ [NusaVox] Successfully started via PM2 on port 17494!"
else
    echo "ℹ️  pm2 not found. Starting in background via nohup..."
    nohup python3 main.py > nusavox.log 2>&1 &
    echo "✅ [NusaVox] Started in background on port 17494!"
fi

echo "🎉 [NusaVox] Engine online at http://127.0.0.1:17494"
