#!/bin/bash
# One-time generation of the boot greeting audio via ElevenLabs (~110 characters).
cd "$(dirname "$0")/.."
mkdir -p hud/audio
EK=$(grep -E '^(ELEVENLABS_API_KEY|ELEVEN_API_KEY|XI_API_KEY)=' ~/.hermes/.env | head -1 | cut -d= -f2)
VOICE=$(grep 'voice_id:' config/server.yaml | awk '{print $2}')
NAME=${1:-there}
gen() {
  curl -s -m 30 -X POST "https://api.elevenlabs.io/v1/text-to-speech/$VOICE" \
    -H "xi-api-key: $EK" -H "Content-Type: application/json" \
    -d "{\"text\":\"$2\",\"model_id\":\"eleven_flash_v2_5\"}" -o "hud/audio/boot_$1.mp3"
}
gen morning   "Systems online. Good morning, $NAME."
gen afternoon "Systems online. Good afternoon, $NAME."
gen evening   "Systems online. Good evening, $NAME."
ls -la hud/audio/
