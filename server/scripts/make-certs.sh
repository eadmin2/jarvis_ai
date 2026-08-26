#!/bin/bash
# Self-signed TLS cert for the HUD. Add your hostnames/IPs to the SAN list.
cd "$(dirname "$0")/.."
mkdir -p certs
HOST_IP=${1:-$(ipconfig getifaddr en0 2>/dev/null || echo 127.0.0.1)}
openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem -days 825 -nodes \
  -subj "/CN=Jarvis HUD" \
  -addext "subjectAltName=DNS:jarvis.local,DNS:jarvis,DNS:localhost,IP:$HOST_IP"
cp certs/cert.pem hud/jarvis.cer   # downloadable from the HUD for phones
echo "cert created for IP $HOST_IP — trust certs/cert.pem on your devices"
