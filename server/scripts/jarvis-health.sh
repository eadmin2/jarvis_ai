#!/bin/bash
ok(){ printf "%-22s %s\n" "$1" "$2"; }
curl -s  -m 3 http://127.0.0.1:8765/docs -o /dev/null && ok "voice ws (8765)" OK || ok "voice ws (8765)" DOWN
curl -sk -m 3 https://127.0.0.1/hud/ -o /dev/null && ok "HUD (443)" OK || ok "HUD (443)" DOWN
curl -s  -m 3 http://127.0.0.1:9119/ -o /dev/null && ok "dashboard (9119 lo)" OK || ok "dashboard (9119 lo)" DOWN
curl -sk -m 3 https://127.0.0.1:9443/ -o /dev/null && ok "dash proxy (9443)" OK || ok "dash proxy (9443)" DOWN
KEY=$(grep '^API_SERVER_KEY=' ~/.hermes/.env | cut -d= -f2)
curl -s -m 5 -H "Authorization: Bearer $KEY" http://127.0.0.1:8642/health | grep -q ok && ok "hermes api (8642)" OK || ok "hermes api (8642)" DOWN
