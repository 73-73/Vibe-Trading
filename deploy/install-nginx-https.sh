#!/usr/bin/env bash
# Run manually from an SSH terminal as user ubuntu:
#   bash /home/ubuntu/vibe-trading/deploy/install-nginx-https.sh
# It prompts locally for sudo; do NOT paste a sudo password into chat.
set -euo pipefail

PROJECT=/home/ubuntu/vibe-trading
SITE_SRC="$PROJECT/deploy/nginx-vibe-trading-https.conf"
SITE_DST=/etc/nginx/sites-available/vibe-trading-https
SITE_LINK=/etc/nginx/sites-enabled/vibe-trading-https
CERT_DIR=/etc/nginx/ssl
CERT="$CERT_DIR/nginx-selfsigned.crt"
KEY="$CERT_DIR/nginx-selfsigned.key"
IP=124.221.98.219
STAMP=$(date +%Y%m%d-%H%M%S)

if [[ ! -f "$SITE_SRC" ]]; then
  echo "Missing Nginx source config: $SITE_SRC" >&2
  exit 1
fi

sudo install -d -m 0755 "$CERT_DIR"

# Existing certificate has no Subject Alternative Name. Modern browsers validate
# the IP address against SAN, so replace it with an IP-SAN self-signed cert.
if [[ -e "$CERT" || -e "$KEY" ]]; then
  sudo mkdir -p "$CERT_DIR/backup-$STAMP"
  [[ -e "$CERT" ]] && sudo cp -a "$CERT" "$CERT_DIR/backup-$STAMP/"
  [[ -e "$KEY" ]] && sudo cp -a "$KEY" "$CERT_DIR/backup-$STAMP/"
fi
sudo openssl req -x509 -nodes -newkey rsa:2048 -sha256 -days 365 \
  -keyout "$KEY" -out "$CERT" \
  -subj "/C=CN/ST=Shanghai/L=Shanghai/O=Vibe-Trading/CN=$IP" \
  -addext "subjectAltName = IP:$IP"
sudo chmod 600 "$KEY"
sudo chmod 644 "$CERT"

# Nginx owns only 443. Caddy continues to own port 80.
# Disable Ubuntu's stock default site: it listens on port 80 and would prevent
# the entire Nginx service (including this 443 vhost) from starting.
sudo install -m 0644 "$SITE_SRC" "$SITE_DST"
sudo rm -f /etc/nginx/sites-enabled/default
sudo rm -f "$SITE_LINK"
sudo ln -s "$SITE_DST" "$SITE_LINK"

sudo nginx -t
sudo systemctl reset-failed nginx
sudo systemctl enable nginx
sudo systemctl restart nginx

printf '\nNginx HTTPS proxy is active. Verification:\n'
sudo ss -ltnp '( sport = :443 )'
curl -kfsS https://127.0.0.1/live -H 'Host: 124.221.98.219' || true
printf '\n\nOpen from your MacBook: https://124.221.98.219\n'
printf 'The browser warning is expected because the certificate is self-signed.\n'
printf 'In the Vibe-Trading Settings page, enter the Server API key configured on the server.\n'
