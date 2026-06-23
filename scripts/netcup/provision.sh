#!/usr/bin/env bash
set -euo pipefail

USERNAME="me"
SSH_PUBKEY="ssh-ed25519 AAAA... you@laptop"
TUNNEL_TOKEN="YOUR_CLOUDFLARE_TUNNEL_TOKEN"
SSH_HOSTNAME="ssh.example.com"

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y openssh-server sudo ca-certificates curl gpg

# Create user
if ! id "${USERNAME}" >/dev/null 2>&1; then
  useradd -m -s /bin/bash -G sudo "${USERNAME}"
fi

install -d -m 700 -o "${USERNAME}" -g "${USERNAME}" "/home/${USERNAME}/.ssh"
printf '%s\n' "${SSH_PUBKEY}" > "/home/${USERNAME}/.ssh/authorized_keys"
chown "${USERNAME}:${USERNAME}" "/home/${USERNAME}/.ssh/authorized_keys"
chmod 600 "/home/${USERNAME}/.ssh/authorized_keys"

cat > "/etc/sudoers.d/90-${USERNAME}" <<EOF
${USERNAME} ALL=(ALL) NOPASSWD:ALL
EOF
chmod 0440 "/etc/sudoers.d/90-${USERNAME}"

# SSH hardening, IPv6 only
install -d -m 755 /etc/ssh/sshd_config.d

cat > /etc/ssh/sshd_config.d/99-hardening.conf <<'EOF'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
PubkeyAuthentication yes
AddressFamily inet6
EOF

sshd -t
systemctl enable --now ssh
systemctl restart ssh

# Store bootstrap config
install -d -m 0700 /root/cloudflare-bootstrap

cat > /root/cloudflare-bootstrap/env <<EOF
TUNNEL_TOKEN="${TUNNEL_TOKEN}"
SSH_HOSTNAME="${SSH_HOSTNAME}"
EOF
chmod 0600 /root/cloudflare-bootstrap/env

# First-boot installer.
# This runs after normal boot and retries automatically.
cat > /root/cloudflare-bootstrap/install-cloudflared.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
. /root/cloudflare-bootstrap/env

echo "[cloudflared-bootstrap] starting"

# Optional but useful on netcup IPv6-only images where 127.0.0.53 has no upstream yet.
install -d -m 0755 /etc/systemd/resolved.conf.d
cat > /etc/systemd/resolved.conf.d/99-ipv6-dns.conf <<'DNS_EOF'
[Resolve]
DNS=2606:4700:4700::1111 2606:4700:4700::1001 2001:4860:4860::8888
FallbackDNS=2606:4700:4700::1111 2001:4860:4860::8888
DNSDefaultRoute=yes
DNS_EOF

systemctl restart systemd-resolved || true

# Do one short check only. If not ready, exit non-zero and systemd retries later.
if ! ip -6 route | grep -q '^default'; then
  echo "[cloudflared-bootstrap] no IPv6 default route yet"
  exit 75
fi

if ! getent ahosts pkg.cloudflare.com >/dev/null 2>&1; then
  echo "[cloudflared-bootstrap] DNS not ready yet"
  exit 75
fi

apt-get update
apt-get install -y curl gpg ca-certificates

install -d -m 0755 /usr/share/keyrings

curl -6 -fsSL "https://pkg.cloudflare.com/cloudflare-main.gpg" -o /tmp/cloudflare-main.gpg

rm -f /usr/share/keyrings/cloudflare-main.gpg
gpg --dearmor -o /usr/share/keyrings/cloudflare-main.gpg /tmp/cloudflare-main.gpg
chmod 0644 /usr/share/keyrings/cloudflare-main.gpg

cat > /etc/apt/sources.list.d/cloudflared.list <<'APT_EOF'
deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main
APT_EOF

apt-get update
apt-get install -y cloudflared

install -d -m 0755 /etc/cloudflared

cat > /etc/cloudflared/token.env <<TOKEN_EOF
TUNNEL_TOKEN="${TUNNEL_TOKEN}"
TOKEN_EOF
chmod 0600 /etc/cloudflared/token.env

cat > /etc/cloudflared/config.yml <<CONFIG_EOF
ingress:
  - hostname: "${SSH_HOSTNAME}"
    service: tcp://localhost:22
  - service: http_status:404
CONFIG_EOF
chmod 0600 /etc/cloudflared/config.yml

cat > /etc/systemd/system/cloudflared-tunnel.service <<'SERVICE_EOF'
[Unit]
Description=Cloudflare Tunnel for SSH
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/cloudflared/token.env
ExecStart=/usr/bin/cloudflared tunnel --edge-ip-version 6 --no-autoupdate --config /etc/cloudflared/config.yml run --token ${TUNNEL_TOKEN}
Restart=always
RestartSec=10s

[Install]
WantedBy=multi-user.target
SERVICE_EOF

systemctl daemon-reload
systemctl enable --now cloudflared-tunnel

# Disable installer only after successful tunnel service creation.
systemctl disable install-cloudflared-after-boot.service || true
rm -f /etc/systemd/system/install-cloudflared-after-boot.service
rm -f /root/cloudflare-bootstrap/install-cloudflared.sh

echo "[cloudflared-bootstrap] success"
EOF

chmod 0700 /root/cloudflare-bootstrap/install-cloudflared.sh

cat > /etc/systemd/system/install-cloudflared-after-boot.service <<'EOF'
[Unit]
Description=Install Cloudflare Tunnel after first boot
After=network-online.target systemd-resolved.service
Wants=network-online.target systemd-resolved.service
StartLimitIntervalSec=0

[Service]
Type=oneshot
ExecStart=/root/cloudflare-bootstrap/install-cloudflared.sh
Restart=on-failure
RestartSec=60s

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable install-cloudflared-after-boot.service

echo "netcup bootstrap completed. cloudflared will install automatically after boot."
