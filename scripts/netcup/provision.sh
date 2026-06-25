#!/usr/bin/env bash
set -euo pipefail

USERNAME="me"
SSH_PUBKEY="ssh-ed25519 AAAA... you@laptop"
TUNNEL_TOKEN="YOUR_CLOUDFLARE_TUNNEL_TOKEN"
SSH_HOSTNAME="ssh.example.com"

export DEBIAN_FRONTEND=noninteractive

# Fix DNS for netcup IPv6-only install environment
install -d -m 0755 /etc/systemd/resolved.conf.d

cat > /etc/systemd/resolved.conf.d/99-ipv6-dns.conf <<'EOF'
[Resolve]
DNS=2606:4700:4700::1111 2606:4700:4700::1001 2001:4860:4860::8888
FallbackDNS=2606:4700:4700::1111 2001:4860:4860::8888
DNSDefaultRoute=yes
EOF

systemctl restart systemd-resolved || true
sleep 3

apt-get update
apt-get install -y curl gpg ca-certificates openssh-server sudo

# Create user
if ! id "${USERNAME}" >/dev/null 2>&1; then
  useradd -m -s /bin/bash -G sudo "${USERNAME}"
fi

install -d -m 700 -o "${USERNAME}" -g "${USERNAME}" "/home/${USERNAME}/.ssh"
printf '%s\n' "${SSH_PUBKEY}" > "/home/${USERNAME}/.ssh/authorized_keys"
chown "${USERNAME}:${USERNAME}" "/home/${USERNAME}/.ssh/authorized_keys"
chmod 600 "/home/${USERNAME}/.ssh/authorized_keys"

# Passwordless sudo
cat > "/etc/sudoers.d/90-${USERNAME}" <<EOF
${USERNAME} ALL=(ALL) NOPASSWD:ALL
EOF
chmod 0440 "/etc/sudoers.d/90-${USERNAME}"

# SSH hardening
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

# Install cloudflared from Cloudflare APT repo
install -d -m 0755 /usr/share/keyrings

curl -fsSL "https://pkg.cloudflare.com/cloudflare-main.gpg" \
  -o /tmp/cloudflare-main.gpg

gpg --dearmor \
  -o /usr/share/keyrings/cloudflare-main.gpg \
  /tmp/cloudflare-main.gpg

chmod 0644 /usr/share/keyrings/cloudflare-main.gpg

cat > /etc/apt/sources.list.d/cloudflared.list <<'EOF'
deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main
EOF

apt-get update
apt-get install -y cloudflared

# Configure cloudflared
install -d -m 0755 /etc/cloudflared

cat > /etc/cloudflared/token.env <<EOF
TUNNEL_TOKEN="${TUNNEL_TOKEN}"
EOF
chmod 0600 /etc/cloudflared/token.env

cat > /etc/systemd/system/cloudflared-tunnel.service <<'EOF'
[Unit]
Description=Cloudflare Tunnel for SSH
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/cloudflared/token.env
ExecStart=/usr/bin/cloudflared tunnel --edge-ip-version 6 --no-autoupdate run --token ${TUNNEL_TOKEN}
Restart=always
RestartSec=10s

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now cloudflared-tunnel

echo "netcup post-install script completed successfully."