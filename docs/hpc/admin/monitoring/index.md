---
title: HPC Web Monitoring
sidebar_label: Monitoring
sidebar_position: 6
---

# HPC Web Monitoring with Grafana + Prometheus

This document covers the complete setup of a web-based monitoring dashboard for the HPC cluster with NVIDIA GPUs.

---

## Architecture Overview

```
Users (Browser) → Grafana (:3000) → Prometheus (:9090) → Exporters
                                                          ├── node_exporter (:9100)       — CPU, RAM, Disk, Network
                                                          ├── nvidia_gpu_exporter (:9835) — GPU metrics
                                                          └── slurm_exporter (:9341)      — SLURM job/node info
```

### Node Roles

| Host | Role | Services |
|------|------|----------|
| Control Node | Head node + Compute | slurmctld, slurmd, Prometheus, Grafana, all exporters |
| Compute Node | Compute only | slurmd, node_exporter, nvidia_gpu_exporter |

---

## Step 1: Install Prometheus

```bash
# Create prometheus user
sudo useradd --no-create-home --shell /bin/false prometheus

# Create directories
sudo mkdir -p /etc/prometheus /var/lib/prometheus
sudo chown prometheus:prometheus /var/lib/prometheus

# Download Prometheus (check https://prometheus.io/download/ for latest version)
cd /tmp
wget https://github.com/prometheus/prometheus/releases/download/v3.12.0/prometheus-3.12.0.linux-amd64.tar.gz
tar xvf prometheus-3.12.0.linux-amd64.tar.gz
cd prometheus-3.12.0.linux-amd64

# Install binaries
sudo cp prometheus promtool /usr/local/bin/
sudo chown prometheus:prometheus /usr/local/bin/prometheus /usr/local/bin/promtool
sudo chown -R prometheus:prometheus /etc/prometheus
```

### Prometheus Configuration

```bash
sudo tee /etc/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["node01:9090"]

  - job_name: "node"
    static_configs:
      - targets: ["node01:9100", "node02:9100"]

  - job_name: "gpu"
    static_configs:
      - targets: ["node01:9835", "node02:9835"]

  - job_name: "slurm"
    static_configs:
      - targets: ["node01:9341"]
EOF
sudo chown prometheus:prometheus /etc/prometheus/prometheus.yml
```

:::tip
Use actual hostnames instead of `localhost` so instance labels in Grafana are readable. Add all compute nodes under the `node` and `gpu` jobs.
:::

### Prometheus Systemd Service

```bash
sudo tee /etc/systemd/system/prometheus.service << 'EOF'
[Unit]
Description=Prometheus Monitoring
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/var/lib/prometheus/ \
  --storage.tsdb.retention.time=90d
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now prometheus
```

:::warning
This service does **not** support `reload`. Always use `sudo systemctl restart prometheus` after config changes.
:::

Verify: visit `http://<your-ip>:9090` — you should see the Prometheus web UI.

---

## Step 2: Install Node Exporter (CPU, RAM, Disk, Network)

```bash
# Create user with explicit UID/GID for consistency across nodes
sudo groupadd --gid 64032 node_exporter
sudo useradd --no-create-home --shell /bin/false --uid 64032 --gid 64032 node_exporter

# Download (check https://github.com/prometheus/node_exporter/releases for latest)
cd /tmp
wget https://github.com/prometheus/node_exporter/releases/download/v1.11.1/node_exporter-1.11.1.linux-amd64.tar.gz
tar xvf node_exporter-1.11.1.linux-amd64.tar.gz
sudo cp node_exporter-1.11.1.linux-amd64/node_exporter /usr/local/bin/
sudo chown node_exporter:node_exporter /usr/local/bin/node_exporter
```

### Node Exporter Systemd Service

```bash
sudo tee /etc/systemd/system/node_exporter.service << 'EOF'
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/usr/local/bin/node_exporter
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now node_exporter
```

Verify:

```bash
curl http://localhost:9100/metrics | head
```

:::note
Install node_exporter on **all nodes** (control node and compute node). Use the same UID/GID (64032) across nodes for consistency.
:::

---

## Step 3: Install NVIDIA GPU Exporter

Uses `nvidia-smi` to collect GPU metrics. No DCGM or Docker dependency required.

Run on **all nodes** (control node and compute node).

```bash
# Install via .deb package (check https://github.com/utkuozdemir/nvidia_gpu_exporter/releases for latest)
cd /tmp
wget https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v1.4.1/nvidia-gpu-exporter_1.4.1_linux_amd64.deb
sudo apt install ./nvidia-gpu-exporter_1.4.1_linux_amd64.deb

# The .deb package installs the binary and systemd service automatically
sudo systemctl enable --now nvidia_gpu_exporter
```

Verify:

```bash
curl http://localhost:9835/metrics | grep nvidia_smi | head
```

This exposes metrics like:

| Metric | Description |
|--------|-------------|
| `nvidia_smi_utilization_gpu` | GPU utilization % |
| `nvidia_smi_memory_used` | GPU memory used (MiB) |
| `nvidia_smi_memory_free` | GPU memory free (MiB) |
| `nvidia_smi_temperature_gpu` | GPU temperature |
| `nvidia_smi_power_draw` | Power draw (W) |

---

## Step 4: Install Prometheus SLURM Exporter

Run on the **control node only**. The exporter calls `squeue` and `sinfo` locally.

```bash
# Install Go if not already present
sudo apt-get install -y golang-go

# Clone and build
cd /tmp
git clone https://github.com/vpenso/prometheus-slurm-exporter.git
cd prometheus-slurm-exporter
make

# Install
sudo cp bin/prometheus-slurm-exporter /usr/local/bin/
```

### SLURM Exporter Systemd Service

```bash
sudo tee /etc/systemd/system/slurm-exporter.service << 'EOF'
[Unit]
Description=Prometheus SLURM Exporter
After=slurmctld.service
Wants=slurmctld.service

[Service]
Type=simple
ExecStart=/usr/local/bin/prometheus-slurm-exporter -listen-address=:9341
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now slurm-exporter
```

Verify:

```bash
curl http://localhost:9341/metrics | grep slurm
```

This exports metrics like:

| Metric | Description |
|--------|-------------|
| `slurm_queue_pending` / `slurm_queue_running` | Job counts |
| `slurm_node_alloc` / `slurm_node_idle` | Node states |
| `slurm_cpus_total` / `slurm_cpus_alloc` | CPU allocation |

:::note
The SLURM exporter returns metrics per partition. The Grafana dashboard wraps all SLURM queries with `sum()` to show cluster-wide totals rather than per-partition duplicates.
:::

---

## Step 5: Install Grafana

```bash
# Add Grafana APT repository
sudo apt-get install -y apt-transport-https software-properties-common
sudo mkdir -p /etc/apt/keyrings/
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list

# Install
sudo apt-get update
sudo apt-get install -y grafana

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now grafana-server
```

:::note
This guide was tested with Grafana 13. Check `grafana-server -v` after installation to confirm your version.
:::

Grafana is now available at `http://<your-ip>:3000`

Default login: `admin` / `admin` (you'll be prompted to change the password on first login)

### Resetting the Admin Password

If you forget the Grafana admin password:

```bash
sudo grafana-cli admin reset-admin-password <newpassword>
sudo systemctl restart grafana-server
```

---

## Step 6: Configure Grafana

### Set `root_url`

Edit `grafana.ini` to set the server's IP before doing anything else:

```bash
sudo sed -i 's|^;root_url = .*|root_url = http://192.168.220.75/|' /etc/grafana/grafana.ini
sudo systemctl restart grafana-server
```

### Add Prometheus Data Source

1. Log into Grafana at `http://<your-ip>:3000`
2. Go to **Connections → Data sources → Add data source**
3. Select **Prometheus**
4. Set URL to `http://localhost:9090`
5. Click **Save & Test** — should show "Successfully queried the Prometheus API"

### Import HPC Overview Dashboard

Download the dashboard JSON file: [hpc-overview-nas.json](/hpc/hpc-overview-nas.json)

:::warning
The dashboard JSON has the datasource UID hardcoded. Since your Prometheus datasource UID will likely differ from the one in the file, you **must** update it before importing — otherwise all panels will show "No data".

To get your Prometheus datasource UID:

```bash
curl -s http://admin:<password>@localhost:3000/api/datasources | python3 -m json.tool | grep -E '"uid"|"name"'
```

Then replace all occurrences of the existing UID in the JSON with your actual UID:

```bash
sed 's/<existing-uid>/<your-uid>/g' hpc-overview-nas.json > hpc-overview-nas-fixed.json
```

Do **not** use a `${datasource}` template variable — public dashboards do not support template variables.
:::

To import:

1. Go to **Dashboards → New → Import**
2. Click **Upload dashboard JSON file** and select the updated `hpc-overview-nas-fixed.json`
3. Click **Import**

### Dashboard Structure

```
Control Node · System Overview
  CPU Usage, RAM Usage, Disk Usage (/), Disk Usage (/home) — gauges
  Disk Used/Total, Total RAM, Total CPU Cores — stats
  CPU & RAM Over Time, Network Traffic — time series

Control Node · GPU
  GPU Utilization, GPU Memory Usage — gauges
  GPU Temperature, GPU Power Draw, GPU Memory Used/Total — stats
  GPU Utilization Over Time, GPU Memory Over Time — time series

Compute Node · System Overview
  (same panels as above, filtered to compute node)

Compute Node · GPU
  (same panels as above, filtered to compute node)

SLURM Jobs
  Jobs Running, Jobs Pending, CPUs Allocated, CPUs Idle, CPUs Total, Nodes Idle
  SLURM Jobs Over Time, SLURM CPU Allocation Over Time
```

Each node's panels are filtered by Prometheus instance label so they display independently.

---

## Step 7: Enable Public Dashboard

Instead of anonymous access, Grafana's Public Dashboard feature is used to share the HPC overview with users. This exposes only the specific dashboard via a unique token URL — no Grafana UI, no ability to browse other dashboards or modify queries.

1. Open the HPC Overview dashboard
2. Click the **Share** button (top right)
3. Go to the **Share externally** tab
4. Click **Generate public URL**
5. Copy the generated URL — it will look like: `http://192.168.220.75/public-dashboards/<token>`

:::note
- Public dashboards do **not** support template variables. Ensure the dashboard JSON uses hardcoded datasource UIDs (see Step 6).
- The public dashboard inherits the default time range set on the dashboard itself. To change it, update the time picker on the dashboard, save it, and the public view will reflect the new default.
:::

---

## Step 8: Configure Apache Reverse Proxy

Apache proxies port 80 → Grafana's port 3000, and redirects root `/` directly to the public dashboard URL so users don't need to know the port or token.

```bash
# Enable required Apache modules
sudo a2enmod proxy proxy_http rewrite

# Disable the default Apache site
sudo a2dissite 000-default

# Create Grafana site configuration
sudo tee /etc/apache2/sites-available/grafana.conf << 'EOF'
<VirtualHost *:80>
    ServerName 192.168.220.75

    ProxyPreserveHost On
    RewriteEngine On

    # Redirect root to public dashboard
    RewriteCond %{REQUEST_URI} ^/$
    RewriteRule ^/$ /public-dashboards/<your-token-here> [R=302,L]

    ProxyPass / http://localhost:3000/
    ProxyPassReverse / http://localhost:3000/

    <Proxy *>
        Require all granted
    </Proxy>

    ErrorLog ${APACHE_LOG_DIR}/grafana_error.log
    CustomLog ${APACHE_LOG_DIR}/grafana_access.log combined
</VirtualHost>
EOF

# Enable the new site
sudo a2ensite grafana

# Test configuration
sudo apache2ctl configtest

# If "Syntax OK", restart Apache
sudo systemctl restart apache2
```

:::note
Replace `192.168.220.75` with your actual server IP and `<your-token-here>` with the token from Step 7.
:::

Result: Users can now visit `http://<your-ip>` and immediately see the HPC monitoring dashboard.

### Note on WebSocket Logs

After setting up the Apache proxy, you will see repeated entries like this in the Grafana logs:

```
path=/api/live/ws status=400
```

This is **normal and harmless**. Apache's default proxy configuration does not support WebSocket upgrades, so Grafana's live streaming endpoint always returns 400. The dashboard still refreshes correctly on its own interval.

---

## Step 9: Security & Access Configuration

### Firewall

```bash
# Allow Grafana port from your internal network
sudo ufw allow from 192.168.0.0/16 to any port 3000

# The following ports should NOT be exposed externally:
# - Port 9090 (Prometheus)
# - Port 9100 (Node Exporter)
# - Port 9835 (NVIDIA GPU Exporter)
# - Port 9341 (SLURM Exporter)
```

### Quick Reference — Ports

| Service | Port | Access |
|---------|------|--------|
| Apache (reverse proxy) | 80 | Open to LAN |
| Grafana (direct access) | 3000 | Open to LAN (or localhost only if using Apache) |
| Prometheus | 9090 | Localhost only |
| Node Exporter | 9100 | Localhost only |
| NVIDIA GPU Exporter | 9835 | Localhost only |
| SLURM Exporter | 9341 | Localhost only |

---

## Verification Checklist

```bash
# Check all services are running
sudo systemctl status prometheus node_exporter nvidia_gpu_exporter slurm-exporter grafana-server apache2

# Check exporters are serving metrics
curl -s http://localhost:9100/metrics | head -5   # Node Exporter
curl -s http://localhost:9835/metrics | head -5   # NVIDIA GPU Exporter
curl -s http://localhost:9341/metrics | head -5   # SLURM Exporter

# Check Prometheus can reach all targets (all should show "health": "up")
curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool | grep -E '"instance"|"health"'

# If using Apache reverse proxy, test the redirect
curl -I http://localhost/   # Should show 302 redirect to public dashboard
```

Expected targets and their health:

| Instance | Job | Expected |
|----------|-----|----------|
| `node01:9090` | prometheus | up |
| `node01:9100` | node | up |
| `node02:9100` | node | up |
| `node01:9835` | gpu | up |
| `node02:9835` | gpu | up |
| `node01:9341` | slurm | up |

---

## Adding More Compute Nodes

To add a new node (e.g. a third workstation):

1. On the new node, install `node_exporter` and `nvidia_gpu_exporter` following Steps 2 and 3 above. Use the same UID/GID for `node_exporter` as on existing nodes:

```bash
sudo groupadd --gid 64032 node_exporter
sudo useradd --no-create-home --shell /bin/false --uid 64032 --gid 64032 node_exporter
```

2. On the control node, add the new node to `/etc/prometheus/prometheus.yml`:

```yaml
- job_name: "node"
  static_configs:
    - targets: ["node01:9100", "node02:9100", "exxact3:9100"]

- job_name: "gpu"
  static_configs:
    - targets: ["node01:9835", "node02:9835", "exxact3:9835"]
```

Then restart Prometheus:

```bash
sudo systemctl restart prometheus
```

3. Update the `hpc-overview-nas.json` to add a new node section, following the same per-node panel structure (System Overview + GPU rows).

Grafana will automatically start showing the new node once Prometheus is scraping it.

---

## Uninstallation

To completely remove the monitoring stack:

### 1. Stop and Disable All Services

```bash
sudo systemctl stop grafana-server prometheus node_exporter nvidia_gpu_exporter slurm-exporter
sudo systemctl disable grafana-server prometheus node_exporter nvidia_gpu_exporter slurm-exporter

# Restore default Apache site if configured
sudo a2dissite grafana
sudo a2ensite 000-default
sudo systemctl restart apache2
```

### 2. Remove Systemd Service Files

```bash
sudo rm /etc/systemd/system/prometheus.service
sudo rm /etc/systemd/system/node_exporter.service
sudo rm /etc/systemd/system/slurm-exporter.service
# nvidia_gpu_exporter.service is managed by the .deb package
sudo systemctl daemon-reload
```

### 3. Remove Binaries

```bash
sudo rm /usr/local/bin/prometheus
sudo rm /usr/local/bin/promtool
sudo rm /usr/local/bin/node_exporter
sudo rm /usr/local/bin/prometheus-slurm-exporter
```

### 4. Remove Configuration Files and Data

```bash
sudo rm -rf /etc/prometheus
sudo rm -rf /var/lib/prometheus

sudo apt-get remove --purge grafana
sudo rm -rf /etc/grafana /var/lib/grafana /var/log/grafana

sudo apt-get remove --purge nvidia-gpu-exporter
sudo rm /etc/apache2/sites-available/grafana.conf
```

### 5. Remove System Users

```bash
sudo userdel prometheus
sudo userdel node_exporter
# Grafana user is removed automatically when the package is purged
```

### 6. Clean Up (Optional)

```bash
# Remove downloaded files
rm -f /tmp/prometheus-*.tar.gz
rm -f /tmp/node_exporter-*.tar.gz
rm -f /tmp/nvidia-gpu-exporter_*.deb
rm -rf /tmp/prometheus-slurm-exporter

# Remove Grafana repository
sudo rm /etc/apt/sources.list.d/grafana.list
sudo rm /etc/apt/keyrings/grafana.gpg
sudo apt-get update
sudo apt-get autoremove
```

### Verify Uninstallation

```bash
# Check if any monitoring services are still running
sudo systemctl status prometheus node_exporter nvidia_gpu_exporter slurm-exporter grafana-server

# Check if binaries are removed
which prometheus promtool node_exporter prometheus-slurm-exporter

# Check if ports are no longer in use
sudo ss -tlnp | grep -E ':(3000|9090|9100|9341|9835)'
```

:::note
This uninstallation does **not** remove Apache itself, NTP time synchronization, or any firewall rules you may have added.
:::

---

## Troubleshooting

### Dashboard Shows Blank or Keeps Loading Forever

**Cause:** The dashboard JSON was corrupted, imported with a mismatched datasource UID, or the dashboard was accidentally deleted from Grafana.

**Fix:**

1. Check if the dashboard exists:

```bash
curl -s http://admin:<password>@localhost:3000/api/dashboards/uid/hpc-overview | python3 -m json.tool | grep title
```

2. If it returns `"Access denied"` or `"Dashboard not found"`, re-import the JSON via **Dashboards → New → Import**.

### Public Dashboard Shows "No Data"

**Cause:** The dashboard JSON uses a `${datasource}` template variable. Public dashboards do not support template variables, so all queries fail silently.

**Fix:**

1. Get your Prometheus datasource UID:

```bash
curl -s http://admin:<password>@localhost:3000/api/datasources | python3 -m json.tool | grep -E '"uid"|"name"'
```

2. Replace all occurrences of `${datasource}` in the dashboard JSON with the actual UID:

```bash
sed 's/\${datasource}/<your-uid>/g' hpc-overview-nas.json > hpc-overview-nas-fixed.json
```

3. Also remove the `templating` variable from the JSON, then re-import.

### "Server Time is Out of Sync"

**Cause:** The system clock is drifting.

**Fix:** Enable NTP synchronization:

```bash
sudo apt install -y systemd-timesyncd
sudo timedatectl set-ntp true
timedatectl status
# Should show: System clock synchronized: yes
```

**If NTP times out (corporate firewall):** Your firewall may be blocking outbound NTP traffic (UDP port 123). Use the default gateway as the NTP server instead:

```bash
# Find your gateway IP
ip route | grep default

# Configure it as the NTP source
sudo sed -i 's/^#NTP=.*/NTP=192.168.220.1/' /etc/systemd/timesyncd.conf
sudo systemctl restart systemd-timesyncd

# Verify after a few seconds
sleep 5
timedatectl timesync-status
# Should show: Packet count >= 1, and a small Offset value
```

Once synced, the Prometheus warning will disappear on the next page refresh.
