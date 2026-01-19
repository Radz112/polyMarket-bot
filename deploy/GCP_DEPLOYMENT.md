# Deploying Polymarket Bot to Google Cloud (Free Tier)

This guide walks you through deploying the bot to a free GCP e2-micro VM that runs 24/7.

## Prerequisites

1.  A Google account
2.  A credit card (for verification only - you won't be charged on free tier)
3.  Your Polymarket private key

---

## Step 1: Create a GCP Account & Project

1.  Go to [Google Cloud Console](https://console.cloud.google.com/)
2.  Sign in and accept the terms
3.  Create a new project (e.g., "PolymarketBot")

> [!TIP]
> New accounts get $300 free credits for 90 days. The e2-micro is also part of the "Always Free" tier.

---

## Step 2: Create the VM Instance

1.  Navigate to **Compute Engine → VM Instances**
2.  Click **Create Instance**
3.  Configure:

| Setting | Value |
|---------|-------|
| Name | `polymarket-bot` |
| Region | `us-west1`, `us-central1`, or `us-east1` (free tier regions) |
| Machine type | `e2-micro` (0.25 vCPU, 1 GB RAM) |
| Boot disk | Debian 11, 30GB Standard (free tier) |
| Firewall | ✅ Allow HTTP, ✅ Allow HTTPS |

4.  Click **Create** (takes ~1 minute)

---

## Step 3: Connect to Your VM

1.  Click **SSH** button next to your VM in the console
2.  A browser-based terminal will open

Or from your local terminal:
```bash
gcloud compute ssh polymarket-bot --zone=YOUR_ZONE
```

---

## Step 4: Clone and Set Up the Bot

Run these commands on your VM:

```bash
# Clone your repo (replace with your actual repo URL)
git clone https://github.com/YOUR_USERNAME/PolyMarket-bot.git
cd PolyMarket-bot

# Run the setup script
# This will install Docker, Python, and seed the database
chmod +x deploy/setup.sh
./deploy/setup.sh
```

---

## Step 5: Configure Your Credentials

```bash
# Edit the .env file
nano .env
```

Set these values for live trading:
```
PAPER_TRADING=false
POLYMARKET_PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

> [!CAUTION]
> Never share your private key. Keep it secure!

---

## Step 6: Set Up the Systemd Services

```bash
# Create log directory
sudo mkdir -p /var/log/polymarket-bot
sudo chown $USER:$USER /var/log/polymarket-bot

# Update the service files with your username
sed -i "s/YOUR_USERNAME/$USER/g" deploy/polymarket-bot.service
sed -i "s/YOUR_USERNAME/$USER/g" deploy/polymarket-dashboard.service

# Copy service files
sudo cp deploy/polymarket-bot.service /etc/systemd/system/
sudo cp deploy/polymarket-dashboard.service /etc/systemd/system/

# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable polymarket-bot
sudo systemctl enable polymarket-dashboard

# Start the services
sudo systemctl start polymarket-bot
sudo systemctl start polymarket-dashboard
```

---

## Step 7: Verify It's Running

```bash
# Check bot status
sudo systemctl status polymarket-bot

# Check dashboard status
sudo systemctl status polymarket-dashboard

# View logs (since they are redirected to a file)
tail -f /var/log/polymarket-bot/bot.log
```

---

## Step 8: Access the Dashboard (Optional)

To access the dashboard from your browser:

1.  Go to **VPC Network → Firewall** in GCP Console
2.  Click **Create Firewall Rule**:
   - Name: `allow-dashboard`
   - Targets: All instances
   - Source IP ranges: `0.0.0.0/0` (or your IP for security)
   - Protocols: `tcp:8000`
3.  Click **Create**

Access at: `http://YOUR_VM_EXTERNAL_IP:8000`

---

## Managing the Bot

```bash
# Stop the bot
sudo systemctl stop polymarket-bot

# Restart the bot
sudo systemctl restart polymarket-bot

# View live logs
journalctl -u polymarket-bot -f
```

---

## Cost Breakdown (Free Tier)

| Resource | Free Limit | Your Usage |
|----------|-----------|------------|
| e2-micro VM | 1 instance/month | ✅ Within limit |
| Boot disk | 30 GB Standard | ✅ Within limit |
| Egress | 1 GB/month | ✅ Minimal API calls |

**Expected cost: $0/month** on free tier.

---

## Troubleshooting

### Bot won't start
```bash
# Check logs for errors
tail -f /var/log/polymarket-bot/bot.log
# Or check service status
sudo systemctl status polymarket-bot
```

### Database empty / Bot silent
If logs show the bot starting but doing nothing:
```bash
# Verify database has data
python scripts/seed_db.py
```

### Missing dependencies
```bash
cd ~/PolyMarket-bot
source .venv/bin/activate
pip install -r requirements.txt
```

### Permission denied
```bash
sudo chown -R $USER:$USER ~/PolyMarket-bot
```
