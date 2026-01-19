#!/bin/bash
# ============================================
# Polymarket Bot - GCP Deployment Setup Script
# Run this ONCE after cloning the repo on your VM
# ============================================

set -e

echo "🚀 Setting up Polymarket Bot..."

# Update system
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python and dependencies (uses system default, e.g., 3.11 on Debian 12)
sudo apt-get install -y python3 python3-venv python3-pip git

# Install Node.js 18+ for frontend (optional if you want dashboard)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Docker & Docker Compose
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
    "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
    "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
else
    echo "🐳 Docker already installed."
fi

# Start Docker Services
echo "🔄 Starting Database & Cache..."
sudo docker compose up -d

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install uvicorn websockets

echo "✅ Dependencies installed!"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit your .env file with your real credentials!"
    echo "   nano .env"
    echo ""
    echo "   Required for live trading:"
    echo "   - PAPER_TRADING=false"
    echo "   - POLYMARKET_PRIVATE_KEY=0x..."
    echo ""
fi

# Database Initialization
echo "🗄️  Initializing Database..."
python scripts/init_db.py
echo "🌱 Seeding Database with active markets..."
python scripts/seed_db.py

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit your .env file: nano .env"
echo "2. Install the systemd service: sudo cp deploy/polymarket-bot.service /etc/systemd/system/"
echo "3. Start the bot: sudo systemctl enable polymarket-bot && sudo systemctl start polymarket-bot"
echo "4. Check logs: tail -f /var/log/polymarket-bot/bot.log"
echo ""
