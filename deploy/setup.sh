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

# Install Python 3.10+ and dependencies
sudo apt-get install -y python3.10 python3.10-venv python3-pip git

# Install Node.js 18+ for frontend (optional if you want dashboard)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Create virtual environment
python3.10 -m venv .venv
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

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit your .env file: nano .env"
echo "2. Install the systemd service: sudo cp deploy/polymarket-bot.service /etc/systemd/system/"
echo "3. Start the bot: sudo systemctl enable polymarket-bot && sudo systemctl start polymarket-bot"
echo ""
