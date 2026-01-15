#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Polymarket Bot - Deployment Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${RED}ERROR: .env file not found!${NC}"
    echo "Please create a .env file with your configuration first."
    echo "You can copy from .env.example: cp .env.example .env"
    exit 1
fi

echo -e "${YELLOW}[1/6] Installing system dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv postgresql postgresql-contrib redis-server

echo -e "${YELLOW}[2/6] Setting up PostgreSQL...${NC}"
# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user (ignore errors if already exists)
sudo -u postgres psql -c "CREATE USER postgres WITH PASSWORD 'postgres';" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE polymarket_bot OWNER postgres;" 2>/dev/null || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE polymarket_bot TO postgres;" 2>/dev/null || true

echo -e "${YELLOW}[3/6] Setting up Redis...${NC}"
sudo systemctl start redis-server
sudo systemctl enable redis-server

echo -e "${YELLOW}[4/6] Setting up Python environment...${NC}"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${YELLOW}[5/6] Initializing database tables...${NC}"
source .venv/bin/activate
PYTHONPATH="$SCRIPT_DIR" python3 scripts/init_db.py

echo -e "${YELLOW}[6/6] Creating systemd service...${NC}"

# Create systemd service file
sudo tee /etc/systemd/system/polymarket-bot.service > /dev/null <<EOF
[Unit]
Description=Polymarket Trading Bot
After=network.target postgresql.service redis-server.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
Environment="PATH=$SCRIPT_DIR/.venv/bin"
Environment="PYTHONPATH=$SCRIPT_DIR"
ExecStart=$SCRIPT_DIR/.venv/bin/python run.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable polymarket-bot

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Commands:"
echo -e "  ${YELLOW}Start bot:${NC}    sudo systemctl start polymarket-bot"
echo -e "  ${YELLOW}Stop bot:${NC}     sudo systemctl stop polymarket-bot"
echo -e "  ${YELLOW}View logs:${NC}    sudo journalctl -u polymarket-bot -f"
echo -e "  ${YELLOW}Check status:${NC} sudo systemctl status polymarket-bot"
echo ""
echo -e "${YELLOW}Starting the bot now...${NC}"
sudo systemctl start polymarket-bot
sudo systemctl status polymarket-bot --no-pager

echo ""
echo -e "${GREEN}Bot is running! Use 'sudo journalctl -u polymarket-bot -f' to view logs.${NC}"
