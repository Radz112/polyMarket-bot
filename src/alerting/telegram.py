"""
Telegram integration for alerts.
"""
import asyncio
import logging
from typing import Optional, Callable, Awaitable, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Optional telegram import
try:
    from telegram import Bot, Update
    from telegram.constants import ParseMode
    from telegram.ext import Application, CommandHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None
    Update = None
    ParseMode = type('ParseMode', (), {'HTML': 'HTML'})()
    Application = None
    CommandHandler = None
    ContextTypes = type('ContextTypes', (), {'DEFAULT_TYPE': None})()
    logger.warning("python-telegram-bot not installed. Telegram features disabled.")


@dataclass
class TelegramConfig:
    """Telegram configuration."""
    bot_token: str
    chat_id: str
    enabled: bool = True


class TelegramClient:
    """
    Telegram client for sending alerts and handling commands.
    
    Uses python-telegram-bot library.
    """
    
    def __init__(self, config: TelegramConfig):
        self.config = config
        self.bot: Optional[Bot] = None
        self._connected = False
        
        if TELEGRAM_AVAILABLE and config.enabled and config.bot_token:
            try:
                self.bot = Bot(token=config.bot_token)
                self._connected = True
                logger.info("Telegram client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if Telegram is available and configured."""
        return TELEGRAM_AVAILABLE and self._connected and self.config.enabled
    
    async def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False
    ) -> bool:
        """
        Send a text message to the configured chat.
        
        Args:
            text: Message text (can include HTML formatting)
            parse_mode: Parse mode (HTML or Markdown)
            disable_notification: If True, send silently
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_available:
            logger.warning("Telegram not available, message not sent")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.config.chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_notification=disable_notification
            )
            logger.debug("Telegram message sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    async def send_signal_alert(self, signal: Any) -> bool:
        """Send formatted signal alert."""
        from .formatters import AlertFormatter
        text = AlertFormatter.format_signal_telegram(signal)
        return await self.send_message(text)
    
    async def send_trade_alert(self, trade: Any) -> bool:
        """Send formatted trade confirmation."""
        from .formatters import AlertFormatter
        text = AlertFormatter.format_trade_telegram(trade)
        return await self.send_message(text)
    
    async def send_risk_alert(self, event: Any) -> bool:
        """Send formatted risk alert."""
        from .formatters import AlertFormatter
        text = AlertFormatter.format_risk_telegram(event)
        return await self.send_message(text)
    
    async def send_position_alert(self, position: Any, reason: str) -> bool:
        """Send formatted position alert."""
        from .formatters import AlertFormatter
        text = AlertFormatter.format_position_telegram(position, reason)
        return await self.send_message(text)
    
    async def send_daily_summary(self, summary: Any) -> bool:
        """Send daily summary."""
        from .formatters import AlertFormatter
        text = AlertFormatter.format_daily_summary_telegram(summary)
        return await self.send_message(text)
    
    async def test_connection(self) -> dict:
        """Test Telegram connection."""
        if not TELEGRAM_AVAILABLE:
            return {"success": False, "error": "python-telegram-bot not installed"}
        
        if not self.config.bot_token:
            return {"success": False, "error": "Bot token not configured"}
        
        try:
            bot = Bot(token=self.config.bot_token)
            me = await bot.get_me()
            
            # Try sending a test message
            await bot.send_message(
                chat_id=self.config.chat_id,
                text="✅ <b>Test successful!</b>\n\nPolymarket Bot is connected.",
                parse_mode="HTML"
            )
            
            return {
                "success": True,
                "bot_username": me.username,
                "bot_name": me.first_name
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class TelegramCommandHandler:
    """
    Handler for incoming Telegram commands.
    
    Supports commands like /status, /positions, /signals, /halt, /resume.
    """
    
    def __init__(
        self,
        config: TelegramConfig,
        get_status: Callable[[], Awaitable[dict]],
        get_positions: Callable[[], Awaitable[list]],
        get_signals: Callable[[], Awaitable[list]],
        execute_trade: Callable[[str], Awaitable[dict]],
        close_position: Callable[[str], Awaitable[dict]],
        halt_trading: Callable[[], Awaitable[bool]],
        resume_trading: Callable[[], Awaitable[bool]],
        get_stats: Callable[[], Awaitable[dict]],
    ):
        self.config = config
        self.get_status = get_status
        self.get_positions = get_positions
        self.get_signals = get_signals
        self.execute_trade = execute_trade
        self.close_position = close_position
        self.halt_trading = halt_trading
        self.resume_trading = resume_trading
        self.get_stats = get_stats
        
        self.application = None
    
    async def start(self):
        """Start the command handler."""
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram not available, command handler not started")
            return
        
        try:
            self.application = Application.builder().token(self.config.bot_token).build()
            
            # Register command handlers
            self.application.add_handler(CommandHandler("status", self._cmd_status))
            self.application.add_handler(CommandHandler("positions", self._cmd_positions))
            self.application.add_handler(CommandHandler("signals", self._cmd_signals))
            self.application.add_handler(CommandHandler("halt", self._cmd_halt))
            self.application.add_handler(CommandHandler("resume", self._cmd_resume))
            self.application.add_handler(CommandHandler("stats", self._cmd_stats))
            self.application.add_handler(CommandHandler("help", self._cmd_help))
            
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            logger.info("Telegram command handler started")
        except Exception as e:
            logger.error(f"Failed to start Telegram command handler: {e}")
    
    async def stop(self):
        """Stop the command handler."""
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram command handler stopped")
    
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        try:
            status = await self.get_status()
            text = f"""📊 <b>BOT STATUS</b>

<b>Trading:</b> {'🟢 Active' if status.get('trading_active') else '🔴 Halted'}
<b>Mode:</b> {status.get('mode', 'Unknown')}
<b>Uptime:</b> {status.get('uptime', 'N/A')}

<b>Today:</b>
P&L: ${status.get('daily_pnl', 0):+.2f}
Trades: {status.get('trades_today', 0)}

<b>Open Positions:</b> {status.get('open_positions', 0)}
<b>Active Signals:</b> {status.get('active_signals', 0)}"""
            await update.message.reply_html(text)
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
    
    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command."""
        try:
            positions = await self.get_positions()
            if not positions:
                await update.message.reply_text("No open positions")
                return
            
            text = "<b>📈 OPEN POSITIONS</b>\n\n"
            for pos in positions[:5]:
                pnl_emoji = "📈" if pos.get('unrealized_pnl', 0) >= 0 else "📉"
                text += f"""<b>{pos.get('market_name', 'Unknown')[:30]}</b>
{pos.get('side', '?')} | ${pos.get('size', 0):.2f} @ {pos.get('entry_price', 0)*100:.1f}¢
{pnl_emoji} ${pos.get('unrealized_pnl', 0):+.2f}

"""
            if len(positions) > 5:
                text += f"<i>+{len(positions)-5} more...</i>"
            
            await update.message.reply_html(text)
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
    
    async def _cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command."""
        try:
            signals = await self.get_signals()
            if not signals:
                await update.message.reply_text("No active signals")
                return
            
            text = "<b>🚨 ACTIVE SIGNALS</b>\n\n"
            for sig in signals[:5]:
                text += f"""Score: <b>{sig.get('score', 0)}</b> | {sig.get('type', 'Unknown')}
Divergence: {sig.get('divergence', 0)*100:.1f}¢
/trade_{sig.get('id', 'xxx')[:8]}

"""
            await update.message.reply_html(text)
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
    
    async def _cmd_halt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /halt command."""
        try:
            success = await self.halt_trading()
            if success:
                await update.message.reply_text("🛑 Trading HALTED")
            else:
                await update.message.reply_text("Failed to halt trading")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
    
    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command."""
        try:
            success = await self.resume_trading()
            if success:
                await update.message.reply_text("✅ Trading RESUMED")
            else:
                await update.message.reply_text("Failed to resume trading")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
    
    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        try:
            stats = await self.get_stats()
            text = f"""📊 <b>PERFORMANCE STATS</b>

<b>Total P&L:</b> ${stats.get('total_pnl', 0):+.2f}
<b>Win Rate:</b> {stats.get('win_rate', 0)*100:.1f}%
<b>Total Trades:</b> {stats.get('total_trades', 0)}

<b>Best Trade:</b> ${stats.get('best_trade', 0):+.2f}
<b>Worst Trade:</b> ${stats.get('worst_trade', 0):+.2f}

<b>Sharpe Ratio:</b> {stats.get('sharpe_ratio', 0):.2f}"""
            await update.message.reply_html(text)
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
    
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        text = """<b>📋 COMMANDS</b>

/status - Current bot status
/positions - Open positions
/signals - Active signals
/stats - Performance statistics
/halt - Emergency halt trading
/resume - Resume trading
/help - This message"""
        await update.message.reply_html(text)
