"""
Email integration for alerts.
"""
import asyncio
import logging
from typing import Optional, List
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional aiosmtplib import
try:
    import aiosmtplib
    AIOSMTPLIB_AVAILABLE = True
except ImportError:
    AIOSMTPLIB_AVAILABLE = False
    logger.warning("aiosmtplib not installed. Email features disabled.")


@dataclass
class EmailConfig:
    """Email configuration."""
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    from_addr: str
    to_addrs: List[str]
    use_tls: bool = True
    enabled: bool = True


class EmailClient:
    """
    Email client for sending alerts and reports.
    
    Uses aiosmtplib for async email sending.
    """
    
    def __init__(self, config: EmailConfig):
        self.config = config
        self._connected = AIOSMTPLIB_AVAILABLE and config.enabled
    
    @property
    def is_available(self) -> bool:
        """Check if email is available and configured."""
        return (
            AIOSMTPLIB_AVAILABLE and 
            self.config.enabled and 
            self.config.smtp_host and 
            self.config.username
        )
    
    async def send_email(
        self,
        subject: str,
        body_text: str,
        body_html: Optional[str] = None,
        to_addrs: Optional[List[str]] = None
    ) -> bool:
        """
        Send an email.
        
        Args:
            subject: Email subject
            body_text: Plain text body
            body_html: Optional HTML body
            to_addrs: Override recipient addresses
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_available:
            logger.warning("Email not available, message not sent")
            return False
        
        recipients = to_addrs or self.config.to_addrs
        
        try:
            # Build message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.config.from_addr
            message["To"] = ", ".join(recipients)
            
            # Add text part
            message.attach(MIMEText(body_text, "plain"))
            
            # Add HTML part if provided
            if body_html:
                message.attach(MIMEText(body_html, "html"))
            
            # Send
            await aiosmtplib.send(
                message,
                hostname=self.config.smtp_host,
                port=self.config.smtp_port,
                username=self.config.username,
                password=self.config.password,
                use_tls=self.config.use_tls,
            )
            
            logger.info(f"Email sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    async def send_daily_report(self, summary: "DailySummary") -> bool:
        """Send daily performance report."""
        from .formatters import AlertFormatter, DailySummary
        
        subject = f"Polymarket Bot Daily Report - {summary.date.strftime('%b %d, %Y')}"
        body_text = AlertFormatter.format_daily_email_text(summary)
        body_html = AlertFormatter.format_daily_email_html(summary)
        
        return await self.send_email(subject, body_text, body_html)
    
    async def send_risk_alert(self, event: any) -> bool:
        """Send risk alert email."""
        severity_prefix = {
            "warning": "⚠️",
            "high": "🚨",
            "critical": "🔴",
        }.get(getattr(event, 'severity', 'warning'), "⚠️")
        
        subject = f"{severity_prefix} Risk Alert - {getattr(event, 'alert_type', 'Unknown')}"
        
        body_text = f"""
POLYMARKET BOT - RISK ALERT
{'='*40}

Type: {getattr(event, 'alert_type', 'Unknown')}
Severity: {getattr(event, 'severity', 'Unknown').upper()}
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

{getattr(event, 'message', 'No message provided')}

Please review the bot status immediately.

---
This is an automated alert from Polymarket Bot.
"""
        
        body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, sans-serif; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 500px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; }}
        .header {{ background: #ef4444; padding: 20px; color: white; text-align: center; }}
        .content {{ padding: 30px; }}
        .severity {{ display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; }}
        .severity.warning {{ background: #fef3c7; color: #d97706; }}
        .severity.high {{ background: #fed7aa; color: #ea580c; }}
        .severity.critical {{ background: #fee2e2; color: #dc2626; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚠️ Risk Alert</h1>
        </div>
        <div class="content">
            <p><strong>Type:</strong> {getattr(event, 'alert_type', 'Unknown')}</p>
            <p><strong>Severity:</strong> 
                <span class="severity {getattr(event, 'severity', 'warning')}">{getattr(event, 'severity', 'Unknown').upper()}</span>
            </p>
            <p><strong>Time:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            <hr>
            <p>{getattr(event, 'message', 'No message provided')}</p>
            <hr>
            <p><em>Please review the bot status immediately.</em></p>
        </div>
    </div>
</body>
</html>
"""
        
        return await self.send_email(subject, body_text, body_html)
    
    async def test_connection(self) -> dict:
        """Test email connection."""
        if not AIOSMTPLIB_AVAILABLE:
            return {"success": False, "error": "aiosmtplib not installed"}
        
        if not self.config.smtp_host:
            return {"success": False, "error": "SMTP host not configured"}
        
        try:
            # Test SMTP connection
            smtp = aiosmtplib.SMTP(
                hostname=self.config.smtp_host,
                port=self.config.smtp_port,
                use_tls=self.config.use_tls
            )
            await smtp.connect()
            await smtp.login(self.config.username, self.config.password)
            await smtp.quit()
            
            # Send test email
            await self.send_email(
                subject="Polymarket Bot - Email Test",
                body_text="Email connection test successful!",
                body_html="<h3>✅ Email connection test successful!</h3><p>Polymarket Bot can send emails.</p>"
            )
            
            return {"success": True, "message": "Test email sent"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
