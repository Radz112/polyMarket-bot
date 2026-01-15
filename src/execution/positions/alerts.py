"""
Position alerts for monitoring and risk management.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from src.config.settings import Config
from src.execution.positions.types import (
    Position,
    PositionAlert,
    AlertType,
    AlertSeverity,
)
from src.execution.positions.manager import PositionManager

logger = logging.getLogger(__name__)


class PositionAlerts:
    """
    Generates alerts for position monitoring.
    
    Alert types:
    - P&L threshold exceeded (gain or loss)
    - Position held too long
    - Market resolution approaching
    - Conflicting correlated positions
    """
    
    def __init__(
        self,
        position_manager: PositionManager,
        config: Config
    ):
        """
        Initialize alerts.
        
        Args:
            position_manager: Manager for accessing positions
            config: Configuration with thresholds
        """
        self.position_manager = position_manager
        self.config = config
        
        # Alert thresholds from config
        self.pnl_alert_threshold = config.position_pnl_alert_pct
        self.time_alert_threshold_hours = config.position_time_alert_hours
        self.resolution_warning_hours = config.position_resolution_warning_hours
        
        # Track dispatched alerts to avoid duplicates
        self._dispatched_alerts: set = set()
    
    async def check_alerts(self) -> List[PositionAlert]:
        """
        Check all positions for alert conditions.
        
        Returns:
            List of new alerts (not previously dispatched)
        """
        alerts = []
        
        for position in self.position_manager.get_all_positions():
            # P&L alerts
            pnl_alert = self.check_pnl_alerts(position)
            if pnl_alert and self._is_new_alert(pnl_alert):
                alerts.append(pnl_alert)
            
            # Time alerts
            time_alert = self.check_time_alerts(position)
            if time_alert and self._is_new_alert(time_alert):
                alerts.append(time_alert)
            
            # Resolution alerts
            res_alert = self.check_resolution_alerts(position)
            if res_alert and self._is_new_alert(res_alert):
                alerts.append(res_alert)
            
            # Correlation alerts
            corr_alert = self.check_correlation_alerts(position)
            if corr_alert and self._is_new_alert(corr_alert):
                alerts.append(corr_alert)
        
        # Mark as dispatched
        for alert in alerts:
            self._mark_dispatched(alert)
        
        if alerts:
            logger.info(f"Generated {len(alerts)} position alerts")
        
        return alerts
    
    def check_pnl_alerts(self, position: Position) -> Optional[PositionAlert]:
        """
        Check if position P&L exceeds threshold.
        
        Args:
            position: Position to check
            
        Returns:
            PositionAlert if threshold exceeded, None otherwise
        """
        pnl_pct = position.unrealized_pnl_pct / 100  # Convert to decimal
        
        if pnl_pct > self.pnl_alert_threshold:
            # Large gain - consider taking profit
            return PositionAlert(
                alert_type=AlertType.PNL_GAIN,
                severity=AlertSeverity.INFO,
                position_id=position.id,
                market_id=position.market_id,
                message=f"Position has {position.unrealized_pnl_pct:.1f}% unrealized gain",
                recommended_action="Consider taking partial or full profit",
                current_value=pnl_pct,
                threshold_value=self.pnl_alert_threshold
            )
        
        if pnl_pct < -self.pnl_alert_threshold:
            # Large loss - consider cutting
            severity = AlertSeverity.WARNING
            if pnl_pct < -self.pnl_alert_threshold * 2:
                severity = AlertSeverity.URGENT
            
            return PositionAlert(
                alert_type=AlertType.PNL_LOSS,
                severity=severity,
                position_id=position.id,
                market_id=position.market_id,
                message=f"Position has {position.unrealized_pnl_pct:.1f}% unrealized loss",
                recommended_action="Review position and consider cutting loss",
                current_value=pnl_pct,
                threshold_value=-self.pnl_alert_threshold
            )
        
        return None
    
    def check_time_alerts(self, position: Position) -> Optional[PositionAlert]:
        """
        Check if position has been held too long.
        
        Divergence trades should converge quickly.
        Long holds may indicate the thesis is wrong.
        
        Args:
            position: Position to check
            
        Returns:
            PositionAlert if held too long, None otherwise
        """
        hours_held = position.time_held.total_seconds() / 3600
        threshold = self.time_alert_threshold_hours
        
        if hours_held > threshold * 2:
            # Very long hold
            return PositionAlert(
                alert_type=AlertType.TIME_HELD,
                severity=AlertSeverity.WARNING,
                position_id=position.id,
                market_id=position.market_id,
                message=f"Position held for {hours_held:.1f} hours (threshold: {threshold}h)",
                recommended_action="Reevaluate thesis - divergence may not converge",
                current_value=hours_held,
                threshold_value=threshold
            )
        
        if hours_held > threshold:
            # Threshold exceeded
            return PositionAlert(
                alert_type=AlertType.TIME_HELD,
                severity=AlertSeverity.INFO,
                position_id=position.id,
                market_id=position.market_id,
                message=f"Position held for {hours_held:.1f} hours",
                recommended_action="Monitor closely - consider exit if no convergence",
                current_value=hours_held,
                threshold_value=threshold
            )
        
        return None
    
    def check_resolution_alerts(self, position: Position) -> Optional[PositionAlert]:
        """
        Check if market resolution is approaching.
        
        Args:
            position: Position to check
            
        Returns:
            PositionAlert if near resolution, None otherwise
        """
        if position.distance_to_resolution is None:
            return None
        
        hours_to_resolution = position.distance_to_resolution.total_seconds() / 3600
        warning_threshold = self.resolution_warning_hours
        
        if hours_to_resolution < 1:
            # Very close to resolution
            return PositionAlert(
                alert_type=AlertType.RESOLUTION_NEAR,
                severity=AlertSeverity.URGENT,
                position_id=position.id,
                market_id=position.market_id,
                message=f"Market resolves in {hours_to_resolution*60:.0f} minutes!",
                recommended_action="Decide now: hold through resolution or exit",
                current_value=hours_to_resolution,
                threshold_value=1
            )
        
        if hours_to_resolution < warning_threshold:
            # Approaching resolution
            return PositionAlert(
                alert_type=AlertType.RESOLUTION_NEAR,
                severity=AlertSeverity.WARNING,
                position_id=position.id,
                market_id=position.market_id,
                message=f"Market resolves in {hours_to_resolution:.1f} hours",
                recommended_action="Plan exit strategy before resolution",
                current_value=hours_to_resolution,
                threshold_value=warning_threshold
            )
        
        return None
    
    def check_correlation_alerts(self, position: Position) -> Optional[PositionAlert]:
        """
        Check for conflicting correlated positions.
        
        Example warning: Long Trump PA YES and Long Trump National NO
        
        Args:
            position: Position to check
            
        Returns:
            PositionAlert if conflicts detected, None otherwise
        """
        if not position.correlated_positions:
            return None
        
        conflicts = []
        
        for corr_id in position.correlated_positions:
            corr_pos = self.position_manager.get_position_by_id(corr_id)
            if corr_pos is None:
                continue
            
            # Check if positions are conflicting
            # Same underlying, opposite sides = potential conflict
            if position.side != corr_pos.side:
                conflicts.append(corr_pos)
        
        if conflicts:
            conflict_ids = [c.id for c in conflicts]
            return PositionAlert(
                alert_type=AlertType.CORRELATION_CONFLICT,
                severity=AlertSeverity.WARNING,
                position_id=position.id,
                market_id=position.market_id,
                message=f"Position conflicts with {len(conflicts)} correlated position(s)",
                recommended_action=f"Review positions {conflict_ids} - may cancel out or indicate confusion",
                current_value=len(conflicts),
                threshold_value=0
            )
        
        return None
    
    def _is_new_alert(self, alert: PositionAlert) -> bool:
        """Check if this alert hasn't been dispatched recently."""
        key = self._alert_key(alert)
        return key not in self._dispatched_alerts
    
    def _mark_dispatched(self, alert: PositionAlert) -> None:
        """Mark alert as dispatched."""
        key = self._alert_key(alert)
        self._dispatched_alerts.add(key)
    
    def _alert_key(self, alert: PositionAlert) -> str:
        """Generate unique key for alert deduplication."""
        return f"{alert.position_id}:{alert.alert_type.value}"
    
    def clear_dispatched(self) -> None:
        """Clear dispatched alerts cache."""
        self._dispatched_alerts.clear()
