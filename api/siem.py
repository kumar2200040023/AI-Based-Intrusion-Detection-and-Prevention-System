"""
AI-Based IDS — SIEM & SOAR Integration
============================================
Syslog forwarding, webhook dispatch, and
integration with Splunk/QRadar/Sentinel/XSOAR.
"""

import os
import sys
import json
import socket
import logging
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("antigravity.siem")


# ─────────────────────────────────────────────────────────
# Syslog Forwarder
# ─────────────────────────────────────────────────────────
class SyslogForwarder:
    """Forward alerts via Syslog (RFC 5424) to SIEM platforms."""

    FACILITY_LOCAL0 = 16
    SEVERITY_WARNING = 4
    SEVERITY_CRITICAL = 2

    def __init__(self, host=None, port=None):
        self.host = host or config.SYSLOG_HOST
        self.port = port or config.SYSLOG_PORT
        self.sock = None

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logger.info(f"Syslog forwarder ready → {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Syslog connection failed: {e}")
            return False

    def forward_alert(self, alert_dict):
        """Send alert as syslog message."""
        severity = (self.SEVERITY_CRITICAL
                    if alert_dict.get('threat_score', 0) > 0.8
                    else self.SEVERITY_WARNING)
        priority = self.FACILITY_LOCAL0 * 8 + severity

        msg = (
            f"<{priority}>1 {alert_dict.get('timestamp', '')} "
            f"antigravity-ids - - - "
            f"[alert src_ip=\"{alert_dict.get('src_ip', '')}\" "
            f"dst_ip=\"{alert_dict.get('dst_ip', '')}\" "
            f"attack=\"{alert_dict.get('attack_type', '')}\" "
            f"confidence=\"{alert_dict.get('confidence_score', '')}\" "
            f"score=\"{alert_dict.get('threat_score', '')}\"]"
        )

        try:
            if self.sock:
                self.sock.sendto(msg.encode(), (self.host, self.port))
                return True
        except Exception as e:
            logger.error(f"Syslog send failed: {e}")
        return False

    def close(self):
        if self.sock:
            self.sock.close()


# ─────────────────────────────────────────────────────────
# Webhook Dispatcher
# ─────────────────────────────────────────────────────────
class WebhookDispatcher:
    """Dispatch alerts to SOAR platforms via webhooks."""

    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url or config.WEBHOOK_URL
        self.dispatch_count = 0

    def dispatch(self, alert_dict):
        """Send alert to webhook endpoint."""
        if not self.webhook_url:
            return False

        try:
            import requests
            response = requests.post(
                self.webhook_url,
                json=alert_dict,
                headers={
                    'Content-Type': 'application/json',
                    'X-Source': 'AntiGravity-IDS',
                    'X-Model-Version': config.MODEL_VERSION,
                },
                timeout=5,
            )
            self.dispatch_count += 1
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Webhook dispatch failed: {e}")
            return False


# ─────────────────────────────────────────────────────────
# SIEM Integration Manager
# ─────────────────────────────────────────────────────────
class SIEMIntegration:
    """
    Unified SIEM/SOAR integration manager.
    Supports: Splunk, QRadar, Sentinel, XSOAR
    """

    def __init__(self):
        self.syslog = SyslogForwarder()
        self.webhook = WebhookDispatcher()
        self.alert_log = []

    def initialize(self):
        """Initialize all integration channels."""
        self.syslog.connect()
        logger.info("SIEM Integration initialized")

    def send_alert(self, verdict):
        """
        Send a ThreatVerdict through all configured channels.
        """
        if not verdict.is_threat:
            return

        alert = verdict.to_siem_alert()

        # Log locally
        self.alert_log.append(alert)

        # Forward via syslog
        self.syslog.forward_alert(alert)

        # Dispatch via webhook
        self.webhook.dispatch(alert)

        logger.info(
            f"Alert sent: {alert['attack_type']} "
            f"score={alert['threat_score']:.2f} "
            f"from={alert['src_ip']}"
        )

    def get_recent_alerts(self, limit=50):
        """Get recent alerts."""
        return self.alert_log[-limit:]

    def get_alert_count(self):
        return len(self.alert_log)

    def shutdown(self):
        self.syslog.close()
