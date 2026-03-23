"""
AI-Based IPS — Intrusion Prevention System Engine
=================================================
Core engine for managing blocked IP addresses and preventing threats.
"""

import time
import logging
import datetime
from threading import Lock

logger = logging.getLogger("antigravity.ips")

class IPSEngine:
    def __init__(self, mode="simulated", block_duration=3600):
        """
        Initialize the IPS Engine.
        
        Args:
            mode (str): "simulated" to drop at API level, or "os" for system firewall blocks.
            block_duration (int): Default duration in seconds to keep an IP blocked.
        """
        self.mode = mode.lower()
        self.block_duration = block_duration
        
        # In-memory blocklist: { "ip_address": { "reason": str, "expires_at": int } }
        self._blocklist = {}
        self._lock = Lock()
        logger.info(f"IPS Engine initialized in '{self.mode}' mode.")

    def block_ip(self, ip_address, reason="Malicious activity detected", duration=None):
        """Block an IP address."""
        if duration is None:
            duration = self.block_duration
            
        expires_at = int(time.time()) + duration
        
        with self._lock:
            self._blocklist[ip_address] = {
                "reason": reason,
                "expires_at": expires_at,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            
        logger.warning(f"🚨 [IPS BLOCK] {ip_address} blocked for {duration}s. Reason: {reason}")
        
        if self.mode == "os":
            self._execute_os_block(ip_address)
            
        return True

    def unblock_ip(self, ip_address):
        """Manually unblock an IP address."""
        with self._lock:
            if ip_address in self._blocklist:
                del self._blocklist[ip_address]
                logger.info(f"✅ [IPS UNBLOCK] {ip_address} has been manually unblocked.")
                
                if self.mode == "os":
                    self._execute_os_unblock(ip_address)
                    
                return True
        return False

    def is_blocked(self, ip_address):
        """Check if an IP address is currently blocked."""
        with self._lock:
            if ip_address in self._blocklist:
                info = self._blocklist[ip_address]
                # Check expiration
                if int(time.time()) > info["expires_at"]:
                    del self._blocklist[ip_address]
                    logger.info(f"⏳ [IPS EXPIRED] Block expired for {ip_address}.")
                    if self.mode == "os":
                        self._execute_os_unblock(ip_address)
                    return False
                return True
        return False

    def get_blocklist(self):
        """Get the current active blocklist."""
        active_list = []
        current_time = int(time.time())
        
        with self._lock:
            # Clean up expired entries while iterating
            expired_ips = []
            
            for ip, info in self._blocklist.items():
                if current_time > info["expires_at"]:
                    expired_ips.append(ip)
                else:
                    active_list.append({
                        "ip": ip,
                        "reason": info["reason"],
                        "expires_at": info["expires_at"],
                        "timestamp": info.get("timestamp"),
                        "remaining_seconds": info["expires_at"] - current_time
                    })
                    
            for ip in expired_ips:
                del self._blocklist[ip]
                if self.mode == "os":
                    self._execute_os_unblock(ip)
                    
        return active_list

    def _execute_os_block(self, ip_address):
        """Execute OS-level firewall blocking (placeholder)."""
        logger.info(f"OS-level block execution requested for {ip_address} (Not implemented in demo).")
        # Example for Windows: os.system(f'netsh advfirewall firewall add rule name="IPS Block {ip}" dir=in action=block remoteip={ip}')
        # Example for Linux: os.system(f'iptables -A INPUT -s {ip} -j DROP')
        pass

    def _execute_os_unblock(self, ip_address):
        """Execute OS-level firewall unblocking (placeholder)."""
        logger.info(f"OS-level unblock execution requested for {ip_address} (Not implemented in demo).")
        pass
