"""
AI-Based IDS/IPS — Live Network Packet Sniffer
=================================================
Captures live network packets, builds heuristic flows, 
extracts top ML features, and pushes to the backend API.
"""

import os
import sys
import time
import requests
import joblib
import numpy as np
from collections import defaultdict

try:
    from scapy.all import sniff, IP, TCP, UDP
except ImportError:
    print("❌ Scapy not found! Run: pip install scapy")
    sys.exit(1)

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config

API_URL = f"http://localhost:{config.API_PORT}/detect"
PROCESSED_DIR = config.PROCESSED_DIR

print("Loading Feature Names and Machine Learning Scalers...")
try:
    feature_names = joblib.load(os.path.join(PROCESSED_DIR, "feature_names.joblib"))
    scaler = joblib.load(os.path.join(PROCESSED_DIR, "scaler.joblib"))
    encoders = joblib.load(os.path.join(PROCESSED_DIR, "encoders.joblib"))
except FileNotFoundError:
    print("❌ Trained models not found!")
    print("   Please run 'python train.py' and wait for it to complete.")
    sys.exit(1)

print(f"✅ Extracted Top {len(feature_names)} features expected by the Neural Networks.")

# Maintain active network flows
active_flows = defaultdict(lambda: {
    "start_time": time.time(),
    "last_seen": time.time(),
    "src_bytes": 0,
    "dst_bytes": 0,
    "count": 0,
})

def build_features(flow_key, flow_data):
    """
    Extract known metrics (bytes, count, duration) from live flow.
    Map categoricals using the saved LabelEncoders.
    Leave un-calculateable KDD features at 0.
    Returns exactly the ordered list the AI expects.
    """
    src_ip, dst_ip, src_port, dst_port, proto_num = flow_key
    raw_features = defaultdict(float)
    
    # 1. Populate basic calculable heuristics
    raw_features['duration'] = flow_data['last_seen'] - flow_data['start_time']
    raw_features['src_bytes'] = flow_data['src_bytes']
    raw_features['dst_bytes'] = flow_data['dst_bytes']
    raw_features['count'] = flow_data['count']
    
    # 2. Map & encode Categoricals 
    try:
        if 'protocol_type' in feature_names:
            proto_str = "tcp" if proto_num == 6 else "udp" if proto_num == 17 else "icmp"
            try:
                raw_features['protocol_type'] = float(encoders['protocol_type'].transform([proto_str])[0])
            except ValueError:
                pass
                
        if 'service' in feature_names:
            svc = "http" if 80 in (src_port, dst_port) else "https" if 443 in (src_port, dst_port) else "private"
            try:
                raw_features['service'] = float(encoders['service'].transform([svc])[0])
            except ValueError:
                pass
                
        if 'flag' in feature_names:
            try:
                raw_features['flag'] = float(encoders['flag'].transform(["SF"])[0])
            except ValueError:
                pass
    except Exception:
        pass

    # 3. Create perfectly ordered array
    ordered_features = []
    for f_name in feature_names:
        ordered_features.append(raw_features.get(f_name, 0.0))
        
    return ordered_features

def packet_callback(packet):
    """Process a single packet and stream ready AI payloads."""
    if IP in packet:
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        proto = packet[IP].proto
        
        src_port, dst_port = 0, 0
        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
            
        flow_key = (src_ip, dst_ip, src_port, dst_port, proto)
        flow = active_flows[flow_key]
        
        flow['count'] += 1
        flow['src_bytes'] += len(packet)
        flow['last_seen'] = time.time()
        
        # Burst every 10 packets or > 2 seconds to simulate "Live" monitoring
        if flow['count'] >= 10 or (time.time() - flow['start_time'] > 2):
            raw_array = build_features(flow_key, flow)
            
            try:
                # Important: reshape to 2D array for the scaler!
                scaled_array = scaler.transform([raw_array])[0]
                
                payload = {
                    "features": scaled_array.tolist(),
                    "src_ip": src_ip,
                    "dst_ip": dst_ip
                }
                
                # Fire-and-forget POST to local API
                requests.post(API_URL, json=payload, timeout=0.1)
                
            except requests.exceptions.RequestException:
                pass # Fail silently if API is offline
            except Exception as e:
                pass
                
            # Reset flow counts
            del active_flows[flow_key]

print("\n=======================================================")
print("🌐 STARTING LIVE NETWORK SNIFFER")
print("=======================================================")
print("Actively tracking packets, scaling features, and")
print("streaming vectors directly to the AI Dashboard.")
print("Press Ctrl+C to stop.")
print("=======================================================\n")

try:
    sniff(prn=packet_callback, store=0)
except PermissionError:
    print("❌ Permission Denied! Run VS Code / Terminal as Administrator to sniff raw socket packets!")
except OSError as e:
    if "pcap" in str(e).lower() or "winpcap" in str(e).lower():
         print("❌ Npcap/WinPcap missing!")
         print("To sniff packets on Windows, please install Npcap from: https://npcap.com/")
    else:
         print(f"OS Error: {e}")
