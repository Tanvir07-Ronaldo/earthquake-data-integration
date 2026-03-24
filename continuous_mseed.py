"""
=============================================================================
  Continuous MSEED Recorder  —  WaveServer / Earthworm
  ─────────────────────────────────────────────────────
  Connects to a WaveServer, fetches the last 5 seconds every 5 seconds,
  and saves each trace as a separate .mseed file.

  Files are written to:  ./mseed_data/
  Filename format  :  SM.HSG.HGZ.20260312_040153.mseed
                       (network.station.channel.YYYYMMDD_HHMMSS.mseed)

  Usage:
    python record_continuous_mseed.py

  Dependencies:
    pip install obspy
=============================================================================
"""

from obspy.clients.earthworm import Client
from obspy import UTCDateTime
import time
import os

# ── Configuration ─────────────────────────────────────────────────────────────
HOST     = "192.168.0.110"
PORT     = 16022
NETWORK  = "SM"
STATION  = "HSG"
LOCATION = ""
CHANNELS = ["HGZ", "HGN", "HGE"]
INTERVAL = 5          # seconds between each fetch

# Output folder is ALWAYS placed next to this script file — never depends
# on which folder you run the script from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR     = os.path.join(SCRIPT_DIR, "mseed_data")
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTDIR, exist_ok=True)

print("=" * 60)
print("  Continuous MSEED Recorder")
print(f"  Host    : {HOST}:{PORT}")
print(f"  Station : {NETWORK}.{STATION}  Channels: {', '.join(CHANNELS)}")
print(f"  Output  : {os.path.abspath(OUTDIR)}/")
print(f"  Interval: every {INTERVAL} s")
print("=" * 60)
print("  Press Ctrl+C to stop.\n")

try:
    client = Client(HOST, PORT)
    print(f"  [✓] Connected to {HOST}:{PORT}\n")
except Exception as e:
    print(f"  [!] Could not connect to {HOST}:{PORT}: {e}\n")
    raise

saved_count  = 0
error_count  = 0

while True:
    t2 = UTCDateTime()
    t1 = t2 - INTERVAL

    for ch in CHANNELS:
        try:
            st = client.get_waveforms(NETWORK, STATION, LOCATION, ch, t1, t2)

            if len(st) == 0:
                print(f"  [~] {ch}  — empty stream, skipping")
                continue

            # Filename uses full timestamp (seconds) so every file is unique
            ts_str   = t1.strftime("%Y%m%d_%H%M%S")
            filename = f"{NETWORK}.{STATION}.{ch}.{ts_str}.mseed"
            filepath = os.path.join(OUTDIR, filename)

            st.write(filepath, format="MSEED")
            saved_count += 1

            n_samp = sum(tr.stats.npts for tr in st)
            print(f"  [✓] Saved: {filename}   ({n_samp} samples)")

        except Exception as e:
            error_count += 1
            print(f"  [!] Error on {ch}: {e}")

    time.sleep(INTERVAL)
