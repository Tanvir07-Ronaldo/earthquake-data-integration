"""
=============================================================================
  Live Seismic PGA Monitor + Waveform Dashboard  —  Fast Version
  ─────────────────────────────────────────────────────────────────
  - PGA printed to console every 1 second (same as original script)
  - Live waveform updates smoothly every 1 second
  - All 3 channels fetched in parallel (no sequential delay)
  - Dashboard and PGA show exactly the same number
  - Close the window → everything stops

  Usage       : python pga_live_dashboard.py
  Dependencies: pip install obspy matplotlib numpy
=============================================================================
"""

from obspy.clients.earthworm import Client
from obspy import UTCDateTime
import numpy as np
import matplotlib
matplotlib.use("TkAgg")        # try "Qt5Agg" if TkAgg gives an error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import threading
import time
import sys

# ── Configuration ─────────────────────────────────────────────────────────────
HOST        = "127.0.0.1"
PORT        = 16022
NET         = "SM"
STA         = "HSG"
CHANNELS    = ["HGZ", "HGN", "HGE"]
LOC         = "--"
sensitivity = 2.4e-7        # g per count
WINDOW_SEC  = 30            # seconds shown on plot  (30 = snappier than 60)
FETCH_SEC   = 3             # fetch this many seconds per request (overlap helps)
RETRY_SEC   = 5             # seconds between connection retries
# ─────────────────────────────────────────────────────────────────────────────

# ── Colours ───────────────────────────────────────────────────────────────────
BG     = "#0d1117"
GRID   = "#1e2a3a"
FG     = "#c9d1d9"
COLORS = ["#58a6ff", "#3fb950", "#f78166"]
PBG    = "#161b22"
S_OK   = "#3fb950"
S_WARN = "#f0883e"
# ─────────────────────────────────────────────────────────────────────────────

# ── Shared state ──────────────────────────────────────────────────────────────
lock   = threading.Lock()
shared = {
    "offset"    : 300,
    "pga_now"   : 0.0,
    "pga_avg"   : 0.0,
    "best_ch"   : "---",
    "ch_pga"    : {ch: 0.0 for ch in CHANNELS},
    "status"    : "Starting...",
    "s_color"   : S_WARN,
    "running"   : True,
}

# Ring buffers — one per channel
ring_t = {ch: np.array([]) for ch in CHANNELS}
ring_c = {ch: np.array([]) for ch in CHANNELS}

pga_history = []


# ── Helpers ───────────────────────────────────────────────────────────────────
def push_ring(ch, new_t, new_c):
    """Append new samples, keep only WINDOW_SEC worth."""
    with lock:
        if ring_t[ch].size == 0:
            ring_t[ch] = new_t;  ring_c[ch] = new_c
        else:
            last = ring_t[ch][-1]
            mask = new_t > last
            if mask.any():
                ring_t[ch] = np.concatenate([ring_t[ch], new_t[mask]])
                ring_c[ch] = np.concatenate([ring_c[ch], new_c[mask]])
        if ring_t[ch].size > 0:
            cut = ring_t[ch][-1] - WINDOW_SEC
            m   = ring_t[ch] >= cut
            ring_t[ch] = ring_t[ch][m]
            ring_c[ch] = ring_c[ch][m]


def fetch_one(client, ch, t_start, t_end):
    """Fetch one channel. Returns (times, counts) or (None, None)."""
    try:
        st = client.get_waveforms(NET, STA, LOC, ch, t_start, t_end)
        if len(st) == 0 or st[0].stats.npts == 0:
            return None, None
        tr  = st[0]
        sr  = tr.stats.sampling_rate
        n   = tr.stats.npts
        ts  = float(tr.stats.starttime)
        t   = np.linspace(ts, ts + n / sr, n)
        c   = tr.data.astype(np.float64)
        return t, c
    except Exception as e:
        if "FR" in str(e):
            with lock:
                shared["offset"] += 20
        return None, None


# ── Step 1: Connect ───────────────────────────────────────────────────────────
def connect():
    print("=" * 60)
    print("  Live Seismic PGA Monitor + Waveform Dashboard")
    print(f"  Host: {HOST}:{PORT}   Station: {NET}.{STA}")
    print(f"  Channels: {', '.join(CHANNELS)}   Sensitivity: {sensitivity}")
    print("=" * 60)
    print("\n  [->] Connecting...")
    while shared["running"]:
        try:
            c = Client(HOST, PORT)
            print(f"  [OK] Connected to {HOST}:{PORT}")
            return c
        except Exception as e:
            print(f"  [!]  {e}  — retry in {RETRY_SEC}s...")
            time.sleep(RETRY_SEC)
    return None


# ── Step 2: Find safe offset ──────────────────────────────────────────────────
def find_safe_offset(client):
    print("\n  [->] Checking tank availability...")

    for attempt in range(1, 120):
        if not shared["running"]:
            return 300
        try:
            avail = client.get_availability()
            if avail:
                print(f"  [OK] Tank has data:")
                for a in avail:
                    print(f"       {a[0]}.{a[1]}.{a[3]}  —  "
                          f"{round(float(a[5])-float(a[4]),1)}s")
                break
            print(f"  [~]  Attempt {attempt}: empty — wait {RETRY_SEC}s...")
            time.sleep(RETRY_SEC)
        except Exception as e:
            print(f"  [~]  Attempt {attempt}: {e} — wait {RETRY_SEC}s...")
            time.sleep(RETRY_SEC)

    print("\n  [->] Finding working offset...")
    probe = 300
    for _ in range(80):
        if not shared["running"]:
            return 300
        t, c = fetch_one(client, CHANNELS[0],
                         UTCDateTime() - probe,
                         UTCDateTime() - probe + 1)
        if t is not None:
            print(f"  [OK] Data confirmed at offset={probe}s")
            print(f"  [OK] WaveServerV is running correctly\n")
            return probe
        probe += 30
        time.sleep(0.5)

    return 300


# ══════════════════════════════════════════════════════════════════════════════
#  PGA THREAD — fetches all channels IN PARALLEL every 1 second
# ══════════════════════════════════════════════════════════════════════════════
def pga_thread_func(client):
    print("=== Checking tank range ===")
    try:
        avail = client.get_availability()
        for a in avail:
            print(a)
    except Exception as e:
        print("get_availability failed: " + str(e))
    print("")
    print("=== Starting PGA loop ===")

    results = {}    # ch -> (times, counts)

    def fetch_worker(ch, t_start, t_end):
        results[ch] = fetch_one(client, ch, t_start, t_end)

    while shared["running"]:
        cycle_start = time.time()
        offset      = shared["offset"]
        t_end       = UTCDateTime() - offset
        t_start     = t_end - FETCH_SEC

        # ── Fetch all channels IN PARALLEL ────────────────────────────────────
        results.clear()
        threads = []
        for ch in CHANNELS:
            th = threading.Thread(target=fetch_worker,
                                  args=(ch, t_start, t_end), daemon=True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join(timeout=3)   # max 3s wait per fetch

        # ── Process results ───────────────────────────────────────────────────
        pga_max    = 0.0
        counts_max = 0
        best_ch    = "---"
        got_data   = False

        for ch in CHANNELS:
            t_arr, c_arr = results.get(ch, (None, None))
            if t_arr is None:
                continue
            got_data = True

            # Push to ring buffer for plot
            push_ring(ch, t_arr, c_arr)

            # PGA for this channel — last 1 second (100 samples)
            sr      = 100
            slice1s = c_arr[-sr:] if len(c_arr) >= sr else c_arr
            peak    = int(np.max(np.abs(slice1s)))
            pga_g   = peak * sensitivity

            with lock:
                shared["ch_pga"][ch] = round(pga_g, 6)

            if pga_g > pga_max:
                pga_max    = pga_g
                counts_max = peak
                best_ch    = ch

        # ── No data ───────────────────────────────────────────────────────────
        if not got_data:
            print(str(UTCDateTime()) +
                  " No data at offset=" + str(offset) + "s, trying further back...")
            with lock:
                shared["offset"]  += 30
                shared["status"]   = f"Waiting... offset={shared['offset']}s"
                shared["s_color"]  = S_WARN
            time.sleep(1)
            continue

        # ── Console output ────────────────────────────────────────────────────
        print(
            str(UTCDateTime()) +
            " PGA: " + str(round(pga_max, 8)) +
            " g  (raw counts: " + str(counts_max) + ")" +
            "  [best_ch=" + best_ch + "]" +
            "  [offset=" + str(offset) + "s]"
        )

        # ── Update shared state ───────────────────────────────────────────────
        pga_history.append(pga_max)
        if len(pga_history) > 10:
            pga_history.pop(0)

        with lock:
            shared["pga_now"]  = round(pga_max, 6)
            shared["best_ch"]  = best_ch
            shared["pga_avg"]  = round(np.mean(pga_history), 6)
            shared["status"]   = (
                f"WaveServerV OK  |  Tank LIVE  |  "
                f"{UTCDateTime().strftime('%H:%M:%S')} UTC  |  "
                f"offset={offset}s"
            )
            shared["s_color"]  = S_OK

            # Reduce offset toward real-time
            if offset > 60:
                shared["offset"] -= 1

        # ── Maintain exactly 1-second cycle ──────────────────────────────────
        elapsed = time.time() - cycle_start
        sleep_t = max(0, 1.0 - elapsed)
        time.sleep(sleep_t)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def build_figure():
    n   = len(CHANNELS)
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.canvas.manager.set_window_title(
        f"PGA + Live Waveform  |  {NET}.{STA}  |  {HOST}:{PORT}"
    )

    gs = gridspec.GridSpec(
        n + 1, 1,
        height_ratios=[0.14] + [1.0] * n,
        hspace=0.04,
        left=0.07, right=0.97, top=0.93, bottom=0.07
    )

    # Status bar
    ax_s = fig.add_subplot(gs[0])
    ax_s.set_facecolor("#0a1628")
    ax_s.set_xticks([]); ax_s.set_yticks([])
    for sp in ax_s.spines.values():
        sp.set_edgecolor(GRID)

    st_lbl = ax_s.text(
        0.01, 0.5, "●  Starting...",
        transform=ax_s.transAxes,
        color=S_WARN, fontsize=10, fontweight="bold",
        va="center", fontfamily="monospace"
    )
    pga_lbl = ax_s.text(
        0.99, 0.5, "",
        transform=ax_s.transAxes,
        color="#f0883e", fontsize=11, fontweight="bold",
        va="center", ha="right", fontfamily="monospace"
    )

    axes = []; lines = []; off_lbls = []; ch_lbls = []

    for i, ch in enumerate(CHANNELS):
        ax = fig.add_subplot(gs[i + 1])
        ax.set_facecolor(PBG)
        ax.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
        ax.tick_params(colors=FG, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

        ln, = ax.plot([], [], color=COLORS[i], linewidth=0.8, antialiased=True)

        # Channel name
        ax.text(
            0.005, 0.95, f"{NET}.{STA}.{ch}",
            transform=ax.transAxes,
            color=COLORS[i], fontsize=10, fontweight="bold", va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BG,
                      edgecolor=COLORS[i], alpha=0.85)
        )

        # Per-channel PGA
        cl = ax.text(
            0.005, 0.06, "",
            transform=ax.transAxes,
            color=COLORS[i], fontsize=9, fontweight="bold", va="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BG,
                      edgecolor=COLORS[i], alpha=0.75)
        )

        # Offset
        ol = ax.text(
            0.995, 0.95, "",
            transform=ax.transAxes,
            color=FG, fontsize=8, va="top", ha="right",
            fontfamily="monospace", alpha=0.6
        )

        ax.set_ylabel("counts", color=FG, fontsize=8)
        if i < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("seconds ago  (0 = most recent)",
                          color=FG, fontsize=9)

        axes.append(ax); lines.append(ln)
        off_lbls.append(ol); ch_lbls.append(cl)

    fig.suptitle(
        f"DS EQMS  —  Live PGA + Waveform Monitor  |  "
        f"{NET}.{STA}  |  {HOST}:{PORT}",
        color=FG, fontsize=12, fontweight="bold", y=0.97
    )

    return fig, axes, lines, st_lbl, pga_lbl, off_lbls, ch_lbls


def make_updater(axes, lines, st_lbl, pga_lbl, off_lbls, ch_lbls):

    def update(frame):
        if not shared["running"]:
            return lines + [st_lbl, pga_lbl] + off_lbls + ch_lbls

        with lock:
            offset   = shared["offset"]
            s_text   = shared["status"]
            s_color  = shared["s_color"]
            pga_now  = shared["pga_now"]
            pga_avg  = shared["pga_avg"]
            best     = shared["best_ch"]
            ch_pgas  = dict(shared["ch_pga"])
            # Snapshot ring buffers safely
            snap_t = {ch: ring_t[ch].copy() for ch in CHANNELS}
            snap_c = {ch: ring_c[ch].copy() for ch in CHANNELS}

        for i, ch in enumerate(CHANNELS):
            t_arr = snap_t[ch]
            c_arr = snap_c[ch]

            if t_arr.size < 2:
                off_lbls[i].set_text(f"offset={offset}s  buffering...")
                continue

            now   = t_arr[-1]
            x_rel = t_arr - now          # 0 = most recent

            lines[i].set_data(x_rel, c_arr)
            axes[i].set_xlim(-WINDOW_SEC, 0)

            ymin = c_arr.min(); ymax = c_arr.max()
            pad  = max((ymax - ymin) * 0.12, 500)
            axes[i].set_ylim(ymin - pad, ymax + pad)

            ch_lbls[i].set_text(f"PGA = {ch_pgas.get(ch, 0.0)} g")
            off_lbls[i].set_text(f"offset={offset}s")

        st_lbl.set_text("●  " + s_text)
        st_lbl.set_color(s_color)

        if pga_now > 0:
            pga_lbl.set_text(
                f"PGA(1s) {pga_now} g  |  avg10s {pga_avg} g  |  {best}"
            )

        return lines + [st_lbl, pga_lbl] + off_lbls + ch_lbls

    return update


def on_close(event):
    print("\n\n  [OK] Window closed — stopping everything...")
    shared["running"] = False
    time.sleep(0.3)
    sys.exit(0)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    client = connect()
    if client is None:
        sys.exit(1)

    shared["offset"] = find_safe_offset(client)

    # Start PGA thread
    th = threading.Thread(target=pga_thread_func, args=(client,), daemon=True)
    th.start()

    # Wait 2 seconds so ring buffers have some data before opening window
    print("  [->] Pre-filling buffers (2s)...")
    time.sleep(2)

    print("  [->] Opening dashboard...\n")
    fig, axes, lines, st_lbl, pga_lbl, off_lbls, ch_lbls = build_figure()
    fig.canvas.mpl_connect("close_event", on_close)

    updater = make_updater(axes, lines, st_lbl, pga_lbl, off_lbls, ch_lbls)

    # Animate at 1000ms = 1 second — in sync with PGA thread
    ani = FuncAnimation(
        fig, updater,
        interval=1000,
        blit=True,
        cache_frame_data=False
    )

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  [OK] Stopped.")
        shared["running"] = False
        sys.exit(0)
