

import os
import sys
import glob
import struct
import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy import signal
from scipy.fft import fft, fftfreq


# =============================================================================
#  PART 1-A  —  Parse MSEED Fixed Section of Data Header + Blockette chain
# =============================================================================

def parse_header(raw: bytes) -> dict:
    """Decode 48-byte FSDH and walk blockette chain. Returns metadata dict."""
    hdr  = raw[:48]
    meta = {
        "station"     : raw[8:13].decode("ascii").strip(),
        "location"    : raw[13:15].decode("ascii").strip(),
        "channel"     : raw[15:18].decode("ascii").strip(),
        "network"     : raw[18:20].decode("ascii").strip(),
        "year"        : struct.unpack(">H", hdr[20:22])[0],
        "day"         : struct.unpack(">H", hdr[22:24])[0],
        "hour"        : hdr[24],
        "minute"      : hdr[25],
        "second"      : hdr[26],
        "microsecond" : struct.unpack(">H", hdr[28:30])[0],
        "num_samples" : struct.unpack(">H", hdr[30:32])[0],
        "sr_factor"   : struct.unpack(">h", hdr[32:34])[0],
        "sr_mult"     : struct.unpack(">h", hdr[34:36])[0],
        "data_offset" : struct.unpack(">H", hdr[44:46])[0],
        "encoding"    : 11,
    }

    # Compute sample rate from factor / multiplier fields
    f, m = meta["sr_factor"], meta["sr_mult"]
    if   f > 0 and m > 0: meta["sample_rate"] = float(f * m)
    elif f > 0 and m < 0: meta["sample_rate"] = -float(f) / m
    elif f < 0 and m > 0: meta["sample_rate"] = -float(m) / f
    else:                  meta["sample_rate"] = 1.0 / (f * m)

    # Walk blockette chain — detect Blockette 1000 for encoding type
    blk = struct.unpack(">H", hdr[46:48])[0]
    for _ in range(10):
        if blk == 0 or blk + 4 > len(raw):
            break
        btype    = struct.unpack(">H", raw[blk:blk+2])[0]
        next_blk = struct.unpack(">H", raw[blk+2:blk+4])[0]
        if btype == 1000:
            meta["encoding"] = raw[blk + 4]
        blk = next_blk

    # Build Python datetime for record start
    meta["start_dt"] = (
        datetime.datetime(meta["year"], 1, 1)
        + datetime.timedelta(
            days         = meta["day"] - 1,
            hours        = meta["hour"],
            minutes      = meta["minute"],
            seconds      = meta["second"],
            microseconds = meta["microsecond"] * 100,
        )
    )
    return meta


# =============================================================================
#  PART 1-B  —  Steim-2 Decoder  (SEED Reference Manual §10)
# =============================================================================

def decode_steim2(data: bytes, num_samples: int) -> np.ndarray:
    """Decompress Steim-2 encoded data block. Returns int64 sample array."""
    diffs = []
    x0    = None

    for fi in range(len(data) // 64):
        frame = data[fi * 64: fi * 64 + 64]
        if len(frame) < 64:
            break
        ctrl = struct.unpack(">I", frame[0:4])[0]

        for wi in range(16):
            wb   = frame[wi * 4: wi * 4 + 4]
            w    = struct.unpack(">I", wb)[0]
            ws   = struct.unpack(">i", wb)[0]
            code = (ctrl >> (30 - wi * 2)) & 0x3

            # Frame 0 special words
            if fi == 0 and wi == 0: continue            # control word
            if fi == 0 and wi == 1: x0 = ws; continue   # X0  — first sample
            if fi == 0 and wi == 2: continue             # Xn  — last sample

            if code == 0:
                continue                                 # unused word

            elif code == 1:                              # 4 × 8-bit differences
                for i in range(3, -1, -1):
                    b = (w >> (i * 8)) & 0xFF
                    diffs.append(b - 256 if b & 0x80 else b)

            elif code == 2:
                dnib = (w >> 30) & 0x3
                if dnib == 1:                            # 1 × 30-bit
                    v = w & 0x3FFFFFFF
                    diffs.append(v - 0x40000000 if v & 0x20000000 else v)
                elif dnib == 2:                          # 2 × 15-bit
                    for i in range(1, -1, -1):
                        v = (w >> (i * 15)) & 0x7FFF
                        diffs.append(v - 0x8000 if v & 0x4000 else v)
                elif dnib == 3:                          # 3 × 10-bit
                    for i in range(2, -1, -1):
                        v = (w >> (i * 10)) & 0x3FF
                        diffs.append(v - 0x400 if v & 0x200 else v)

            elif code == 3:
                dnib = (w >> 30) & 0x3
                if dnib == 0:                            # 5 × 6-bit
                    for i in range(4, -1, -1):
                        v = (w >> (i * 6)) & 0x3F
                        diffs.append(v - 0x40 if v & 0x20 else v)
                elif dnib == 1:                          # 6 × 5-bit
                    for i in range(5, -1, -1):
                        v = (w >> (i * 5)) & 0x1F
                        diffs.append(v - 0x20 if v & 0x10 else v)
                elif dnib == 2:                          # 7 × 4-bit
                    for i in range(6, -1, -1):
                        v = (w >> (i * 4)) & 0x0F
                        diffs.append(v - 0x10 if v & 0x08 else v)

    if x0 is None:
        return np.array(diffs, dtype=np.int64)[:num_samples]

    arr    = np.array([x0] + diffs, dtype=np.int64)
    result = np.cumsum(arr)
    return result[:num_samples]


# =============================================================================
#  PART 1-C  —  Extract Specific Time-Window Sections
# =============================================================================

def extract_sections(samples: np.ndarray, sr: float, windows: list) -> dict:
    """
    Cut sample array into named time windows.
    windows : list of (t_start_s, t_end_s) tuples
    Returns : dict  { 'T0–T1s' : np.ndarray, … }
    """
    duration  = len(samples) / sr
    extracted = {}
    for (t0, t1) in windows:
        t0  = max(0.0, t0)
        t1  = min(duration, t1)
        i0  = int(round(t0 * sr))
        i1  = int(round(t1 * sr))
        lbl = f"{t0:.2f}-{t1:.2f}s"
        seg = samples[i0:i1]
        extracted[lbl] = seg
        print(f"    [{lbl}]  {len(seg)} samples  "
              f"min={seg.min():.0f}  max={seg.max():.0f}")
    return extracted


# =============================================================================
#  PART 2  —  Waveform Processing
# =============================================================================

def process_waveform(samples: np.ndarray, sr: float,
                     f_low: float = 1.0, f_high: float = 45.0) -> dict:
    """Detrend + zero-phase 4-pole Butterworth bandpass filter."""
    detrended = samples - np.mean(samples)
    nyq       = sr / 2.0
    b, a      = signal.butter(4, [f_low / nyq, f_high / nyq], btype="band")
    filtered  = signal.filtfilt(b, a, detrended)
    time_axis = np.arange(len(samples)) / sr
    return {
        "raw"      : samples,
        "detrended": detrended,
        "filtered" : filtered,
        "time"     : time_axis,
        "peak_idx" : int(np.argmax(np.abs(filtered))),
    }


# =============================================================================
#  PART 3  —  Frequency Analysis
# =============================================================================

def frequency_analysis(detrended: np.ndarray, sr: float,
                       f_high: float = 45.0) -> dict:
    """FFT spectrum, Welch PSD, STFT spectrogram."""
    N = len(detrended)

    # FFT
    freqs = fftfreq(N, d=1.0 / sr)[: N // 2]
    fft_a = np.abs(fft(detrended))[: N // 2] * 2.0 / N

    # Welch PSD
    f_w, psd = signal.welch(
        detrended, fs=sr,
        nperseg=min(512, N // 4),
        window="hann", scaling="density",
    )

    # STFT Spectrogram
    nps       = min(256, N // 8)
    novlp     = min(int(nps * 0.75), N // 10)
    f_s, t_s, Sxx = signal.spectrogram(
        detrended, fs=sr,
        nperseg=nps, noverlap=novlp, window="hann",
    )

    # Dominant frequency from Welch (ignore DC / sub-0.5 Hz)
    mask   = (f_w > 0.5) & (f_w <= f_high)
    peak_f = float(f_w[mask][np.argmax(psd[mask])])

    return {
        "freqs"   : freqs,   "fft_a" : fft_a,
        "f_w"     : f_w,     "psd"   : psd,
        "f_s"     : f_s,     "t_s"   : t_s,   "Sxx"    : Sxx,
        "peak_f"  : peak_f,  "peak_T": 1.0 / peak_f,
        "rms"     : float(np.sqrt(np.mean(detrended ** 2))),
        "peak_amp": float(np.max(np.abs(detrended))),
    }


# =============================================================================
#  PLOT  —  7-Panel Figure
# =============================================================================

def plot_results(wf: dict, fa: dict, meta: dict,
                 sections: dict, fname: str):

    # ── Colour palette ────────────────────────────────────────────────────────
    BG     = "#07090f"
    PANEL  = "#0d1117"
    BORDER = "#1c2333"
    CYAN   = "#00d4ff"
    GREEN  = "#00ff88"
    ORANGE = "#ff6b35"
    PURPLE = "#a855f7"
    RED    = "#ff4c4c"
    MUTED  = "#8b949e"
    TEXT   = "#e6edf3"
    CSEC   = [GREEN, ORANGE, PURPLE]

    sr   = meta["sample_rate"]
    ns   = meta["num_samples"]
    sdt  = meta["start_dt"]
    edt  = sdt + datetime.timedelta(seconds=ns / sr)
    pidx = wf["peak_idx"]
    t    = wf["time"]

    def style(ax, ylabel="", xlabel=""):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.tick_params(colors=TEXT, labelsize=9, which="both", pad=3)
        ax.grid(True, color=BORDER, lw=0.5, ls="--", alpha=0.7)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=9, labelpad=5)
        if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=9, labelpad=5)

    def ptitle(ax, txt, color):
        ax.set_title(txt, color=color, fontsize=10, fontweight="bold",
                     pad=4, loc="left", fontfamily="monospace")

    # ── Figure  22 × 34 in ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 34), facecolor=BG)

    # Layout (top → bottom, no zones touch each other):
    #   gs_hdr : 0.990 – 0.955   header (title + subtitle)
    #   gs_top : 0.945 – 0.575   3 waveform panels
    #   gs_mid : 0.555 – 0.200   FFT / PSD / Spectrogram
    #   gs_sum : 0.178 – 0.022   Analysis Summary

    gs_hdr = gridspec.GridSpec(
        1, 1,
        top=0.998, bottom=0.960,
        left=0.075, right=0.975,
    )
    gs_top = gridspec.GridSpec(
        3, 1,
        top=0.950, bottom=0.575,
        left=0.075, right=0.975,
        hspace=0.85,
    )
    gs_mid = gridspec.GridSpec(
        2, 2,
        top=0.530, bottom=0.195,
        left=0.075, right=0.975,
        hspace=1.10, wspace=0.28,
    )
    gs_sum = gridspec.GridSpec(
        1, 1,
        top=0.178, bottom=0.022,
        left=0.075, right=0.975,
    )

    # ── Header axes — title + subtitle, fully isolated from plots ─────────────
    ax_hdr = fig.add_subplot(gs_hdr[0])
    ax_hdr.set_facecolor(BG)
    ax_hdr.axis("off")
    # Title — smaller font so it fits on one line at any screen width
    ax_hdr.text(0.5, 1.0,
                f"SEISMIC DATA ANALYSIS  —  "
                f"{meta['network']} · {meta['station']} · {meta['channel']}",
                transform=ax_hdr.transAxes,
                ha="center", va="top",
                fontsize=15, fontweight="bold",
                color=TEXT, fontfamily="monospace")
    # Subtitle — on second line, smaller
    ax_hdr.text(0.5, 0.42,
                f"File: {fname}     "
                f"Start: {sdt.strftime('%Y-%m-%d  %H:%M:%S UTC')}     "
                f"Duration: {ns/sr:.2f} s     "
                f"Fs: {sr:.0f} Hz     Encoding: Steim-2",
                transform=ax_hdr.transAxes,
                ha="center", va="top",
                fontsize=8, color=MUTED, fontfamily="monospace")

    # =========================================================================
    #  P1  —  Raw Waveform
    # =========================================================================
    ax1 = fig.add_subplot(gs_top[0])
    ax1.plot(t, wf["raw"], color=CYAN, lw=0.7, alpha=0.9, label="Raw counts")
    ax1.axhline(np.mean(wf["raw"]), color=RED, lw=1.0, ls="--",
                alpha=0.8, label=f"Mean = {np.mean(wf['raw']):.0f}")
    for i, (lbl, _) in enumerate(sections.items()):
        t0, t1 = [float(x) for x in lbl.replace("s", "").split("-")]
        ax1.axvspan(t0, t1, alpha=0.15, color=CSEC[i % 3],
                    label=f"Section  {lbl}")
    ax1.set_xlim(0, ns / sr)
    ptitle(ax1, "Raw Waveform  —  Extracted Sections Highlighted", CYAN)
    style(ax1, ylabel="Amplitude (counts)", xlabel="Time (s)")
    ax1.legend(loc="upper right", fontsize=8, facecolor=PANEL,
               labelcolor=TEXT, edgecolor=BORDER, framealpha=0.95,
               handlelength=1.2, borderpad=0.6, labelspacing=0.35)

    # =========================================================================
    #  P2  —  Extracted Sections Mean-Centred
    # =========================================================================
    ax2 = fig.add_subplot(gs_top[1])
    for i, (lbl, seg) in enumerate(sections.items()):
        ax2.plot(np.arange(len(seg)) / sr, seg - np.mean(seg),
                 color=CSEC[i % 3], lw=0.85, alpha=0.9,
                 label=f"Section  {lbl}")
    ptitle(ax2, "Extracted Sections — Mean-Centred Comparison", GREEN)
    style(ax2, ylabel="Amplitude - mean (counts)", xlabel="Relative Time (s)")
    ax2.legend(loc="upper right", fontsize=8, facecolor=PANEL,
               labelcolor=TEXT, edgecolor=BORDER, framealpha=0.95,
               handlelength=1.2, borderpad=0.6, labelspacing=0.35)

    # =========================================================================
    #  P3  —  Bandpass-Filtered Waveform
    # =========================================================================
    ax3 = fig.add_subplot(gs_top[2])
    ax3.plot(t, wf["detrended"], color=MUTED, lw=0.4,
             alpha=0.4, label="Detrended")
    ax3.plot(t, wf["filtered"], color=GREEN, lw=1.0, alpha=0.95,
             label="Bandpass  1-45 Hz  (4-pole Butterworth, zero-phase)")
    ax3.axvline(t[pidx], color=RED, lw=1.2, ls=":",
                label=f"Peak @ {t[pidx]:.3f} s  ({wf['filtered'][pidx]:.0f} counts)")
    ax3.set_xlim(0, ns / sr)
    ptitle(ax3, "Bandpass-Filtered Waveform  (1-45 Hz, zero-phase)", GREEN)
    style(ax3, ylabel="Amplitude (counts)", xlabel="Time (s)")
    ax3.legend(loc="upper right", fontsize=8, facecolor=PANEL,
               labelcolor=TEXT, edgecolor=BORDER, framealpha=0.95,
               handlelength=1.2, borderpad=0.6, labelspacing=0.35)

    # =========================================================================
    #  P4  —  FFT Amplitude Spectrum
    # =========================================================================
    ax4 = fig.add_subplot(gs_mid[0, 0])
    mf  = fa["freqs"] <= 50
    ax4.semilogy(fa["freqs"][mf], fa["fft_a"][mf] + 1e-9,
                 color=ORANGE, lw=0.9)
    ax4.fill_between(fa["freqs"][mf], fa["fft_a"][mf] + 1e-9,
                     alpha=0.13, color=ORANGE)
    ax4.axvline(fa["peak_f"], color=RED, lw=1.3, ls="--",
                label=f"Peak  {fa['peak_f']:.3f} Hz")
    ax4.set_xlim(0, 50)
    ptitle(ax4, "FFT Amplitude Spectrum", ORANGE)
    style(ax4, ylabel="Amplitude (counts)", xlabel="Frequency (Hz)")
    ax4.tick_params(axis="y", labelcolor=MUTED)
    ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT,
               edgecolor=BORDER, framealpha=0.95, borderpad=0.6)

    # =========================================================================
    #  P5  —  Welch PSD
    # =========================================================================
    ax5 = fig.add_subplot(gs_mid[0, 1])
    mw  = fa["f_w"] <= 50
    ax5.semilogy(fa["f_w"][mw], fa["psd"][mw], color=PURPLE, lw=0.9)
    ax5.fill_between(fa["f_w"][mw], fa["psd"][mw],
                     alpha=0.13, color=PURPLE)
    ax5.axvline(fa["peak_f"], color=RED, lw=1.3, ls="--",
                label=f"Dominant  {fa['peak_f']:.3f} Hz\n"
                      f"Period      {fa['peak_T']:.4f} s")
    ax5.set_xlim(0, 50)
    ptitle(ax5, "Power Spectral Density  (Welch)", PURPLE)
    style(ax5, ylabel="PSD (counts^2 / Hz)", xlabel="Frequency (Hz)")
    ax5.tick_params(axis="y", labelcolor=MUTED)
    ax5.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT,
               edgecolor=BORDER, framealpha=0.95, borderpad=0.6)

    # =========================================================================
    #  P6  —  Spectrogram
    # =========================================================================
    ax6 = fig.add_subplot(gs_mid[1, 0])
    fm   = fa["f_s"] <= 45
    Sdb  = 10 * np.log10(fa["Sxx"][fm, :] + 1e-30)
    img  = ax6.pcolormesh(fa["t_s"], fa["f_s"][fm], Sdb,
                          shading="gouraud", cmap="inferno")
    ax6.set_ylim(0, 45)
    cb = plt.colorbar(img, ax=ax6, pad=0.02, aspect=20)
    cb.set_label("Power (dB)", color=MUTED, fontsize=8, labelpad=5)
    cb.ax.tick_params(labelcolor=MUTED, labelsize=8)
    ptitle(ax6, "Spectrogram  (STFT, Hann window)", PURPLE)
    style(ax6, ylabel="Frequency (Hz)", xlabel="Time (s)")

    # gs_mid[1,1] stays blank — summary has full width below
    ax6b = fig.add_subplot(gs_mid[1, 1])
    ax6b.set_facecolor(BG)
    ax6b.axis("off")

    # =========================================================================
    #  P7  —  Analysis Summary  (full-width, 3-column, pixel-safe text layout)
    # =========================================================================
    ax7 = fig.add_subplot(gs_sum[0])
    ax7.set_facecolor(PANEL)
    for sp in ax7.spines.values():
        sp.set_edgecolor(CYAN)
        sp.set_linewidth(0.8)
        sp.set_alpha(0.5)
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.axis("off")

    # ── Summary title ────────────────────────────────────────────────────────
    ax7.text(0.5, 0.98, "ANALYSIS  SUMMARY",
             transform=ax7.transAxes, ha="center", va="top",
             fontsize=11, fontweight="bold", color=CYAN,
             fontfamily="monospace")
    # Divider sits BELOW the title text (title is ~0.10 tall at fontsize 11)
    ax7.plot([0.01, 0.99], [0.86, 0.86],
             transform=ax7.transAxes,
             color=CYAN, lw=0.6, alpha=0.4, clip_on=False)

    # ── Column definitions ───────────────────────────────────────────────────
    col1 = [
        ("RECORD",      None),
        ("Network",     meta["network"]),
        ("Station",     meta["station"]),
        ("Channel",     meta["channel"]),
        ("Encoding",    "Steim-2"),
        ("Sample Rate", f"{sr:.0f} Hz"),
        ("N Samples",   f"{ns:,}"),
        ("Duration",    f"{ns/sr:.3f} s"),
        ("Start",       sdt.strftime("%Y-%m-%d %H:%M:%S")),
        ("End",         edt.strftime("%H:%M:%S UTC")),
    ]
    col2 = [
        ("SIGNAL",          None),
        ("Dominant Freq.",  f"{fa['peak_f']:.3f} Hz"),
        ("Period",          f"{fa['peak_T']:.4f} s"),
        ("RMS Amplitude",   f"{fa['rms']:,.1f} counts"),
        ("Peak Amplitude",  f"{fa['peak_amp']:,.1f} counts"),
        ("Peak Time",       f"{t[pidx]:.3f} s"),
        ("Bandpass",        "1-45 Hz"),
        ("Filter",          "Butterworth (4-pole)"),
        ("Phase",           "Zero-phase (filtfilt)"),
    ]
    col3 = [("SECTIONS", None)]
    for lbl, seg in sections.items():
        col3.append((lbl,         f"{len(seg):,} samples"))
        col3.append(("  min/max", f"{seg.min():.0f} / {seg.max():.0f}"))
        col3.append(("  mean",    f"{seg.mean():.1f}"))

    # ── Draw column — each row exactly lh apart, guaranteed no overlap ────────
    def draw_column(items, x_hdr, x_key, x_val, y_top, lh):
        y = y_top
        for key, val in items:
            if val is None:                        # section header row
                ax7.text(x_hdr, y, key,
                         transform=ax7.transAxes, va="top",
                         fontsize=9, fontweight="bold",
                         color=ORANGE, fontfamily="monospace")
                y -= lh                            # ONE full lh below header
            else:                                  # data row
                ax7.text(x_key, y, key + ":",
                         transform=ax7.transAxes, va="top",
                         fontsize=8, color=MUTED)
                ax7.text(x_val, y, str(val),
                         transform=ax7.transAxes, va="top",
                         fontsize=8, color=TEXT, fontweight="bold")
                y -= lh                            # ONE full lh per data row

    lh = 0.082   # line height — same for every row, no fractional steps
    y0 = 0.82    # start below the divider line

    draw_column(col1, x_hdr=0.015, x_key=0.015, x_val=0.150, y_top=y0, lh=lh)
    draw_column(col2, x_hdr=0.350, x_key=0.350, x_val=0.490, y_top=y0, lh=lh)
    draw_column(col3, x_hdr=0.685, x_key=0.685, x_val=0.810, y_top=y0, lh=lh)

    # Vertical column dividers — safely below divider line, above bottom
    for xd in [0.335, 0.670]:
        ax7.plot([xd, xd], [0.02, 0.85],
                 transform=ax7.transAxes,
                 color=BORDER, lw=1.0, alpha=0.9, clip_on=False)

    # ── Maximise window and show ──────────────────────────────────────────────
    manager = plt.get_current_fig_manager()
    try:
        manager.window.state("zoomed")
    except Exception:
        try:
            manager.full_screen_toggle()
        except Exception:
            pass

    print("[✓]  Figure displayed — close the window to continue.\n")
    plt.show()


# =============================================================================
#  RUNNER  —  End-to-end analysis of one file
# =============================================================================

def analyse_file(filepath: str):
    fname = os.path.basename(filepath)

    print("=" * 65)
    print(f"  FILE : {fname}")
    print("=" * 65)

    with open(filepath, "rb") as fh:
        raw = fh.read()
    print(f"[i] Loaded  {len(raw):,} bytes\n")

    # ── PART 1A : header ─────────────────────────────────────────────────────
    print("[PART 1]  Extracting data from MSEED ...")
    meta = parse_header(raw)
    sr   = meta["sample_rate"]
    ns   = meta["num_samples"]
    print(f"  Network / Station / Channel : "
          f"{meta['network']}.{meta['station']}.{meta['channel']}")
    print(f"  Start      : {meta['start_dt'].strftime('%Y-%m-%d  %H:%M:%S UTC')}")
    print(f"  Fs         : {sr} Hz   |   N samples = {ns}")
    print(f"  Encoding   : Steim-2  (Big-Endian)   |   "
          f"Data offset = {meta['data_offset']} bytes")

    # ── PART 1B : decode Steim-2 ─────────────────────────────────────────────
    samples = decode_steim2(raw[meta["data_offset"]:], ns).astype(np.float64)
    print(f"  Decoded    : {len(samples)} samples  "
          f"[min={samples.min():.0f}  "
          f"max={samples.max():.0f}  "
          f"mean={samples.mean():.1f}]")

    # ── PART 1C : extract three equal-thirds sections ────────────────────────
    dur  = ns / sr
    wins = [
        (0.0,        dur * 0.33),
        (dur * 0.33, dur * 0.66),
        (dur * 0.66, dur),
    ]
    print(f"\n  Extracting 3 sections from {dur:.2f} s record:")
    sections = extract_sections(samples, sr, wins)

    # ── PART 2 : waveform processing ─────────────────────────────────────────
    print("\n[PART 2]  Waveform processing ...")
    wf = process_waveform(samples, sr, f_low=1.0, f_high=45.0)
    rms = np.sqrt(np.mean(wf["detrended"] ** 2))
    pk  = np.max(np.abs(wf["filtered"]))
    print(f"  RMS amplitude  : {rms:,.1f} counts")
    print(f"  Peak filtered  : {pk:,.1f} counts  "
          f"@ t = {wf['time'][wf['peak_idx']]:.3f} s")

    # ── PART 3 : frequency analysis ──────────────────────────────────────────
    print("\n[PART 3]  Frequency analysis ...")
    fa = frequency_analysis(wf["detrended"], sr)
    print(f"  Dominant freq  : {fa['peak_f']:.3f} Hz  "
          f"(period = {fa['peak_T']:.4f} s)")
    print(f"  Peak amplitude : {fa['peak_amp']:,.1f} counts")
    print(f"  RMS amplitude  : {fa['rms']:,.1f} counts")

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("\n[PLOT]  Generating figure ...")
    plot_results(wf, fa, meta, sections, fname)


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # mseed_data folder is ALWAYS next to this script — same place the
    # recorder writes to when using the fixed record_continuous_mseed.py
    mseed_data_dir = os.path.join(script_dir, "mseed_data")

    search_dirs = [
        mseed_data_dir,                            # ← primary: where recorder saves
        script_dir,                                # ← secondary: files placed manually
        os.path.join(os.getcwd(), "mseed_data"),   # ← fallback: cwd/mseed_data
        os.getcwd(),                               # ← fallback: cwd itself
    ]

    print(f"\n  Looking for .mseed files in:")
    for d in search_dirs:
        exists = "✓" if os.path.isdir(d) else "✗ (not found)"
        print(f"    [{exists}] {d}")

    # ── Collect candidate files ───────────────────────────────────────────────
    if len(sys.argv) > 1:
        candidates = []
        for arg in sys.argv[1:]:
            hits = glob.glob(arg)
            if not hits:
                for d in search_dirs:
                    hits = glob.glob(os.path.join(d, arg))
                    if hits:
                        break
            candidates.extend(hits)
        if not candidates:
            print(f"\n[!] No files matched: {sys.argv[1:]}")
            print(f"    Searched in: {search_dirs}\n")
            sys.exit(1)
        candidates = sorted(set(candidates))
    else:
        # Auto-detect .mseed files from all search locations, newest file first
        seen = set()
        candidates = []
        for d in search_dirs:
            for fp in glob.glob(os.path.join(d, "*.mseed")):
                real = os.path.realpath(fp)
                if real not in seen:
                    seen.add(real)
                    candidates.append(fp)
        # Sort newest-modified first so latest recordings appear at top of menu
        candidates.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        if not candidates:
            print("\n[!] No .mseed files found.")
            print("    Searched in:")
            for d in search_dirs:
                print(f"      {d}")
            print("\n    Make sure record_continuous_mseed.py is running")
            print("    and saving files to the mseed_data/ folder.\n")
            sys.exit(1)

    # ── Single file → run directly ────────────────────────────────────────────
    if len(candidates) == 1:
        analyse_file(candidates[0])
        sys.exit(0)

    # ── Multiple files → interactive menu ─────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Found {len(candidates)} MSEED file(s)  —  newest first")
    print(f"{'='*65}\n")

    # Show up to 500 files; if more, only show newest 500
    display = candidates[:500]
    for i, fp in enumerate(display, start=1):
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fp))
        size  = os.path.getsize(fp)
        print(f"  [{i:>4}]  {os.path.basename(fp):<45}  "
              f"{size/1024:>5.1f} KB   {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    if len(candidates) > 500:
        print(f"\n  ... and {len(candidates)-500} more older files (not shown)")

    print(f"\n  [   0]  Process ALL files one by one")
    print(f"  [  -1]  Process latest file automatically\n")

    choice = input("Enter number to analyse (0=all, -1=latest): ").strip()

    if choice == "-1":
        print(f"\n  → Analysing latest: {os.path.basename(candidates[0])}\n")
        analyse_file(candidates[0])
    elif choice == "0":
        for fp in candidates:
            analyse_file(fp)
    elif choice.lstrip("-").isdigit():
        idx = int(choice)
        if 1 <= idx <= len(candidates):
            analyse_file(candidates[idx - 1])
        else:
            print(f"\n[!] Invalid choice '{choice}'. Must be 1–{len(candidates)}.\n")
            sys.exit(1)
    else:
        print(f"\n[!] Invalid choice '{choice}'. Exiting.\n")
        sys.exit(1)
