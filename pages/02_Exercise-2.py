"""
Exercise 2: Frequency Domain Analysis
DSP Assignment – Hannah Wimmer
Streamlit page for the pages/ folder of the existing app.

Place this file in your pages/ directory as:
    pages/2_Frequency_Domain_Analysis.py

The data files (signal.csv, events.csv) are expected either:
  - in the same directory as this file, OR
  - adjustable via the sidebar file-path inputs
"""

import streamlit as st
import numpy as np
import os
import json

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
try:
    st.set_page_config(
        page_title="Exercise 2 – Frequency Domain Analysis",
        page_icon="📡",
        layout="wide",
    )
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
div[data-testid="stAlertContainer"] {
    background: rgba(126, 48, 225, 0.3); 
    border-left: 4px solid rgba(126, 48, 225, 1);
    padding: 12px 16px; 
    border-radius: 0 6px 6px 0; 
    margin: 10px 0; 
    font-size: 0.92em;
}
.warn-box {
    background: rgba(255, 155, 69, 0.1); 
    border-left: 4px solid rgba(255, 155, 69, 1);
    padding: 12px 16px; 
    border-radius: 0 6px 6px 0; 
    margin: 10px 0; 
    font-size: 0.92em;
}
            
/* Main sidebar container */
section[data-testid="stSidebar"] {
    background-color: rgba(255, 247, 209, 1);
    border-right: 1px solid rgba(255, 247, 209, 1);
}

/* Sidebar content wrapper */
section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem;
}

/* Divider */
hr {
    border-color: rgba(227, 106, 106, 1);
}

/* Main background */
.stApp, .stAppHeader {
    background-color: rgba(255, 251, 241, 1);
}

/* General text and headers */
p, span, div, h1, h2, h3, h4 {
    color: rgba(114, 35, 35, 1);
}

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# DSP CORE – manual DFT, no np.fft used for the transform itself
# ═══════════════════════════════════════════════════════════════


def dft_manual(x: np.ndarray) -> np.ndarray:
    """
    Discrete Fourier Transform computed from first principles.

        X[k] = sum_{n=0}^{N-1}  x[n] * exp(-j * 2*pi * k * n / N)

    Implemented as chunked matrix-vector products (no fft function anywhere).
    chunk=512 keeps peak RAM reasonable even for N=4096.
    """
    N = len(x)
    n = np.arange(N)
    X = np.zeros(N, dtype=complex)
    chunk = 512
    for k0 in range(0, N, chunk):
        k1 = min(k0 + chunk, N)
        ks = np.arange(k0, k1)
        W = np.exp(-2j * np.pi * np.outer(ks, n) / N)  # DFT matrix rows k0..k1
        X[k0:k1] = W @ x
    return X


def freq_axis(N: int, fs: float) -> np.ndarray:
    """One-sided frequency axis in Hz (arithmetic only, no fftfreq)."""
    return np.arange(N // 2) * (fs / N)


def one_sided_magnitude(X: np.ndarray, N: int, window_power: float = 1.0) -> np.ndarray:
    """Amplitude-corrected one-sided magnitude spectrum."""
    return (2.0 / (N * window_power)) * np.abs(X[: N // 2])


def power_db(X: np.ndarray, N: int, window_power: float = 1.0) -> np.ndarray:
    pwr = (np.abs(X[: N // 2]) / (N * window_power)) ** 2 * 2
    pwr = np.where(pwr > 0, pwr, 1e-30)
    return 10.0 * np.log10(pwr)


def hann_power(w: np.ndarray) -> float:
    """RMS power of a window for amplitude correction."""
    return float(np.sqrt(np.mean(w**2)))


def find_peaks_manual(
    mag: np.ndarray,
    freqs: np.ndarray,
    min_height_frac: float = 0.04,
    min_dist_hz: float = 1.5,
) -> list:
    """Local-maxima peak finder. Returns list of (idx, freq_hz, amplitude)."""
    threshold = mag.max() * min_height_frac
    df = freqs[1] - freqs[0]
    min_dist_i = max(1, int(min_dist_hz / df))
    peaks = []
    # Skip DC bin (index 0)
    for i in range(2, len(mag) - 1):
        if mag[i] >= threshold and mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]:
            if not peaks or (i - peaks[-1][0]) >= min_dist_i:
                peaks.append((i, freqs[i], mag[i]))
    peaks.sort(key=lambda x: -x[2])
    return peaks[:12]


# ═══════════════════════════════════════════════════════════════
# Data loading (cached)
# ═══════════════════════════════════════════════════════════════


@st.cache_data(show_spinner="Loading signal …")
def load_signal(path: str):
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    time       = raw[:, 0].astype(float)
    signal     = raw[:, 1].astype(float)
    dt         = time[1] - time[0]
    fs         = round(1.0 / dt)
    max_jitter = float(np.max(np.abs(np.diff(time) - dt)))
    return time, signal, float(fs), dt, max_jitter


@st.cache_data(show_spinner="Loading events …")
def load_events(path: str) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    return raw[:, 1].astype(float)


# ═══════════════════════════════════════════════════════════════
# Spectrum computation (cached)
# ═══════════════════════════════════════════════════════════════


@st.cache_data(show_spinner="Running manual DFT …")
def compute_spectrum(seg_key: tuple, segment: np.ndarray, fs: float):
    """
    Hann-windowed, mean-removed DFT of `segment`.
    seg_key is a tuple used only for cache keying (e.g. (hash, N, fs)).
    Returns (freqs, mag, pdb).
    """
    N = len(segment)
    win = np.hanning(N)
    wp = hann_power(win)
    seg_w = (segment - segment.mean()) * win
    X = dft_manual(seg_w)
    freqs = freq_axis(N, fs)
    mag = one_sided_magnitude(X, N, wp)
    pdb = power_db(X, N, wp)
    return freqs, mag, pdb


@st.cache_data(show_spinner="Extracting epochs …")
def compute_epochs(
    signal: np.ndarray, fs: float, event_times: np.ndarray, pre_s: float, post_s: float
):
    ep_pre = int(pre_s * fs)
    ep_post = int(post_s * fs)
    total = ep_pre + ep_post
    epochs = []
    for et in event_times:
        idx = int(et * fs)
        start = idx - ep_pre
        end = idx + ep_post
        if start >= 0 and end < len(signal):
            bl = signal[start : start + ep_pre].mean()
            epochs.append(signal[start:end] - bl)
    epochs = np.array(epochs, dtype=float)
    t = np.linspace(-pre_s, post_s, total)
    return epochs, t


# ═══════════════════════════════════════════════════════════════
# Plotly inline-HTML helpers (no matplotlib dependency)
# ═══════════════════════════════════════════════════════════════

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"
_uid_counter = [0]


def _uid():
    _uid_counter[0] += 1
    return f"plt{_uid_counter[0]}"


DARK = {
    "plot_bgcolor": "#0e1117",
    "paper_bgcolor": "#0e1117",
    "font": {"color": "#cccccc"},
    "xaxis": {"color": "#aaaaaa", "gridcolor": "#1e2a3a", "zerolinecolor": "#2a3a50"},
    "yaxis": {"color": "#aaaaaa", "gridcolor": "#1e2a3a", "zerolinecolor": "#2a3a50"},
    "margin": {"t": 45, "b": 45, "l": 65, "r": 20},
}


def _merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def plotly_html(traces: list, layout_extra: dict, height: int = 260) -> str:
    uid = _uid()
    layout = _merge(DARK, layout_extra)
    layout["height"] = height
    return (
        f'<div id="{uid}" style="width:100%"></div>'
        f'<script src="{PLOTLY_CDN}"></script>'
        f"<script>Plotly.newPlot('{uid}', {json.dumps(traces)}, "
        f"{json.dumps(layout)}, {{responsive:true,displayModeBar:false}});</script>"
    )


def render(html: str, height: int = 270):
    st.components.v1.html(html, height=height, scrolling=False)


# ═══════════════════════════════════════════════════════════════
# PAGE
# ═══════════════════════════════════════════════════════════════


def main():
    st.title("📡 Exercise 2 – Frequency Domain Analysis")
    st.caption("DSP / iDSP  ·  Angel Vellayil Antony | Leni Iype  ·  March 2026")

    # ── Load data first (needed for seg_start slider max_value) ───
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data/exercise_02")
    sig_default = os.path.join(DATA_DIR, "signal.csv")
    evt_default = os.path.join(DATA_DIR, "events.csv")

    try:
        time, signal, fs, dt, max_jitter = load_signal(sig_default)
        event_times = load_events(evt_default)
    except Exception as exc:
        st.error(f"Could not load data: {exc}\n\nAdjust paths in the sidebar.")
        st.stop()

    N_total = len(signal)
    duration = time[-1]
    nyquist = fs / 2.0

    # ── Sidebar ───────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("DFT Segment")
        dft_seg = st.select_slider(
            "Segment length (samples)",
            options=[512, 1024, 2048, 4096],
            value=4096,
            help="Longer → finer frequency resolution (Δf = fs/N).",
        )
        seg_start = st.slider(
            "Segment start (samples)",
            min_value=0,
            max_value=max(0, N_total - dft_seg),
            value=0,
            step=dft_seg // 4,
            help="Slide to analyze a different part of the recording.",
        )

        st.divider()
        st.subheader("Downsampling")
        ds_factor = st.select_slider(
            "Downsampling factor  (no filter)",
            options=[1, 2, 4, 8, 16, 32, 64],
            value=1,
        )

        st.divider()
        st.subheader("Epoch Window")
        pre_s = st.slider("Pre-stimulus (s)", 0.5, 3.0, 1.0, 0.5)
        post_s = st.slider("Post-stimulus (s)", 2.0, 12.0, 6.0, 0.5)

    # ── Downsample ────────────────────────────────────────────────
    sig_ds = signal[::ds_factor]
    time_ds = time[::ds_factor]
    fs_ds = fs / ds_factor

    # ── Spectra ───────────────────────────────────────────────────
    N_orig = min(dft_seg, N_total - seg_start)
    seg_o = signal[seg_start : seg_start + N_orig]
    freqs_o, mag_o, pdb_o = compute_spectrum(
        (hash(seg_o.tobytes()), N_orig, fs), seg_o, fs
    )
    peaks_o = find_peaks_manual(mag_o, freqs_o)

    N_ds = min(dft_seg, len(sig_ds))
    seg_d = sig_ds[:N_ds]
    freqs_d, mag_d, pdb_d = compute_spectrum(
        (int(seg_d[0] * 1e6), N_ds, fs_ds), seg_d, fs_ds
    )

    # ── Epochs ────────────────────────────────────────────────────
    epochs, t_ep = compute_epochs(signal, fs, event_times, pre_s, post_s)
    mean_ep = epochs.mean(axis=0)
    std_ep = epochs.std(axis=0)
    ieis = np.diff(event_times)

    # ═══════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📖 Introduction",
            "🔬 Methods",
            "📊 Results",
            "💬 Discussion",
        ]
    )

    # ──────────────────────────────────────────────────────────
    # TAB 1 – INTRODUCTION
    # ──────────────────────────────────────────────────────────
    with tab1:
        st.header("Introduction to the Discrete Fourier Transform")

        st.markdown(r"""
The **Discrete Fourier Transform (DFT)** decomposes a finite discrete-time signal into a sum of
complex sinusoids, revealing which frequencies carry energy. It is the cornerstone of digital
signal processing and is used everywhere from audio compression to brain-computer interfaces.
""")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("The DFT definition")
            st.markdown(r"""
For a discrete signal $x[n]$, $n = 0 \ldots N-1$, the DFT is:

$$X[k] = \sum_{n=0}^{N-1} x[n]\; e^{-j2\pi kn/N}, \quad k=0,\ldots,N-1$$

$X[k]$ is **complex**: its magnitude gives the amplitude of that frequency component,
its angle gives the phase.  The corresponding physical frequency is $f_k = k \cdot f_s / N$.

**Inverse DFT** (perfect reconstruction):

$$x[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k]\; e^{j2\pi kn/N}$$

In this exercise the DFT is implemented entirely from the definition above using
chunked matrix–vector products; no `np.fft` function is called anywhere.
""")

        with c2:
            st.subheader("Frequency axis & resolution")
            st.markdown(rf"""
Each bin $k$ maps to:

$$f_k = \frac{{k \cdot f_s}}{{N}} \quad \text{{Hz}}$$

The **frequency resolution** (bin spacing) is:

$$\Delta f = \frac{{f_s}}{{N}}$$

For this recording with $f_s = {fs:.0f}$ Hz:

| Segment $N$ | $\Delta f$ |
|---|---|
| 512  | {fs / 512:.4f} Hz |
| 1024 | {fs / 1024:.4f} Hz |
| 2048 | {fs / 2048:.4f} Hz |
| 4096 | {fs / 4096:.4f} Hz |

Longer segments → finer resolution, but also average over a longer time window
(stationarity trade-off).
""")

        st.subheader("The Nyquist Criterion")
        st.markdown(rf"""
The **Nyquist–Shannon sampling theorem** states that to perfectly reconstruct a band-limited
signal with maximum frequency $f_{{max}}$, the sampling rate must satisfy:

$$f_s \geq 2 \cdot f_{{max}}$$

The **Nyquist frequency** $f_N = f_s / 2$ is the highest frequency that can be represented.
Any energy **above** $f_N$ present in the signal **aliases** — it folds back onto lower
frequencies, irrecoverably corrupting them.

For this recording: $f_s = {fs:.0f}$ Hz → $f_N = {nyquist:.0f}$ Hz.  The physiological signal
of interest (EDA) is entirely below 5 Hz, so the recording is **heavily oversampled** by design —
a common practice that leaves headroom for quality checking and artefact capture.
""")

        st.subheader("Magnitude vs. Power spectrum")
        st.markdown(r"""
| Quantity | Formula (amplitude-corrected) | Units | Interpretation |
|---|---|---|---|
| **Magnitude spectrum** | (2 / N) · \| X[ k ] \| | signal units | Amplitude of each frequency component |
| **Power spectrum** | ((2 · \| X[ k ] \| ) / N)² | units² | Energy; Parseval: sum = signal variance |
| **Power (dB)** | 10 · log₁₀(power) | dB | Wide dynamic range; noise floor visible |

**How to read spectral peaks:**
- **Narrow spike** → single-frequency source (power line, clock harmonic, electrode oscillation)
- **Broad hump** → band-limited oscillation (neural rhythm, breathing, movement)
- Peak **height** = sinusoidal amplitude; peak **width** ∝ frequency instability over the window

**Spectral leakage & windowing:** the DFT assumes the segment repeats periodically.
A sinusoid that does not complete an integer number of cycles in the window leaks energy into
neighbouring bins.  Multiplying by a **Hann window** tapers the endpoints to zero, greatly
reducing leakage at the cost of a slight broadening of real peaks.
""")

    # ──────────────────────────────────────────────────────────
    # TAB 2 – METHODS
    # ──────────────────────────────────────────────────────────
    with tab2:
        st.header("Methods")

        st.subheader("Signal type and recording")
        st.markdown(rf"""
The dataset is a continuous **Electrodermal Activity (EDA)** recording (also known as
Galvanic Skin Response, GSR), captured at **{fs:.0f} Hz** for **{duration / 60:.1f} minutes**
({N_total:,} samples total).  EDA measures skin conductance via sweat-gland activity and is a
canonical psychophysiological index of sympathetic arousal.  Amplitude values (~0.5–1.5,
likely µS) and waveform morphology are consistent with standard EDA recordings from finger
or palm electrodes.

**{len(event_times)} stimulus events** are provided in a separate file recording onset times
(in seconds) with jittered inter-trial intervals (~{ieis.mean():.0f} s mean).
""")

        st.subheader("Sampling rate and Nyquist implication")
        st.markdown(rf"""
The uniform time step $\Delta t = {1 / fs:.8f}$ s gives:

$$f_s = {fs:.0f}\;\text{{Hz}}, \qquad f_N = {nyquist:.0f}\;\text{{Hz}}$$

EDA contains physiologically meaningful information only below ~5 Hz (phasic SCR) and
~0.05 Hz (tonic drift).  The recording is therefore **oversampled by a factor of ~25×**
relative to the signal bandwidth.  This is intentional: it preserves high-frequency
artefacts (powerline noise, movement) for quality-control purposes and ensures the phasic
EDA waveform is captured with high temporal fidelity.
""")

        st.subheader("Sampling rate derivation")
        st.markdown(rf"""
The sampling rate is **derived directly from the timestamp column**, not assumed:

$$\Delta t = t_1 - t_0 = {dt:.8f}\;\text{{s}}$$

$$f_s = \frac{{1}}{{\Delta t}} = \mathbf{{{fs:.2f}\;\text{{Hz}}}}$$

Verified uniform across all {N_total:,} samples
(max timestamp jitter: **{max_jitter:.2e} s**).

$$f_N = \frac{{f_s}}{{2}} = {nyquist:.2f}\;\text{{Hz}}$$

Safety margin over EDA band (< 5 Hz): **{nyquist/5:.0f}×**
""")

        st.subheader("Spectral analysis pipeline")
        dft_formula = r"X[k] = sum_{n=0}^{N-1}  x[n] * exp(-j * 2*pi * k * n / N)"
        st.markdown(f"""
1. **Segment selection** – the first `{dft_seg}` samples are used (adjustable in sidebar).
   Frequency resolution: Δf = {fs / dft_seg:.5f} Hz/bin.

2. **DC removal** – subtract the segment mean to prevent a dominant DC spike at f=0 Hz.

3. **Hann windowing** – multiply by a Hann window to suppress spectral leakage.
   Amplitudes are corrected for the window's RMS power.

4. **Manual DFT** – every spectral coefficient is computed from the definition:
   ```
   {dft_formula}
   ```
   using chunked 512-row DFT matrices (no FFT library called anywhere).

5. **One-sided output** – only bins 0…N/2−1 are shown; amplitudes are doubled to
   preserve total power.

6. **Peak detection** – local maxima above 4 % of the global peak, spaced ≥ 1.5 Hz apart,
   are labelled automatically.
""")

        st.subheader("Event-related analysis")
        st.markdown(rf"""
Epochs are extracted from **{pre_s:.1f} s before** to **{post_s:.1f} s after** each event
onset (adjustable in sidebar).  Each epoch is baseline-corrected by subtracting the mean of
the pre-stimulus window.  The **grand average** (mean ± SD across all {len(epochs)} valid
epochs) yields the mean Skin Conductance Response (SCR) waveform.  Trials where the window
extends beyond the recording boundary are excluded.
""")

        st.subheader("Downsampling")
        st.markdown("""
The signal is naively decimated — every *k*-th sample is kept, with **no anti-aliasing filter**
applied, as required by the exercise.  This deliberately induces aliasing for controlled
comparison.  The spectra of original and downsampled signals are displayed side by side.
""")

    # ──────────────────────────────────────────────────────────
    # TAB 3 – RESULTS
    # ──────────────────────────────────────────────────────────
    with tab3:
        st.header("Results")

        # ── Key metrics ───────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sampling rate", f"{fs:.0f} Hz")
        c2.metric("Duration", f"{duration / 60:.1f} min")
        c3.metric("Nyquist freq.", f"{nyquist:.0f} Hz")
        c4.metric("No. of events", str(len(event_times)))
        c5.metric("Mean ITI", f"{ieis.mean():.1f} s")

        # ── Full time-series ──────────────────────────────────
        st.subheader("1. Raw EDA recording")
        thin = max(1, N_total // 5000)
        t_p = time[::thin].tolist()
        s_p = signal[::thin].tolist()
        ev_shapes = [
            {
                "type": "line",
                "x0": et,
                "x1": et,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "rgba(224,123,84,0.5)", "width": 1},
            }
            for et in event_times
        ]
        seg_shapes = ev_shapes + [{
            "type": "rect",
            "x0": seg_start / fs,
            "x1": (seg_start + N_orig) / fs,
            "y0": 0, "y1": 1, "yref": "paper",
            "fillcolor": "rgba(168,230,163,0.15)",
            "line": {"width": 0},
            "layer": "below",
        }]
        sig_trace = {
            "type": "scatter",
            "x": t_p,
            "y": s_p,
            "mode": "lines",
            "line": {"color": "#4f8ef7", "width": 0.7},
            "name": "EDA",
        }
        render(
            plotly_html(
                [sig_trace],
                {
                    "title": {
                        "text": "Full EDA recording with event markers (orange lines) · green = DFT segment",
                        "font": {"size": 13},
                    },
                    "xaxis": {"title": "Time (s)"},
                    "yaxis": {"title": "EDA (µS)"},
                    "shapes": seg_shapes,
                },
                height=250,
            ),
            height=265,
        )

        # ── Spectra ───────────────────────────────────────────
        st.subheader("2. Frequency-domain analysis (manual DFT)")
        st.caption(
            f"Segment: first {N_orig} samples · Hann window · Δf = {fs / N_orig:.4f} Hz/bin"
        )

        col_m, col_p = st.columns(2)

        # ---- Magnitude ----
        with col_m:
            ann_m = []
            for _, pf, pm in peaks_o[:6]:
                ann_m.append(
                    {
                        "x": float(pf),
                        "y": float(pm),
                        "text": f"{pf:.2f} Hz",
                        "showarrow": True,
                        "arrowhead": 2,
                        "arrowsize": 0.7,
                        "arrowcolor": "#e74c3c",
                        "font": {"size": 9, "color": "#e74c3c"},
                        "ax": 0,
                        "ay": -22,
                    }
                )
            mag_tr = {
                "type": "scatter",
                "x": freqs_o.tolist(),
                "y": mag_o.tolist(),
                "mode": "lines",
                "fill": "tozeroy",
                "line": {"color": "#f7a04f", "width": 1.3},
                "fillcolor": "rgba(247,160,79,0.13)",
            }
            render(
                plotly_html(
                    [mag_tr],
                    {
                        "title": {"text": "Magnitude spectrum", "font": {"size": 12}},
                        "xaxis": {"title": "Frequency (Hz)", "range": [0, nyquist]},
                        "yaxis": {"title": "Amplitude (µS)"},
                        "annotations": ann_m,
                    },
                ),
                height=275,
            )

        # ---- Power (dB) ----
        with col_p:
            pdb_tr = {
                "type": "scatter",
                "x": freqs_o.tolist(),
                "y": pdb_o.tolist(),
                "mode": "lines",
                "fill": "tozeroy",
                "line": {"color": "#7ec8e3", "width": 1.3},
                "fillcolor": "rgba(126,200,227,0.10)",
            }
            render(
                plotly_html(
                    [pdb_tr],
                    {
                        "title": {"text": "Power spectrum (dB)", "font": {"size": 12}},
                        "xaxis": {"title": "Frequency (Hz)", "range": [0, nyquist]},
                        "yaxis": {"title": "Power (dB)"},
                    },
                ),
                height=275,
            )

        # ---- Physiological band zoom (0–10 Hz) ----
        st.markdown("**Physiological band zoom: 0–10 Hz**")
        lf = freqs_o <= 10.0
        lf_tr = {
            "type": "scatter",
            "x": freqs_o[lf].tolist(),
            "y": mag_o[lf].tolist(),
            "mode": "lines",
            "fill": "tozeroy",
            "line": {"color": "#a8e6a3", "width": 1.5},
            "fillcolor": "rgba(168,230,163,0.13)",
        }
        render(
            plotly_html(
                [lf_tr],
                {
                    "title": {
                        "text": "Magnitude spectrum – physiological band (0–10 Hz)",
                        "font": {"size": 12},
                    },
                    "xaxis": {"title": "Frequency (Hz)", "range": [0, 10]},
                    "yaxis": {"title": "Amplitude (µS)"},
                },
                height=230,
            ),
            height=245,
        )

        # ---- Peak table ----
        st.markdown("**Detected spectral peaks:**")

        def interpret(f: float) -> str:
            if abs(f - 50.0) < 0.3:
                return "⚡ Powerline interference (50 Hz, EU mains)"
            if abs(f - 100.0) < 0.3:
                return "⚡ 2nd harmonic of powerline (100 Hz)"
            if abs(f - 16.625) < 0.2:
                return "🔀 Sub-harmonic artefact (50/3 Hz) – hardware intermod"
            if abs(f - 33.25) < 0.2:
                return "🔀 Sub-harmonic artefact (50·2/3 Hz)"
            if f < 0.5:
                return "🌊 Tonic EDA drift (very slow baseline)"
            if f < 5.0:
                return "💧 Phasic EDA / SCR activity"
            if f < 15.0:
                return "🫁 Possible respiratory or movement artefact"
            return "❓ Unknown / electrode artefact"

        rows = {
            "Frequency (Hz)": [f"{pf:.3f}" for (_, pf, _) in peaks_o[:10]],
            "Amplitude (µS)": [f"{pm:.5f}" for (_, _, pm) in peaks_o[:10]],
            "Likely source": [interpret(pf) for (_, pf, _) in peaks_o[:10]],
        }
        if HAS_PANDAS:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            cols = st.columns([1.2, 1.5, 3])
            for header, col in zip(rows.keys(), cols):
                col.markdown(f"**{header}**")
            for i in range(len(rows["Frequency (Hz)"])):
                c = st.columns([1.2, 1.5, 3])
                c[0].write(rows["Frequency (Hz)"][i])
                c[1].write(rows["Amplitude (µS)"][i])
                c[2].write(rows["Likely source"][i])

        # ── Signal vs Noise decomposition ────────────────────
        st.subheader("Signal vs. Noise decomposition")

        SIGNAL_BAND = (0.0, 5.0)
        NOISE_BANDS = [(48.0, 52.0), (98.0, 102.0), (16.0, 18.0)]

        def band_power(mag, freqs, flo, fhi):
            mask = (freqs >= flo) & (freqs <= fhi)
            return float(np.sum(mag[mask] ** 2))

        total_pwr  = band_power(mag_o, freqs_o, 0, nyquist)
        signal_pwr = band_power(mag_o, freqs_o, *SIGNAL_BAND)
        noise_pwr  = sum(band_power(mag_o, freqs_o, lo, hi) for lo, hi in NOISE_BANDS)
        other_pwr  = max(0.0, total_pwr - signal_pwr - noise_pwr)
        snr_db     = 10 * np.log10(signal_pwr / noise_pwr) if noise_pwr > 0 else float("inf")

        col_pie, col_snr = st.columns([2, 1])
        with col_pie:
            pie_trace = [{
                "type": "pie",
                "labels": ["Signal (0–5 Hz)", "Powerline / HW noise", "Other"],
                "values": [signal_pwr, noise_pwr, other_pwr],
                "marker": {"colors": ["#a8e6a3", "#e74c3c", "#7ec8e3"]},
                "textinfo": "label+percent",
                "hole": 0.35,
            }]
            render(plotly_html(pie_trace,
                {"title": {"text": "Power distribution by band", "font": {"size": 12}}},
                height=270), height=285)
        with col_snr:
            st.metric("Signal power (0–5 Hz)",  f"{signal_pwr:.4f} µS²")
            st.metric("Noise power (HF bands)", f"{noise_pwr:.6f} µS²")
            st.metric("SNR", f"{snr_db:.1f} dB")
            st.caption("Signal: 0–5 Hz. Noise bands: 48–52 Hz, 98–102 Hz, 16–18 Hz.")

        # ── Event analysis ────────────────────────────────────
        st.subheader("3. Event-related analysis (mean SCR)")

        # Compute per-trial peak stats for metrics
        scr_mask = (t_ep >= 1.0) & (t_ep <= 6.0)
        per_peaks_a = []
        per_peaks_t = []
        bl_stds = []
        for ep in epochs:
            scr_seg = ep[scr_mask]
            pi = np.argmax(scr_seg)
            per_peaks_a.append(scr_seg[pi])
            per_peaks_t.append(t_ep[scr_mask][pi])
            bl_stds.append(np.std(ep[t_ep < 0]))
        per_peaks_a = np.array(per_peaks_a)
        per_peaks_t = np.array(per_peaks_t)
        bl_stds = np.array(bl_stds)
        responders = per_peaks_a > 2 * bl_stds

        c1e, c2e, c3e, c4e = st.columns(4)
        c1e.metric("Valid epochs", str(len(epochs)))
        c2e.metric("Response rate", f"{100 * responders.mean():.0f}%")
        c3e.metric("Peak latency", f"{np.median(per_peaks_t):.2f} s (median)")
        c4e.metric("Peak amplitude", f"{np.median(per_peaks_a):.3f} µS (median)")

        # ── Event timing analysis ─────────────────────────────
        st.subheader("Event timing analysis")

        scr_window    = (t_ep >= 0) & (t_ep <= post_s)
        mean_scr_post = mean_ep[scr_window]
        peak_val      = mean_scr_post.max()

        onset_idx     = np.where(mean_scr_post >= 0.05 * peak_val)[0]
        scr_onset_lat = float(t_ep[scr_window][onset_idx[0]]) if len(onset_idx) else 0.0

        recovery_idx      = np.where(mean_scr_post >= 0.10 * peak_val)[0]
        response_duration = float(t_ep[scr_window][recovery_idx[-1]]) if len(recovery_idx) else post_s

        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        col_d1.metric("Stimulus onset",        "t = 0 s")
        col_d2.metric("SCR onset latency",     f"{scr_onset_lat:.2f} s")
        col_d3.metric("SCR response duration", f"{response_duration:.1f} s  (10% recovery)")
        col_d4.metric("Mean IEI",              f"{ieis.mean():.1f} s  (SD {ieis.std():.1f})")

        iei_hist_y, iei_hist_x = np.histogram(ieis, bins=12)
        iei_bar = {
            "type": "bar",
            "x": ((iei_hist_x[:-1] + iei_hist_x[1:]) / 2).tolist(),
            "y": iei_hist_y.tolist(),
            "marker": {"color": "#f7a04f"},
        }
        render(plotly_html([iei_bar], {
            "title": {"text": "Inter-event interval distribution", "font": {"size": 12}},
            "xaxis": {"title": "IEI (s)"},
            "yaxis": {"title": "Count"},
        }, height=220), height=235)

        st.caption(
            f"IEI range: {ieis.min():.1f}–{ieis.max():.1f} s · "
            f"Jitter coefficient (SD/mean): {ieis.std()/ieis.mean():.2f} · "
            f"SCR duration = time from stimulus onset to 10% peak recovery on the grand average."
        )

        # Mean waveform ± SD
        upper = (mean_ep + std_ep).tolist()
        lower = (mean_ep - std_ep).tolist()
        tl = t_ep.tolist()
        ep_fill = {
            "type": "scatter",
            "x": tl + tl[::-1],
            "y": upper + lower[::-1],
            "fill": "toself",
            "fillcolor": "rgba(79,142,247,0.15)",
            "line": {"color": "rgba(0,0,0,0)"},
            "showlegend": False,
            "hoverinfo": "skip",
        }
        ep_mean = {
            "type": "scatter",
            "x": tl,
            "y": mean_ep.tolist(),
            "mode": "lines",
            "line": {"color": "#4f8ef7", "width": 2.2},
            "name": "Mean SCR",
        }
        render(
            plotly_html(
                [ep_fill, ep_mean],
                {
                    "title": {
                        "text": f"Mean SCR waveform  (n={len(epochs)} epochs, ±1 SD)",
                        "font": {"size": 13},
                    },
                    "xaxis": {"title": "Time relative to event onset (s)"},
                    "yaxis": {"title": "EDA – baseline-corrected (µS)"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 0,
                            "x1": 0,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": "#e07b54", "width": 1.8, "dash": "dash"},
                        }
                    ],
                    "annotations": [
                        {
                            "x": 0.1,
                            "y": 0.97,
                            "xref": "x",
                            "yref": "paper",
                            "text": "Stimulus onset",
                            "showarrow": False,
                            "font": {"size": 10, "color": "#e07b54"},
                        }
                    ],
                },
                height=280,
            ),
            height=295,
        )
        st.caption(
            "Blue band = ±1 SD across trials. Orange dashed line = stimulus onset (t = 0). "
            "SCR onset ~1 s, peak ~2.5–3 s, recovery over ~6–8 s."
        )

        # ── Downsampling ──────────────────────────────────────
        st.subheader("4. Effect of downsampling (no anti-aliasing filter)")

        if ds_factor == 1:
            st.info(
                "Select a downsampling factor > 1 in the sidebar to explore aliasing."
            )
        else:
            new_fs = fs_ds
            new_nyq = new_fs / 2.0
            # Alias prediction for 50 Hz
            alias_50 = abs(50.0 - round(50.0 / new_fs) * new_fs)
            alias_100 = abs(100.0 - round(100.0 / new_fs) * new_fs)

            st.markdown(rf"""
**Factor {ds_factor}× decimation** → $f_s$ = **{new_fs:.1f} Hz**, Nyquist = **{new_nyq:.1f} Hz**.
Components above {new_nyq:.1f} Hz alias into $[0, {new_nyq:.1f}]$ Hz.

Predicted alias positions:
- 50 Hz → **{alias_50:.2f} Hz**
- 100 Hz → **{alias_100:.2f} Hz**
""")

            cd1, cd2 = st.columns(2)
            with cd1:
                # Downsampled time domain (first 5 s)
                n5 = min(int(5 * new_fs), len(sig_ds))
                ds_t_tr = {
                    "type": "scatter",
                    "x": time_ds[:n5].tolist(),
                    "y": sig_ds[:n5].tolist(),
                    "mode": "lines",
                    "line": {"color": "#f7a04f", "width": 1},
                }
                render(
                    plotly_html(
                        [ds_t_tr],
                        {
                            "title": {
                                "text": f"Downsampled signal ×{ds_factor}  (first 5 s)",
                                "font": {"size": 12},
                            },
                            "xaxis": {"title": "Time (s)"},
                            "yaxis": {"title": "EDA"},
                        },
                        height=230,
                    ),
                    height=245,
                )

            with cd2:
                # Downsampled spectrum
                alias_shapes = []
                for af in [alias_50, alias_100]:
                    if 0 < af < new_nyq:
                        alias_shapes.append(
                            {
                                "type": "line",
                                "x0": af,
                                "x1": af,
                                "y0": 0,
                                "y1": 1,
                                "yref": "paper",
                                "line": {
                                    "color": "#e74c3c",
                                    "width": 1.5,
                                    "dash": "dot",
                                },
                            }
                        )
                ds_sp_tr = {
                    "type": "scatter",
                    "x": freqs_d.tolist(),
                    "y": mag_d.tolist(),
                    "mode": "lines",
                    "fill": "tozeroy",
                    "line": {"color": "#e07b54", "width": 1.3},
                    "fillcolor": "rgba(224,123,84,0.13)",
                }
                render(
                    plotly_html(
                        [ds_sp_tr],
                        {
                            "title": {
                                "text": f"Spectrum after ×{ds_factor} downsampling",
                                "font": {"size": 12},
                            },
                            "xaxis": {"title": "Frequency (Hz)", "range": [0, new_nyq]},
                            "yaxis": {"title": "Amplitude"},
                            "shapes": alias_shapes,
                        },
                        height=230,
                    ),
                    height=245,
                )

            if new_nyq < 50.0:
                st.markdown(
                    f"""
<div class="warn-box">
⚠️ <b>Aliasing confirmed.</b>  The 50 Hz powerline component folds back to
<b>{alias_50:.1f} Hz</b> (red dotted line).  This falls directly inside the physiological
EDA band, making it indistinguishable from real signal without prior knowledge.
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    f"""New Nyquist = {new_nyq:.1f} Hz is still above 50 Hz — powerline noise is not yet aliased.
                    Try factor ≥ 8 (new Nyquist ≤ 16 Hz) to observe aliasing in the physiological band."""
                )

    # ──────────────────────────────────────────────────────────
    # TAB 4 – DISCUSSION
    # ──────────────────────────────────────────────────────────
    with tab4:
        st.header("Discussion")

        st.subheader("Signal identity: Electrodermal Activity (EDA / GSR)")
        st.markdown(rf"""
Multiple features collectively identify the recording as an EDA signal:

- **Amplitude** (~0.5–1.5 units, likely µS): consistent with skin conductance levels
  from finger/palm electrodes.
- **Temporal morphology**: slow tonic baseline punctuated by rapid deflections (SCRs) of
  ~6–8 s duration — the hallmark EDA waveform.
- **Spectral content**: power concentrated below 5 Hz; no fast oscillations typical of EEG,
  EMG, or ECG.
- **Event responses**: stimulus-locked deflections with ~1–3 s onset and ~2–3 s peak latency,
  matching published SCR normative data.
- **Paradigm structure**: {len(event_times)} events, jittered ITI ~{ieis.mean():.0f} s —
  standard psychophysiology design that prevents habituation and avoids spectral artefacts
  from periodic stimulation.
""")

        st.subheader("What are the events? — Stimulus hypothesis")
        st.markdown(rf"""
The jittered inter-trial interval (mean {ieis.mean():.1f} s, SD {ieis.std():.2f} s,
range {ieis.min():.1f}–{ieis.max():.1f} s) rules out a perfectly rhythmic design.  Jitter
is used deliberately in psychophysiology to:
1. prevent anticipatory responses,
2. allow full SCR recovery between trials, and
3. avoid a stimulus-frequency spectral artefact.

**Most likely paradigm:** a brief (~100 ms) **auditory tone burst** or **visual flash**
designed to measure the **orienting response** or **startle reflex** via EDA — the classic
EDA psychophysiology paradigm used in emotion, attention, and arousal research.

**Observed SCR characteristics confirm this:**

| Feature | Observed | Literature norm | Match? |
|---|---|---|---|
| Onset latency | ~1–2 s | 1–3 s | ✅ |
| Peak latency | ~2.5 s (median) | 1–5 s | ✅ |
| Response duration | ~6–8 s | 5–10 s | ✅ |
| Peak amplitude | ~0.13 µS (median) | 0.05–0.5 µS | ✅ |
| Response rate | 79 % | 60–90 % | ✅ |

The non-100 % response rate is normal: sympathetic hyporesponsiveness on some trials is
expected, especially in a long ({duration / 60:.0f}-min) session where habituation progressively
reduces SCR amplitude.
""")

        st.subheader("Noise sources and spectral signatures")
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            st.markdown("""
**50 Hz — Powerline interference (dominant noise)**

The sharpest and tallest non-physiological peak sits at exactly 50 Hz, the European mains
frequency.  Capacitive or inductive coupling from power cables to the recording leads
injects a near-perfect 50 Hz sinusoid.  Its extreme narrowness (< 1 bin wide) is diagnostic:
only an externally driven, highly stable oscillator produces such a thin spectral line.

**100 Hz — Second powerline harmonic**

Non-linearities in amplifier circuits or slightly asymmetric electrode contact generate even
harmonics.  At 100 Hz this is still below the 128 Hz Nyquist limit and is recorded faithfully.

**16.625 Hz — Sub-harmonic / hardware artefact**

The peak at 16.625 Hz ≈ 50/3 Hz is most likely an **intermodulation product** between the
50 Hz line noise and the ADC clock, or a beat frequency from imperfect analogue filtering in
the recording hardware.  It is not a physiological frequency.
""")
        with col_n2:
            st.markdown("""
**< 1 Hz — Tonic drift**

Very slow power reflects electrode polarisation (DC offset drifting over minutes), slow
changes in skin temperature, and accumulation of sweat under the electrode.  This overlaps
with the genuine tonic EDA component, making it difficult to separate without additional
physiological measures.

**Broadband floor — ADC & amplifier noise**

A flat baseline visible in the dB spectrum across all frequencies reflects thermal
(Johnson–Nyquist) noise of the measurement amplifier and quantisation noise of the
analogue-to-digital converter.  It is many orders of magnitude below the signal and has
no practical effect on analysis.

**Genuine signal (0.05–5 Hz)**

The physiological content — phasic SCRs peaking near 0.1–2 Hz, slow tonic fluctuations,
and possible ~0.3 Hz respiratory modulation — lives entirely in this band.  It is well
separated from the 50 Hz line noise, which is why EDA can be usefully recorded even in
electrically noisy environments.
""")

        st.subheader("Downsampling, aliasing, and information content")
        st.markdown(r"""
            **When does aliasing arise?**

            Aliasing occurs whenever the downsampled Nyquist frequency $f_{s,new}/2$ falls *below* any
            component with appreciable energy.  For this signal the critical threshold is when
            $f_{s,new}/2 < 50$ Hz, i.e. $f_{s,new} < 100$ Hz, i.e. **downsampling factor ≥ 3**.
            At factor 8 ($f_{s,new}$ = 32 Hz, $f_N$ = 16 Hz) the 50 Hz line noise aliases dramatically.

            **Predicting aliased frequencies:**

            $$f_\text{alias} = \left| f_\text{orig} - \text{round}\!\left(\frac{f_\text{orig}}{f_{s,\text{new}}}\right) \cdot f_{s,\text{new}} \right|$$

            At factor 8: $f_{alias,50} = |50 - 2 \times 32| = 14$ Hz — squarely inside the
            physiological band.  This contamination is invisible without knowledge of the original
            spectrum.

            **How to identify aliasing in a spectrum:**

            - A peak appears that cannot be explained by any known physiological process at that frequency.
            - The peak's frequency matches the alias prediction formula for a known noise source.
            - The peak disappears or moves when the downsampling factor is changed.
            - The peak is absent in a properly low-pass filtered version of the downsampled signal.

            **Effect on EDA information content:**

            The genuine EDA signal is entirely below 5 Hz, so even aggressive downsampling (factor 64×
            → $f_{s,new}$ = 4 Hz) preserves the physiological information in the *time domain*.  The
            SCR waveform shape, peak latency, and amplitude remain intact.  The damage is in the
            *frequency domain*: aliased noise corrupts the apparent spectral content, making clean
            spectral analysis impossible without prior filtering.

            **Key lesson:** Always apply a **low-pass anti-aliasing filter** at $f_{cutoff} \leq f_{s,new}/2$
            *before* downsampling.  For EDA, a 4th-order Butterworth at 5–10 Hz before decimating to
            32–64 Hz is standard practice.
            """)

        st.subheader("Expected vs. unexpected results")
        expected = {
            "Feature": [
                "Sampling rate",
                "Dominant noise source",
                "SCR onset latency",
                "SCR peak latency",
                "SCR duration",
                "Response rate",
                "16.625 Hz peak",
                "Power concentrated < 5 Hz",
            ],
            "Expected": [
                "128–512 Hz (typical EDA)",
                "50 Hz powerline (EU)",
                "1–3 s",
                "1–5 s",
                "5–10 s",
                "60–90 %",
                "Not typically present",
                "Yes",
            ],
            "Observed": [
                f"{fs:.0f} Hz",
                "50 Hz (largest peak)",
                "~1–2 s",
                "~2.5 s (median)",
                "~6–8 s",
                "79 %",
                "Present at 16.625 Hz",
                "✅ Confirmed",
            ],
            "Assessment": [
                "✅ Within normal range",
                "✅ As expected",
                "✅ Normal",
                "✅ Normal",
                "✅ Normal",
                "✅ Normal with habituation",
                "⚠️ Hardware artefact",
                "✅ As expected",
            ],
        }
        if HAS_PANDAS:
            st.dataframe(
                pd.DataFrame(expected), use_container_width=True, hide_index=True
            )
        else:
            for key in expected:
                st.write(f"**{key}:** {' | '.join(expected[key])}")

        st.info(
            """
            **Key take-away:**  Frequency-domain analysis reveals in seconds what time-domain
            inspection would struggle to quantify: the 50 Hz powerline contamination, an unexpected
            hardware artefact at 16.625 Hz, and a clean physiological band below 5 Hz.  The DFT is not
            only an analysis tool but a quality-control instrument for any biosignal recording pipeline.
            Any preprocessing step (especially downsampling) must be evaluated in both time and frequency
            domains to ensure physiological information is preserved and no artefactual structure is
            introduced.
            """
        )


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    main()
