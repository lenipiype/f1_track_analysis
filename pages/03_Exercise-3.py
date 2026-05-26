"""
Exercise 3: Short-Time Fourier Transform (STFT) Analysis
DSP Assignment – Hannah Wimmer

Place this file in your pages/ directory as:
    pages/03_Exercise-3_STFT_Audio.py

This page performs a fully manual STFT:
- framing + overlap
- windowing
- manual DFT per frame (no np.fft for the forward transform)
- spectrogram visualization
- optional band-pass masking in the STFT domain
- manual inverse STFT using manual IDFT + overlap-add

"""

import io
import json
from pathlib import Path

import numpy as np
import streamlit as st

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
try:
    st.set_page_config(
        page_title="Exercise 3 – STFT Audio Analysis",
        page_icon="🎵",
        layout="wide",
    )
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────
# Custom CSS
# Styled to fit the same warm academic layout as Exercise 2
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
div[data-testid="stAlertContainer"] {
    background: rgba(126, 48, 225, 0.08);
    border-left: 4px solid rgba(126, 48, 225, 0.9);
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin: 10px 0;
    font-size: 0.92em;
}

section[data-testid="stSidebar"] {
    background-color: rgba(255, 247, 209, 1);
    border-right: 1px solid rgba(255, 247, 209, 1);
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem;
}

hr {
    border-color: rgba(227, 106, 106, 1);
}

.stApp, .stAppHeader {
    background-color: rgba(255, 251, 241, 1);
}

p, span, div, h1, h2, h3, h4, h5 {
    color: rgba(114, 35, 35, 1);
}
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════
# DSP CORE – FULLY MANUAL STFT / ISTFT
# No np.fft used for the transform itself.
# ═══════════════════════════════════════════════════════════════

def dft_manual(x: np.ndarray) -> np.ndarray:
    """
    Discrete Fourier Transform from first principles.

        X[k] = sum_{n=0}^{N-1} x[n] * exp(-j 2πkn/N)

    Implemented in chunks for lower memory pressure.
    """
    N = len(x)
    n = np.arange(N)
    X = np.zeros(N, dtype=np.complex128)
    chunk = 256

    for k0 in range(0, N, chunk):
        k1 = min(k0 + chunk, N)
        ks = np.arange(k0, k1)
        W = np.exp(-2j * np.pi * np.outer(ks, n) / N)
        X[k0:k1] = W @ x

    return X


def idft_manual(X: np.ndarray) -> np.ndarray:
    """
    Inverse Discrete Fourier Transform from first principles.

        x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+j 2πkn/N)
    """
    N = len(X)
    k = np.arange(N)
    x = np.zeros(N, dtype=np.complex128)
    chunk = 256

    for n0 in range(0, N, chunk):
        n1 = min(n0 + chunk, N)
        ns = np.arange(n0, n1)
        W = np.exp(2j * np.pi * np.outer(ns, k) / N)
        x[n0:n1] = (W @ X) / N

    return x


def make_window(window_name: str, N: int) -> np.ndarray:
    if window_name == "hann":
        return np.hanning(N)
    if window_name == "hamming":
        return np.hamming(N)
    if window_name == "rectangular":
        return np.ones(N)
    raise ValueError(f"Unsupported window: {window_name}")


def stft_manual(
    x: np.ndarray,
    n_fft: int,
    hop_length: int,
    window_name: str = "hann",
    center: bool = False,
) -> np.ndarray:
    """
    Fully manual STFT:
    - frame the signal
    - apply analysis window
    - manual DFT per frame

    Returns complex matrix with shape [freq_bins, time_frames]
    where freq_bins = n_fft.
    """
    x = np.asarray(x, dtype=np.float64)

    if center:
        pad = n_fft // 2
        x = np.pad(x, (pad, pad), mode="constant")

    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)), mode="constant")

    window = make_window(window_name, n_fft)

    frames = []
    for start in range(0, len(x) - n_fft + 1, hop_length):
        frame = x[start:start + n_fft]
        frame_w = frame * window
        X = dft_manual(frame_w)
        frames.append(X)

    if not frames:
        return np.zeros((n_fft, 0), dtype=np.complex128)

    return np.array(frames, dtype=np.complex128).T


def istft_manual(
    D: np.ndarray,
    n_fft: int,
    hop_length: int,
    window_name: str = "hann",
    length: int | None = None,
    center: bool = False,
) -> np.ndarray:
    """
    Fully manual inverse STFT:
    - manual IDFT per frame
    - synthesis window
    - overlap-add
    - window-power normalization
    """
    window = make_window(window_name, n_fft)
    n_frames = D.shape[1]

    if n_frames == 0:
        return np.array([], dtype=np.float64)

    out_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)

    for i in range(n_frames):
        start = i * hop_length
        frame_t = idft_manual(D[:, i]).real
        frame_w = frame_t * window

        y[start:start + n_fft] += frame_w
        wsum[start:start + n_fft] += window**2

    valid = wsum > 1e-10
    y[valid] /= wsum[valid]

    if center:
        pad = n_fft // 2
        if len(y) > 2 * pad:
            y = y[pad:-pad]

    if length is not None:
        if len(y) >= length:
            y = y[:length]
        else:
            y = np.pad(y, (0, length - len(y)), mode="constant")

    return y


def magnitude_spectrum(D: np.ndarray) -> np.ndarray:
    return np.abs(D)


def phase_spectrum(D: np.ndarray) -> np.ndarray:
    return np.angle(D)


def amplitude_to_db(mag: np.ndarray, ref: float | None = None) -> np.ndarray:
    mag = np.maximum(mag, 1e-12)
    if ref is None:
        ref = np.max(mag)
    ref = max(ref, 1e-12)
    return 20.0 * np.log10(mag / ref)

def power_spectrum(D: np.ndarray) -> np.ndarray:
    return np.abs(D) ** 2


def power_to_db(power: np.ndarray, ref: float | None = None) -> np.ndarray:
    power = np.maximum(power, 1e-12)
    if ref is None:
        ref = np.max(power)
    ref = max(ref, 1e-12)
    return 10.0 * np.log10(power / ref)

def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 32) -> tuple[np.ndarray, np.ndarray]:
    fmin = 0.0
    fmax = sr / 2.0

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft * hz_points) / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))

    for m in range(1, n_mels + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        if center > left:
            fb[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)

        if right > center:
            fb[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)

    return fb, hz_points[1:-1]


def positive_freq_slice(n_fft: int):
    return slice(0, n_fft // 2 + 1)


def freq_axis(n_fft: int, sr: float) -> np.ndarray:
    return np.arange(0, n_fft // 2 + 1) * (sr / n_fft)


def time_axis(n_frames: int, hop_length: int, sr: float) -> np.ndarray:
    return np.arange(n_frames) * (hop_length / sr)


def apply_bandpass_mask(
    D: np.ndarray,
    sr: float,
    n_fft: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """
    Preserve frequencies in [low_hz, high_hz], zero everything else.
    Maintains conjugate symmetry so inverse transform remains real.
    """
    freqs = np.arange(n_fft) * (sr / n_fft)
    D2 = D.copy()

    keep_pos = (freqs >= low_hz) & (freqs <= high_hz)
    keep_neg = ((sr - freqs) >= low_hz) & ((sr - freqs) <= high_hz)

    keep = keep_pos | keep_neg
    D2[~keep, :] = 0
    return D2


def spectral_energy_by_frame(D: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(D) ** 2, axis=0)


def dominant_frequency_per_frame(D: np.ndarray, sr: float, n_fft: int) -> np.ndarray:
    Dpos = np.abs(D[positive_freq_slice(n_fft), :])
    freqs = freq_axis(n_fft, sr)
    idx = np.argmax(Dpos, axis=0)
    return freqs[idx]

def estimate_observed_frequency_span(power_pos: np.ndarray, freqs: np.ndarray, threshold_ratio: float = 0.05):
    mean_power = np.mean(power_pos, axis=1)
    threshold = threshold_ratio * np.max(mean_power)

    active = np.where(mean_power >= threshold)[0]
    if len(active) == 0:
        return 0.0, float(freqs[-1])

    return float(freqs[active[0]]), float(freqs[active[-1]])


def make_subbands(f_low: float, f_high: float, n_bands: int = 4):
    edges = np.linspace(f_low, f_high, n_bands + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n_bands)]


def band_energy_per_frame(power_pos: np.ndarray, freqs: np.ndarray, bands):
    energies = np.zeros((len(bands), power_pos.shape[1]), dtype=np.float64)

    for i, (f0, f1) in enumerate(bands):
        idx = np.where((freqs >= f0) & (freqs < f1))[0]
        if len(idx) > 0:
            energies[i, :] = np.sum(power_pos[idx, :], axis=0)

    return energies


def relative_band_energy_per_frame(power_pos: np.ndarray, freqs: np.ndarray, bands):
    band_energy = band_energy_per_frame(power_pos, freqs, bands)

    total_energy = np.sum(power_pos, axis=0, keepdims=True)
    total_energy = np.maximum(total_energy, 1e-12)

    rel_band_energy = band_energy / total_energy
    mean_rel_band_energy = np.mean(rel_band_energy, axis=1)

    return band_energy, rel_band_energy, mean_rel_band_energy

# ═══════════════════════════════════════════════════════════════
# AUDIO IO
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading audio …")
def load_audio_bytes(file_bytes: bytes):
    if not HAS_SF:
        raise ImportError("Missing dependency: soundfile. Install with `uv add soundfile`.")

    audio, sr = sf.read(io.BytesIO(file_bytes))

    # Convert stereo to mono if needed
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float64)

    # Normalize for safer playback / plotting if needed
    mx = np.max(np.abs(audio)) if len(audio) else 0.0
    if mx > 1.0:
        audio = audio / mx

    return audio, int(sr)


def wav_bytes_from_array(y: np.ndarray, sr: int) -> bytes:
    if not HAS_SF:
        raise ImportError("Missing dependency: soundfile. Install with `uv add soundfile`.")

    y = np.asarray(y, dtype=np.float64)
    mx = np.max(np.abs(y)) if len(y) else 0.0
    if mx > 1.0:
        y = y / mx

    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV")
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════
# PLOTLY INLINE HTML HELPERS
# Same pattern as Exercise 2
# ═══════════════════════════════════════════════════════════════

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"
_uid_counter = [0]

DARK = {
    "plot_bgcolor": "#0e1117",
    "paper_bgcolor": "#0e1117",
    "font": {"color": "#cccccc"},
    "xaxis": {"color": "#aaaaaa", "gridcolor": "#1e2a3a", "zerolinecolor": "#2a3a50"},
    "yaxis": {"color": "#aaaaaa", "gridcolor": "#1e2a3a", "zerolinecolor": "#2a3a50"},
    "margin": {"t": 45, "b": 45, "l": 65, "r": 20},
}


def _uid():
    _uid_counter[0] += 1
    return f"plt{_uid_counter[0]}"


def _merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def plotly_html(traces: list, layout_extra: dict, height: int = 300) -> str:
    uid = _uid()
    layout = _merge(DARK, layout_extra)
    layout["height"] = height
    return (
        f'<div id="{uid}" style="width:100%"></div>'
        f'<script src="{PLOTLY_CDN}"></script>'
        f"<script>Plotly.newPlot('{uid}', {json.dumps(traces)}, "
        f"{json.dumps(layout)}, {{responsive:true,displayModeBar:false}});</script>"
    )

def normalize_series(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    xmin = np.min(x)
    xmax = np.max(x)
    if xmax - xmin < 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def render(html: str, height: int = 300):
    st.components.v1.html(html, height=height, scrolling=False)


# ═══════════════════════════════════════════════════════════════
# PAGE
# ═══════════════════════════════════════════════════════════════

def main():
    st.title("Exercise 3 – Short-Time Fourier Transform (STFT)")
    st.caption("DSP / iDSP · Manual STFT on uploaded audio")

    if not HAS_SF:
        st.error("This page requires `soundfile`. Install with: `uv add soundfile`")
        st.stop()

    audio_dir = Path("data/exercise_03")
    supported_ext = (".wav", ".flac", ".ogg", ".mp3")
    available_audio_paths = sorted(
        [p for p in audio_dir.glob("*") if p.is_file() and p.suffix.lower() in supported_ext],
        key=lambda p: p.name.lower(),
    )

    if len(available_audio_paths) == 0:
        st.error("No audio files found in `data/exercise_03` (supported: WAV, FLAC, OGG, MP3).")
        st.stop()

    preferred_defaults = ["Stan.wav", "RealSlimShady.wav"]
    default_selection = [name for name in preferred_defaults if name in {p.name for p in available_audio_paths}]
    if not default_selection:
        default_selection = [p.name for p in available_audio_paths[:2]]

    selected_names = st.multiselect(
        "Select up to 2 audio files for comparison",
        options=[p.name for p in available_audio_paths],
        default=default_selection,
        max_selections=2,
    )

    if len(selected_names) == 0:
        st.info("Select at least one file.")
        st.stop()

    path_by_name = {p.name: p for p in available_audio_paths}
    selected_paths = [path_by_name[name] for name in selected_names if name in path_by_name]

    uploaded_items = []

    for file_path in selected_paths:
        file_bytes = file_path.read_bytes()

        if not file_bytes:
            st.warning(f"{file_path.name}: file could not be read.")
            continue

        try:
            x_tmp, sr_tmp = load_audio_bytes(file_bytes)
        except Exception as e:
            st.warning(f"{file_path.name}: could not be decoded ({e}).")
            continue

        uploaded_items.append({
            "name": file_path.name,
            "file_bytes": file_bytes,
            "duration": len(x_tmp) / sr_tmp,
        })

    if len(uploaded_items) == 0:
        st.error("None of the selected files could be decoded. Use WAV, FLAC, OGG, or MP3 files.")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Settings")

        n_fft = st.select_slider(
            "Window length N",
            options=[128, 256, 512, 1024, 2048],
            value=512,
            help="Longer windows improve frequency resolution but reduce time resolution.",
        )

        overlap = st.select_slider(
            "Overlap length",
            options=[0, 32, 64, 128, 256, 512, 1024],
            value=min(256, n_fft // 2),
        )

        if overlap >= n_fft:
            st.warning("Overlap must be smaller than the window length.")
            st.stop()

        hop_length = n_fft - overlap

        window_name = st.selectbox(
            "Window function",
            ["hann", "hamming", "rectangular"],
            index=0,
        )

        center = st.checkbox(
            "Center signal with zero-padding",
            value=False,
            help="Optional padding by N/2 samples on both sides.",
        )

        spec_mode = st.radio(
            "Spectrogram view",
            ["Magnitude", "Power", "Decibel", "Mel-scale"],
            index=2,
        )

        n_bands = st.select_slider(
            "Number of subbands",
            options=[3, 4],
            value=4,
        )

        n_mels = st.select_slider(
            "Number of Mel bands",
            options=[16, 24, 32, 40, 64],
            value=32,
        )


    tab1, tab2, tab3, tab4 = st.tabs(
        [  "Introduction"  ,   "Methods"  , "  Results"  , "  Discussion  "]
    )

    # ──────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────
    # TAB 1 – INTRODUCTION
    # ──────────────────────────────────────────────────────────
    with tab1:
        st.header("Introduction to the STFT")

        st.markdown(r"""
    The **Short-Time Fourier Transform (STFT)** extends Fourier analysis to non-stationary signals.

    A standard DFT tells us which frequencies exist in a signal, but it loses timing information.
    The STFT solves this by slicing the signal into short overlapping windows and computing a
    Fourier transform for each window separately. This produces a **time–frequency representation**
    showing how spectral content changes over time.

    Mathematically, for each frame \(m\), the signal is multiplied by a shifted window \(w[n-mH]\)
    and transformed:
    """)

        st.latex(r"X_m[k] = \sum_{n=0}^{N-1} x[n+mH]\,w[n]\,e^{-j2\pi kn/N}")

        st.markdown("""
    where:
    - N = window length
    - H = hop length
    - k = frequency bin
    - m = frame index

    The result is a complex matrix:
    - magnitude → spectral strength
    - phase → timing alignment needed for reconstruction
    """)

        st.markdown(r"""
    ### Why STFT matters for audio
    Audio is highly time-varying:
    - speech changes from phoneme to phoneme
    - music changes note by note
    - noise appears intermittently

    The STFT is therefore one of the most important DSP tools for:
    - spectrogram analysis
    - speech processing
    - denoising
    - source separation
    - feature extraction
    """)

    # ──────────────────────────────────────────────────────────
    # TAB 2 – METHODS
    # ──────────────────────────────────────────────────────────
    with tab2:
        st.header("Methods")
        uploaded_names = [item["name"] for item in uploaded_items]
        selected_recordings_md = "\n".join(f"- {name}" for name in uploaded_names) if uploaded_names else "- none uploaded"

        st.markdown(
            f"""
    This page compares **{len(uploaded_names)} music recordings** using a fully manual STFT implementation.
    The recordings currently selected are:
{selected_recordings_md}

    ### Selected recordings and motivation

    The recordings selected for this analysis are **"Stan"** and **"The Real Slim Shady"** by
    **Eminem**. These songs were chosen because they offer a useful controlled comparison within
    the same artist while still differing substantially in musical structure and spectral content.
    Using two songs by the same artist reduces variability caused by differences in vocal timbre,
    recording context, or artist-specific style. This makes it easier to attribute differences in
    the STFT results primarily to differences in **tempo, instrumentation, rhythmic complexity,
    and production style**.

    "Stan" is characterized by a slower tempo, a more repetitive and stable backing track, and
    strong vocal emphasis. From a time-frequency perspective, this makes it likely to show a more
    stable spectral structure and stronger concentration of energy in lower and mid-frequency bands.
    "The Real Slim Shady" is more rhythmically active and contains more pronounced transients and
    brighter percussive elements. As a result, it is expected to exhibit stronger temporal
    fluctuations in the spectrogram, a wider frequency spread, and more energy in higher-frequency
    bands.

    These two recordings are therefore well suited for comparing:
    - how spectral energy is distributed across subbands,
    - how the dominant frequency changes over time,
    - how dynamic spectral activity differs between recordings,
    - and how different STFT representations highlight those differences.

    For these reasons, this song pair provides a clear and interpretable basis for manual STFT
    analysis.

    ### Approach
    1. Each recording is loaded and converted to mono if needed.
    2. A user-selected **time segment** is extracted for fair comparison across recordings.
    3. The signal is divided into overlapping frames of length **N = {n_fft}** with hop **H = {hop_length}**.
    4. Each frame is windowed with the selected window function.
    5. A **manual DFT** is computed for every frame, producing the STFT.
    6. The STFT is visualized in multiple representations:
    - magnitude
    - power
    - dB scale
    - Mel scale
    7. Subband features are computed from the power spectrogram:
    - observed frequency span
    - relative band energy per frame
    - time-averaged relative band energy
    8. Additional metrics used:
    - spectral energy per frame
    - dominant frequency over time

    ### Why these metrics
    - **Subband energy** summarizes how spectral content is distributed across frequency regions.
    - **Dominant frequency** highlights which spectral region is strongest over time.
    - **Spectral energy** reflects how active or intense a recording segment is.
    """
        )

        st.markdown(r"""
    ### Time–frequency trade-off
    - **Short windows** improve time resolution but reduce frequency resolution.
    - **Long windows** improve frequency resolution but blur short events in time.

    Window choice also affects leakage:
    - Hann and Hamming reduce leakage compared with a rectangular window.
    """)
    # ──────────────────────────────────────────────────────────
    # TAB 3 – RESULTS
    # ──────────────────────────────────────────────────────────

    with tab3:
        st.header("Results")

        # ---- interactive time-segment selection (required by task sheet) ----
        min_duration = min(item["duration"] for item in uploaded_items)

        segment_duration = 10
        st.markdown(f"""Segment duration (seconds): {segment_duration}""")
        # st.slider(
        #     "Segment duration (seconds)",
        #     min_value=2.0,
        #     max_value=float(min_duration),
        #     value=min(5.0, float(min_duration)),
        #     step=0.5,
        #     help="Length of the analyzed excerpt from each song.",
        # )

        max_start = float(min_duration) - segment_duration

        if max_start <= 0:
            segment_start = 0.0
        else:
            segment_start = st.slider(
                "Segment start (seconds)",
                min_value=0.0,
                max_value=max_start,
                value=0.0,
                step=0.5,
                help="Choose which part of the recording to analyze.",
            )

        if hop_length > n_fft:
            st.warning("Hop length should usually be less than or equal to window length.")
            st.stop()


        results = []

        # ============================================================
        # PART A: COMPUTE ONLY
        # ============================================================
        for item in uploaded_items:
            file_bytes = item["file_bytes"]
            x, sr = load_audio_bytes(file_bytes)

            # keep a bounded excerpt, then apply interactive segment selection
            # extract the user-selected segment from the full recording
            start_sample = int(segment_start * sr)
            end_sample = int((segment_start + segment_duration) * sr)
            x_seg = x[start_sample:end_sample]

            if len(x_seg) < n_fft:
                st.warning(f"{item['name']}: selected segment is shorter than the window length.")
                continue

            with st.spinner(f"Running fully manual STFT for {item['name']} …"):
                D = stft_manual(
                    x=x_seg,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window_name=window_name,
                    center=center,
                )

            Dpos = D[positive_freq_slice(n_fft), :]
            mag = magnitude_spectrum(Dpos)
            power = power_spectrum(Dpos)
            db = amplitude_to_db(mag)
            power_db = power_to_db(power)

            freqs = freq_axis(n_fft, sr)
            times = time_axis(D.shape[1], hop_length, sr)

            mel_fb, mel_centers_hz = mel_filterbank(sr, n_fft, n_mels=n_mels)
            mel_power = mel_fb @ power
            mel_db = power_to_db(mel_power)

            f_low, f_high = estimate_observed_frequency_span(power, freqs, threshold_ratio=0.05)
            bands = make_subbands(f_low, f_high, n_bands=n_bands)
            band_energy, rel_band_energy, mean_rel_band_energy = relative_band_energy_per_frame(
                power, freqs, bands
            )

            frame_energy = spectral_energy_by_frame(Dpos)
            dom_freq = dominant_frequency_per_frame(D, sr, n_fft)

            duration_s = len(x_seg) / sr if sr > 0 else 0.0
            delta_f = sr / n_fft
            overlap_pct = 100.0 * (1.0 - hop_length / n_fft)
            segment_audio_bytes = wav_bytes_from_array(x_seg, sr)

            results.append({
                "display_name": item['name'].rsplit(".", 1)[0][:20],
                "name": item['name'],
                "file_bytes": file_bytes,
                "x": x_seg,
                "segment_audio_bytes": segment_audio_bytes,
                "sr": sr,
                "D": D,
                "Dpos": Dpos,
                "mag": mag,
                "power": power,
                "db": db,
                "power_db": power_db,
                "mel_db": mel_db,
                "mel_centers_hz": mel_centers_hz,
                "times": times,
                "freqs": freqs,
                "frame_energy": frame_energy,
                "dom_freq": dom_freq,
                "f_low": f_low,
                "f_high": f_high,
                "bands": bands,
                "rel_band_energy": rel_band_energy,
                "mean_band": mean_rel_band_energy,
                "duration_s": duration_s,
                "delta_f": delta_f,
                "overlap_pct": overlap_pct,
            })

        if len(results) == 0:
            st.info("No valid recordings could be analyzed with the current segment/window settings.")
            st.stop()

        # ============================================================
        # PART B: PLOT COMPARISONS ONLY
        # ============================================================

        st.subheader("Selected segment playback")

        playback_cols = st.columns(len(results))
        for col, res in zip(playback_cols, results):
            with col:
                st.write(f"**{res['name']}**")
                st.audio(res["segment_audio_bytes"], format="audio/wav")

        st.subheader("Spectrogram comparison")

        cols = st.columns(len(results))
        for col, res in zip(cols, results):
            with col:
                st.write(f"**{res['name']}**")

                if spec_mode == "Magnitude":
                    z_data = res["mag"]
                    y_data = res["freqs"]
                    title_txt = "Magnitude"
                    cbar = "Magnitude"
                elif spec_mode == "Power":
                    z_data = res["power"]
                    y_data = res["freqs"]
                    title_txt = "Power"
                    cbar = "Power"
                elif spec_mode == "Decibel":
                    z_data = res["power_db"]
                    y_data = res["freqs"]
                    title_txt = "dB"
                    cbar = "dB"
                else:
                    z_data = res["mel_db"]
                    y_data = res["mel_centers_hz"]
                    title_txt = "Mel-scale"
                    cbar = "Mel dB"

                spec_trace = {
                    "type": "heatmap",
                    "x": res["times"].tolist(),
                    "y": y_data.tolist(),
                    "z": z_data.tolist(),
                    "colorscale": "Plasma",
                    "colorbar": {"title": cbar},
                }

                render(
                    plotly_html(
                        [spec_trace],
                        {
                            "title": {"text": title_txt, "font": {"size": 12}},
                            "xaxis": {"title": "Time (s)"},
                            "yaxis": {"title": "Frequency (Hz)"},
                        },
                        height=320,
                    ),
                    height=340,
                )

        st.subheader("Metrics overview")

        metric_cols = st.columns(len(results))
        for col, res in zip(metric_cols, results):
            with col:
                st.write(f"**{res['name']}**")
                st.metric("Sample rate", f"{res['sr']} Hz")
                st.metric("Segment length", f"{res['duration_s']:.2f} s")
                st.metric("Δf", f"{res['delta_f']:.2f} Hz")
                st.metric("Overlap", f"{res['overlap_pct']:.1f}%")

        st.subheader("Additional STFT metrics")

        col_a, col_b = st.columns(2)

        with col_a:
            traces = []
            for res in results:
                traces.append({
                    "type": "scatter",
                    "x": res["times"].tolist(),
                    "y": normalize_series(res["frame_energy"]).tolist(),
                    "mode": "lines",
                    "name": res["name"],
                })

            render(
                plotly_html(
                    traces,
                    {
                        "title": {"text": "Spectral energy per frame", "font": {"size": 12}},
                        "xaxis": {"title": "Time (s)"},
                        "yaxis": {"title": "Normalized Energy"},
                        "legend": {
                            "orientation": "h",
                            "y": -0.35,
                            "x": 0,
                        },
                    },
                    height=320,
                ),
                height=340,
            )

        with col_b:
            traces = []
            for res in results:
                traces.append({
                    "type": "scatter",
                    "x": res["times"].tolist(),
                    "y": res["dom_freq"].tolist(),
                    "mode": "lines",
                    "name": res["name"],
                })

            render(
                plotly_html(
                    traces,
                    {
                        "title": {"text": "Dominant frequency over time", "font": {"size": 12}},
                        "xaxis": {"title": "Time (s)"},
                        "yaxis": {"title": "Frequency (Hz)"},
                        "legend": {
                            "orientation": "h",
                            "y": -0.35,
                            "x": 0,
                        },
                    },
                    height=320,
                ),
                height=340,
            )

        st.subheader("Relative subband energy comparison")

        span_lines = []
        for res in results:
            span_lines.append(f"- **{res['name']}**: {res['f_low']:.1f} Hz – {res['f_high']:.1f} Hz")
        st.markdown("Observed frequency span per recording:\n" + "\n".join(span_lines))

        band_labels = [f"Band {i+1}" for i in range(n_bands)]

        bar_traces = []
        for res in results:
            bar_traces.append({
                "type": "bar",
                "name": res["name"],
                "x": band_labels,
                "y": res["mean_band"].tolist(),
            })

        render(
            plotly_html(
                bar_traces,
                {
                    "title": {"text": "Time-averaged relative band energy", "font": {"size": 13}},
                    "xaxis": {"title": "Band"},
                    "yaxis": {"title": "Relative energy"},
                    "barmode": "group",
                    "legend": {
                        "orientation": "h",
                        "y": -0.35,
                        "x": 0,
                    },
                },
                height=350,
            ),
            height=365,
        )

        st.markdown("""
    ### Comparison interpretation
    - Differences in **low-frequency energy** indicate bass or kick dominance.
    - Differences in **mid-frequency bands** indicate vocals, guitars, synths, or snare presence.
    - Differences in **high-frequency bands** indicate brightness, cymbals, sharp transients, or noise.

    These comparisons help explain how songs differ in instrumentation, arrangement, and production style.
    """)

    # ──────────────────────────────────────────────────────────
    # TAB 4 – DISCUSSION
    # ──────────────────────────────────────────────────────────
    with tab4:
        st.header("Discussion")
        st.markdown(r"""
    This comparison shows how the STFT reveals differences between recordings in both
    their spectral content and their temporal structure.

    ### Differences across recordings
    Different songs often show different:
    - low-frequency energy distributions (bass and kick emphasis),
    - mid-frequency structure (vocals, guitars, synths),
    - high-frequency content (cymbals, brightness, transients),
    - temporal variation (steady textures vs. rapidly changing events).

    These differences arise from instrumentation, genre, arrangement, recording choices,
    and mixing/mastering style.

    ### Effect of STFT representation
    - **Magnitude / power** preserve raw spectral scale but can be hard to read because strong bins dominate.
    - **dB scale** is usually the most informative for visual inspection because it compresses dynamic range.
    - **Mel scale** is helpful when comparing perceptually meaningful energy distributions, especially for music/audio features.

    For this use case, the **dB representation** is the most useful for visual comparison,
    while **subband and Mel features** are useful for compact quantitative summaries.

    ### Time–frequency trade-off
    The STFT always balances time and frequency resolution:
    - shorter windows capture rapid changes better,
    - longer windows separate frequencies more clearly.

    This matters when choosing window length:
    - transient-heavy music benefits from shorter windows,
    - tonal or harmonic analysis benefits from longer windows.

    ### Features for a classifier
    If I had to train a model to distinguish recordings or genres, I would use:
    - relative subband energies,
    - Mel-band energies,
    - dominant frequency statistics,
    - spectral energy statistics over time.

    These features provide a compact but informative summary of both spectral balance and temporal variation.
    """)


if __name__ == "__main__":
    main()
else:
    main()