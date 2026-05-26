"""
Exercise 1 - Time Domain Analysis
Analyzes F1 telemetry data (Bahrain GP 2024) for a selected pilot
across three telemetry variables: Speed, Throttle, Brake.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import zscore
from PIL import Image
import os

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Exercise 1 – Time Domain", page_icon="🏎️", layout="wide")

st.title("Exercise 1 · Time Domain Analysis")
st.caption("Bahrain Grand Prix 2024 · F1 Telemetry")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TS_PATH = os.path.join(DATA_DIR, "Bahrain_time_series.csv")
PILOTS_PATH = os.path.join(DATA_DIR, "pilotos.csv")
IMG_PATH = os.path.join(DATA_DIR, "bahrain-grand-prix.jpg")

VARIABLES = ["Speed", "Brake", "Throttle"]
VAR_UNITS = {"Speed": "km/h", "Brake": "binary / pressure", "Throttle": "%"}
VAR_COLORS = {"Speed": "#00D2FF", "Brake": "#FF3131", "Throttle": "#39FF14"}


@st.cache_data # To reduce loading time on reload 
def load_data():
    df = pd.read_csv(TS_PATH)
    # Normalise column names -- Remove spaces, tabs or newline
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data
def load_pilots():
    try:
        p = pd.read_csv(PILOTS_PATH)
        p.columns = [c.strip() for c in p.columns]
        return p
    except Exception:
        return None


def normalise_lap(group: pd.DataFrame) -> pd.DataFrame:
    """Resample a lap to 0-1 normalised position so laps of different time-durations
    can be overlaid on the same x-axis."""
    group = group.reset_index(drop=True)
    group["norm_pos"] = np.linspace(0, 1, len(group))
    return group


def detect_outlier_laps(
    lap_stats: pd.DataFrame, col: str = "Speed_mean", z_thresh: float = 2.5
):
    """Flag laps whose mean speed deviates more than z_thresh std-devs from the median."""
    if len(lap_stats) < 4:
        return []
    zs = np.abs(zscore(lap_stats[col]))
    return lap_stats.index[zs > z_thresh].tolist()

st.markdown("""
    <style>
    /* Expander container */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(0, 191, 255, 0.1);
        border-radius: 8px;
        overflow: hidden;
    }

    /* Expander header (the clickable bar) */
    div[data-testid="stExpander"] summary {
        background-color: rgba(0, 191, 255, 0.1);
        color: rgba(0, 80, 120, 1.0);
        font-weight: 500;
        padding: 12px 16px;
    }

    /* Expander header hover state */
    div[data-testid="stExpander"] summary:hover {
        background-color: rgba(0, 191, 255, 0.4);
    }

    /* Expander content area */
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        padding: 16px;
        border-top: 1px solid rgba(0, 191, 255, 0.1);
        color: rgba(0, 80, 120, 1.0);
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 1 · Introduction
# ─────────────────────────────────────────────
with st.expander("📖 1 · Introduction", expanded=True):
    st.markdown(
        """
        In Digital Signal Processing (DSP), signals are analyzed as functions of time.

        In this project, telemetry data (Speed, Throttle, Brake) is treated as time-domain signals.

        Each lap represents one realization of these signals. By comparing multiple laps,
        we can analyze patterns, consistency, and driving behavior.

        To enable comparison, laps are normalized to a common time axis [0,1]
        """
    )

# ─────────────────────────────────────────────
# 2 · Methods
# ─────────────────────────────────────────────
with st.expander("🔬 2 · Methods", expanded=False):
    st.markdown(
        """
        ### Approach

        This analysis investigates driver behaviour purely in the **time domain**, focusing on how telemetry signals evolve over a lap and across multiple laps.

        ---

        ### 1. Data preparation

        The full telemetry dataset is filtered to a single driver. We've kept the option to select other drivers as well, as part of trying to make it scalable.
        Laps with insufficient samples are removed to exclude incomplete or non-representative laps.

        ---

        ### 2. Lap normalisation

        Since laps differ in duration and sampling length, each lap is rescaled to a  
        **normalised position axis [0, 1]**, where:
        - 0 = start of lap  
        - 1 = end of lap  

        This allows direct comparison of signal behaviour across laps.

        The normalisation preserves signal shape but removes absolute timing information.

        ---

        ### 3. Signal alignment and resampling

        Each lap is interpolated onto a fixed grid of points using linear interpolation.  
        This ensures all laps have identical resolution and can be aggregated.

        ---

        ### 4. Statistical representation

        For each signal:
        - The **median** represents typical driver behaviour  
        - The **standard deviation** represents variability  
        - Individual laps are plotted transparently to show distribution  

        This combination provides both central tendency and spread in a single visualization.

        ---

        ### 5. Outlier detection

        Laps with abnormal behaviour are detected using a **z-score on mean speed**.  
        Laps exceeding a threshold (|z| > 2.5) are considered outliers and optionally excluded.

        These typically correspond to:
        - pit laps  
        - safety car periods  
        - abnormal driving conditions  

        ---

        ### 6. Consistency metric

        Consistency is quantified using the **coefficient of variation (CV)**:

        CV = (standard deviation / mean) × 100%

        Lower CV values indicate more consistent driving behaviour across laps.

        ---

        ### 7. Interpretation strategy

        The analysis focuses on identifying:
        - **Driving phases** (braking, cornering, acceleration)
        - **Event locations** (braking points, throttle application)
        - **Signal relationships** (e.g., brake → speed drop, throttle → speed increase)
        - **Consistency vs variability** across laps

        All insights are derived directly from time-domain behaviour without frequency analysis.
        """
    )

# ─────────────────────────────────────────────
# Data load
# ─────────────────────────────────────────────
df_raw = load_data()
pilots_df = load_pilots()

lap_col = "LapNumber"

# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    /* Main sidebar container */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 191, 255, 0.1);
        border-right: 1px solid rgba(0, 191, 255, 0.1);
    }

    /* Sidebar content wrapper */
    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] .stMarkdown p {
        color: rgba(0, 80, 120, 1.0);
    }

    /* Sidebar headers */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: rgba(0, 80, 120, 1.0);
    }

    /* Selectbox / widget labels */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stRadio label {
        color: rgba(0, 80, 120, 1.0);
        font-size: 0.85rem;
    }

    /* Selectbox dropdown box */
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(0, 191, 255, 0.1);
        border: 1px solid rgba(0, 191, 255, 0.1);
        border-radius: 6px;
    }

    /* Radio buttons */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        color: rgba(0, 80, 120, 1.0);
    }

    /* Divider */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(0, 191, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")

    # Pilot selector
    available_pilots = (
        sorted(df_raw["Driver"].dropna().unique()) if "Driver" in df_raw.columns else []
    )
    default_pilot = (
        "VER"
        if "VER" in available_pilots
        else (available_pilots[0] if available_pilots else None)
    )
    pilot = st.selectbox(
        "Pilot (Driver code)",
        available_pilots,
        index=available_pilots.index(default_pilot) if default_pilot else 0,
    )

    # Variable selector
    # Variable selectors (dropdowns)
    avail_vars = [v for v in VARIABLES if v in df_raw.columns]

    default_vars = [v for v in ["Speed", "Brake", "Throttle"] if v in avail_vars]
    while len(default_vars) < 3 and len(default_vars) < len(avail_vars):
        for v in avail_vars:
            if v not in default_vars:
                default_vars.append(v)
            if len(default_vars) == 3:
                break

    var1 = st.selectbox(
        "Telemetry variable 1",
        avail_vars,
        index=avail_vars.index(default_vars[0]),
        key="var1",
    )

    remaining_vars_2 = [v for v in avail_vars if v != var1]
    var2 = st.selectbox(
        "Telemetry variable 2",
        remaining_vars_2,
        index=remaining_vars_2.index(default_vars[1]) if default_vars[1] in remaining_vars_2 else 0,
        key="var2",
    )

    remaining_vars_3 = [v for v in avail_vars if v not in [var1, var2]]
    var3 = st.selectbox(
        "Telemetry variable 3",
        remaining_vars_3,
        index=remaining_vars_3.index(default_vars[2]) if default_vars[2] in remaining_vars_3 else 0,
        key="var3",
    )

    selected_vars = [var1, var2, var3]

    # Outlier exclusion
    exclude_outliers = st.checkbox("Exclude outlier laps", value=True)

    #change2i
    pilot_lap_sizes = df_raw[df_raw["Driver"] == pilot].groupby(lap_col).size()

    max_sample_size = int(pilot_lap_sizes.max())
    median_sample_size = int(pilot_lap_sizes.median())
    p10_sample_size = int(pilot_lap_sizes.quantile(0.10))

    min_samples = st.slider(
        "Min samples per lap (to include)",
        min_value=max(50, int(pilot_lap_sizes.min())),
        max_value=max_sample_size,
        value=p10_sample_size,
        step=25,
    )

    st.caption(
        f"Lap sample counts vary by lap. "
        f"Median: {median_sample_size}, 10th percentile: {p10_sample_size}, max: {max_sample_size}."
    )
    #change2e

    st.divider()
    st.caption("Data: Bahrain GP 2024 telemetry")

# ─────────────────────────────────────────────
# Filter data
# ─────────────────────────────────────────────
df_pilot = df_raw[df_raw["Driver"] == pilot].copy()

if df_pilot.empty:
    st.warning(f"No data found for pilot **{pilot}**.")
    st.stop()

# Filter short laps
lap_sizes = df_pilot.groupby(lap_col).size()
valid_laps = lap_sizes[lap_sizes >= min_samples].index
df_pilot = df_pilot[df_pilot[lap_col].isin(valid_laps)]
showOutlierInfo = False

# Outlier detection on Speed (if available)
if "Speed" in df_pilot.columns and exclude_outliers:
    lap_speed_mean = (
        df_pilot.groupby(lap_col)["Speed"].mean().rename("Speed_mean").reset_index()
    )
    outlier_laps = detect_outlier_laps(lap_speed_mean.set_index(lap_col))
    showOutlierInfo = outlier_laps
    df_pilot = df_pilot[~df_pilot[lap_col].isin(outlier_laps)]


# Normalise each lap
df_pilot = df_pilot.groupby(lap_col, group_keys=False).apply(normalise_lap)
laps = sorted(df_pilot[lap_col].unique())

if not selected_vars:
    st.warning("Please select at least one variable in the sidebar.")
    st.stop()

# ─────────────────────────────────────────────
# 3 · Results
# ─────────────────────────────────────────────
with st.expander(f"📊 3 · Results — {pilot} · {len(laps)} laps", expanded=True):
    if showOutlierInfo:
        st.info(
            f"🚩 Outlier laps detected and excluded: **{outlier_laps}**  \n(mean speed z-score > 2.5)"
        )
    tab_labels = [f"{v}" for v in selected_vars] + ["📈 Lap Statistics", "🔥 Heatmap", "🔗 Correlation", "🗺️ Track Heatmap"]
    tabs = st.tabs(tab_labels)

    N_BINS = 500  # normalised position bins for resampling


    def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
        """Convert a hex color string to an rgba() string for Plotly fillcolor."""
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"


    def resample_to_bins(group, var, n_bins=N_BINS):
        """Interpolate a lap signal onto a fixed grid of n_bins points."""
        x = group["norm_pos"].values
        y = group[var].values
        xi = np.linspace(0, 1, n_bins)
        return np.interp(xi, x, y)
    
    #change3i
    def find_track_columns(df: pd.DataFrame):
        """Try common X/Y column names used in telemetry datasets."""
        possible_x = ["X", "x", "PosX", "PositionX", "WorldPositionX"]
        possible_y = ["Y", "y", "PosY", "PositionY", "WorldPositionY"]

        x_col = next((c for c in possible_x if c in df.columns), None)
        y_col = next((c for c in possible_y if c in df.columns), None)

        return x_col, y_col

    #change3e

    xi_axis = np.linspace(0, 1, N_BINS)

    for i, var in enumerate(selected_vars):
        if var not in df_pilot.columns:
            with tabs[i]:
                st.warning(f"Column '{var}' not found in data.")
            continue

        color = VAR_COLORS.get(var, "#ffffff")
        unit = VAR_UNITS.get(var, "")

        # Build matrix [laps × bins]
        matrix = np.array(
            [resample_to_bins(df_pilot[df_pilot[lap_col] == lap], var) for lap in laps]
        )
        median_trace = np.median(matrix, axis=0)
        std_trace = np.std(matrix, axis=0)

        with tabs[i]:
            st.markdown(f"#### {var} ({unit}) — lap overlay")
            st.markdown(
                "Each thin line is one lap. The **bold line** is the median across all laps. "
                "The shaded band shows ±1 standard deviation."
            )

            fig = go.Figure()

            # Individual laps (semi-transparent)
            for j, lap in enumerate(laps):
                fig.add_trace(
                    go.Scatter(
                        x=xi_axis,
                        y=matrix[j],
                        mode="lines",
                        line=dict(color=color, width=0.7),
                        opacity=0.25,
                        name=f"Lap {lap}",
                        showlegend=False,
                        hovertemplate=f"Lap {lap}<br>pos=%{{x:.3f}}<br>{var}=%{{y:.1f}} {unit}<extra></extra>",
                    )
                )

            # Std band
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([xi_axis, xi_axis[::-1]]),
                    y=np.concatenate(
                        [median_trace + std_trace, (median_trace - std_trace)[::-1]]
                    ),
                    fill="toself",
                    fillcolor=hex_to_rgba(color, 0.12),
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                    name="±1 std",
                )
            )

            # Median
            fig.add_trace(
                go.Scatter(
                    x=xi_axis,
                    y=median_trace,
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=f"Median {var}",
                )
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Normalised lap position (0 = lap start, 1 = lap end)",
                yaxis_title=f"{var} [{unit}]",
                height=420,
                margin=dict(t=30, b=40),
                legend=dict(orientation="h", y=-0.18),
            )
            st.plotly_chart(fig, use_container_width=True)
            if var =="Speed":
                st.markdown(
                    """
                    **Interpretation:**  
                    The speed profile shows a clear cyclical pattern of acceleration and deceleration across the lap.  
                    High-speed peaks correspond to straights, while sharp drops indicate braking zones before corners.  
                    The tight clustering of lap traces around the median suggests high consistency in speed execution,  
                    with slightly increased variability in more technical sections of the track.
                    """
                )
            if var =="Brake":
                st.markdown(
                    """
                    **Interpretation:**  
                    The brake signal is highly event-based, with sharp spikes marking distinct braking zones.  
                    These occur at consistent positions across laps, indicating precise and repeatable braking points.  
                    The mostly binary nature of the signal suggests aggressive and decisive braking behaviour.
                    The very high Avg CV value highlights a limitation of standard statistical metrics when applied to sparse or binary signals.
                    """
                )
            if var == "Throttle":
                st.markdown(
                """
                **Interpretation:**  
                The throttle signal shows full application on straights and rapid reduction before braking zones.  
                Throttle is reapplied progressively after corners, indicating controlled acceleration.  
                Increased variability in some sections reflects differences in corner exits and grip conditions.
                """
            )
            

            # CV metric
            cv = (std_trace / (np.abs(median_trace) + 1e-6)) * 100
            mean_cv = np.nanmean(cv)
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean across lap", f"{np.nanmean(median_trace):.1f} {unit}")
            col2.metric("Std across lap", f"{np.nanmean(std_trace):.2f} {unit}")
            col3.metric(
                "Avg. CV (consistency)",
                f"{mean_cv:.1f}%",
                delta="lower = more consistent",
                delta_color="off",
            )

    # ── Lap statistics tab
    with tabs[len(selected_vars)]:
        st.markdown("#### Per-lap statistics over race distance")
        st.markdown(
            "This view shows how each variable's **mean per lap** evolves over the race, "
            "helping identify tyre degradation, fuel load effects, or driver adaptations."
        )

        fig2 = make_subplots(
            rows=len(selected_vars),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"{v} – mean per lap" for v in selected_vars],
            vertical_spacing=0.08,
        )

        for row_i, var in enumerate(selected_vars, start=1):
            if var not in df_pilot.columns:
                continue
            lap_stats = df_pilot.groupby(lap_col)[var].agg(["mean", "std"]).reset_index()
            color = VAR_COLORS.get(var, "#ffffff")

            # Std band
            fig2.add_trace(
                go.Scatter(
                    x=pd.concat([lap_stats[lap_col], lap_stats[lap_col].iloc[::-1]]),
                    y=pd.concat(
                        [
                            lap_stats["mean"] + lap_stats["std"],
                            (lap_stats["mean"] - lap_stats["std"]).iloc[::-1],
                        ]
                    ),
                    fill="toself",
                    fillcolor=hex_to_rgba(color, 0.13),
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_i,
                col=1,
            )

            fig2.add_trace(
                go.Scatter(
                    x=lap_stats[lap_col],
                    y=lap_stats["mean"],
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                    name=var,
                    hovertemplate=f"Lap %{{x}}<br>Mean {var}=%{{y:.1f}}<extra></extra>",
                ),
                row=row_i,
                col=1,
            )

            fig2.update_yaxes(title_text=VAR_UNITS.get(var, ""), row=row_i, col=1)

        fig2.update_xaxes(title_text="Lap number", row=len(selected_vars), col=1)
        fig2.update_layout(
            template="plotly_dark",
            height=200 * len(selected_vars) + 80,
            showlegend=False,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)


    # ── Heatmap tab
    with tabs[len(selected_vars) + 1]:
        st.markdown("#### Heatmap: lap x track position")
        st.markdown(
            "Each row is a lap; each column is a normalised position on the track. "
            "Bright spots reveal where on the track behaviour is most variable across laps."
        )

        hm_var = st.selectbox("Variable for heatmap", selected_vars, key="hm_var")

        if hm_var in df_pilot.columns:
            matrix_hm = np.array(
                [
                    resample_to_bins(df_pilot[df_pilot[lap_col] == lap], hm_var)
                    for lap in laps
                ]
            )

            fig3 = go.Figure(
                go.Heatmap(
                    z=matrix_hm,
                    x=xi_axis,
                    y=[str(l) for l in laps],
                    colorscale="Plasma",
                    colorbar=dict(title=VAR_UNITS.get(hm_var, "")),
                    hovertemplate="Lap %{y}<br>pos=%{x:.3f}<br>"
                    + hm_var
                    + "=%{z:.1f}<extra></extra>",
                )
            )
            fig3.update_layout(
                template="plotly_dark",
                xaxis_title="Normalised lap position",
                yaxis_title="Lap",
                height=max(400, 14 * len(laps)),
                margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig3, use_container_width=True)
    # ── Correlation tab
    with tabs[len(selected_vars) + 2]:
        st.markdown("#### Correlation between telemetry variables")

        # Use only selected variables that exist
        corr_vars = [v for v in selected_vars if v in df_pilot.columns]

        if len(corr_vars) < 2:
            st.warning("Select at least two variables to compute correlation.")
        else:
            # -------------------------
            # 1. Correlation Heatmap
            # -------------------------
            st.markdown("**Correlation matrix**")

            corr_df = df_pilot[corr_vars].corr(numeric_only=True)

            fig_corr = px.imshow(
                corr_df,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
            )

            fig_corr.update_layout(
                template="plotly_dark",
                height=400,
            )

            st.plotly_chart(fig_corr, use_container_width=True)

            # -------------------------
            # Correlation bar chart
            # -------------------------
            st.markdown("**Correlation strength (simple view)**")

            corr_df = df_pilot[corr_vars].corr(numeric_only=True)

            # pick reference variable
            ref_var = st.selectbox("Reference variable", corr_vars, key="ref_var")

            corr_values = corr_df[ref_var].drop(ref_var)

            fig_bar = px.bar(
                x=corr_values.index,
                y=corr_values.values,
                labels={"x": "Variable", "y": "Correlation"},
            )

            fig_bar.update_layout(
                template="plotly_dark",
                yaxis=dict(range=[-1, 1]),
            )

            st.plotly_chart(fig_bar, use_container_width=True)

            
            # -------------------------
            # 3. Interpretation
            # -------------------------
            st.markdown(
                """
                **Interpretation:**  
                The correlation matrix summarises linear relationships between telemetry variables.  
                Typically, throttle and speed show a positive correlation, while brake is negatively  
                correlated with both, reflecting alternating acceleration and deceleration phases.  

                The scatter plot provides a more detailed view, showing how these relationships  
                emerge from different driving phases such as braking, cornering, and acceleration.
                """
            )

    # ── Track Heatmap tab
    with tabs[len(selected_vars) + 3]:
        st.markdown("#### Track heatmap of telemetry variables")
        st.markdown(
            """
            This view maps telemetry directly onto the track using the X/Y coordinates.
            The driven path is colored by the selected variable, making it easy to identify
            braking zones, acceleration zones, and high-speed sections.
            """
        )

        x_col, y_col = find_track_columns(df_pilot)

        if x_col is None or y_col is None:
            st.warning(
                "No usable X/Y coordinate columns were found in the dataset. "
                "Check your column names (e.g. X, Y)."
            )
        else:
            track_var = st.selectbox(
                "Telemetry variable on track",
                [v for v in ["Speed", "Brake", "Throttle"] if v in df_pilot.columns],
                index=0,
                key="track_heatmap_var",
            )

            track_lap_mode = st.radio(
                "Track heatmap mode",
                ["All laps", "Single lap"],
                horizontal=True,
                key="track_heatmap_mode",
            )

            if track_lap_mode == "Single lap":
                selected_lap_track = st.selectbox("Lap", laps, key="track_heatmap_lap")
                plot_df = (
                    df_pilot[df_pilot[lap_col] == selected_lap_track]
                    .dropna(subset=[x_col, y_col, track_var])
                    .copy()
                )
                subtitle = f"{track_var} on track · Lap {selected_lap_track}"
            else:
                plot_df = df_pilot.dropna(subset=[x_col, y_col, track_var]).copy()
                subtitle = f"{track_var} on track · All laps"

            if plot_df.empty:
                st.warning("No valid X/Y telemetry data available for this selection.")
            else:
                fig_track_hm = go.Figure()

                # faint background track
                fig_track_hm.add_trace(
                    go.Scatter(
                        x=plot_df[x_col],
                        y=plot_df[y_col],
                        mode="markers",
                        marker=dict(
                            size=4,
                            color="rgba(180,180,180,0.15)",
                        ),
                        name="Track outline",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                # Define proper color ranges
                if track_var == "Brake":
                    cmin, cmax = 0, 1
                elif track_var == "Throttle":
                    cmin, cmax = 0, 100
                else:  # Speed, RPM, etc.
                    cmin, cmax = plot_df[track_var].min(), plot_df[track_var].max()

                # telemetry-colored overlay
                fig_track_hm = go.Figure()

                # Background track outline (all points, faint grey)
                outline_df = df_pilot.dropna(subset=[x_col, y_col]).copy()

                fig_track_hm.add_trace(
                    go.Scatter(
                        x=outline_df[x_col],
                        y=outline_df[y_col],
                        mode="markers",
                        marker=dict(
                            size=3,
                            color="rgba(180,180,180,0.10)",
                        ),
                        name="Track outline",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

                # Filter for brake events only
                if track_var == "Brake":
                    display_df = plot_df[plot_df[track_var] > 0.5].copy()
                else:
                    display_df = plot_df.copy()

                if display_df.empty:
                    st.warning("No active points available for this variable selection.")
                else:
                    # Variable-specific scale
                    if track_var == "Brake":
                        cmin, cmax = 0, 1
                        colorscale = "Reds"
                    elif track_var == "Throttle":
                        cmin, cmax = 0, 100
                        colorscale = "Plasma"
                    else:
                        cmin, cmax = display_df[track_var].min(), display_df[track_var].max()
                        colorscale = "Plasma"

                    fig_track_hm.add_trace(
                        go.Scatter(
                            x=display_df[x_col],
                            y=display_df[y_col],
                            mode="markers",
                            marker=dict(
                                size=6,
                                color=display_df[track_var],
                                colorscale=colorscale,
                                cmin=cmin,
                                cmax=cmax,
                                showscale=True,
                                colorbar=dict(title=track_var),
                                opacity=0.9,
                            ),
                            text=[
                                f"Lap: {lap}<br>{track_var}: {val:.1f}"
                                for lap, val in zip(display_df[lap_col], display_df[track_var])
                            ],
                            hovertemplate=f"{x_col}: %{{x:.1f}}<br>{y_col}: %{{y:.1f}}<br>%{{text}}<extra></extra>",
                            name=track_var,
                            showlegend=False,
                        )
                    )

                fig_track_hm.update_layout(
                    template="plotly_dark",
                    title=subtitle,
                    height=700,
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                    margin=dict(t=50, b=30, l=30, r=30),
                )

                st.plotly_chart(fig_track_hm, use_container_width=True)

                st.markdown(
                    """
                    **How to interpret this plot:**  
                    - **Speed:** highlights fast straights and slow corners  
                    - **Brake:** shows concentrated braking zones before turns  
                    - **Throttle:** shows where power is reapplied after corners and where the car remains fully committed
                    """
                )

                # -------------------------
                # 3. Interpretation
                # -------------------------
                st.markdown(
                    """
                    **Interpretation:**  
                    This plot shows how the relationship between two telemetry variables changes over the lap.  
                    Positive correlation means both increase together (e.g., throttle & speed on straights).  
                    Negative correlation indicates opposite behaviour (e.g., brake vs speed before corners).  
                    This helps identify braking zones, acceleration zones, and transition phases.
                    """
                )

# ─────────────────────────────────────────────
# 4 · Discussion
# ─────────────────────────────────────────────
with st.expander("💬 4 · Discussion", expanded=False):
    st.markdown(
        """
        ### Findings - Max Verstappen (VER), Bahrain GP 2024

        **Integrated Time-Domain Behaviour**

        The combined analysis of **Speed**, **Brake**, and **Throttle** reveals a highly structured and repeatable driving pattern across all laps.  
        The signals together form a clear cycle of:
        **full-throttle acceleration → throttle release → sharp braking → cornering → progressive throttle reapplication**.

        This cycle repeats consistently across the normalised lap and directly reflects the structure of the race track.

        The coefficient of variation becomes unstable for telemetry variables that spend part of the lap near zero, such as brake and throttle. 
        In these regions, even small absolute variability produces extremely large relative variability. 
        Therefore, CV is only interpreted for active signal regions, while brake is better described using event-based consistency measures.
        """
    )
    
    st.divider()
    
    st.markdown(
        """
        ### Relationship between signals

        The three telemetry signals are strongly interdependent:

        - **Brake vs Speed:**  
          Every braking spike corresponds to a sharp drop in speed, clearly marking corner entry points.

        - **Throttle vs Speed:**  
          Speed increases only after throttle is reapplied, confirming that acceleration phases are driven by controlled throttle input.

        - **Brake vs Throttle:**  
          The signals are largely non-overlapping, indicating clean transitions between braking and acceleration with minimal instability.

        This relationship allows clear identification of driving phases without needing track coordinates.

        ---

        ### Driver behaviour over a lap

        - **Straights:**  
          Characterised by near-maximum throttle, zero braking, and stable high speeds.  
          These sections show very low variability across laps.

        - **Braking zones:**  
          Identified by sudden brake spikes and rapid speed reduction.  
          Braking is sharp and binary, indicating an aggressive but controlled braking style.

        - **Cornering phases:**  
          Speed reaches local minima while throttle remains low or gradually increasing.  
          These sections show slightly higher variability, reflecting the complexity of corner execution.

        - **Corner exits:**  
          Throttle is reapplied progressively rather than instantly, suggesting careful traction management.

        ---

        ### Consistency across laps
        The analysis shows that Max Verstappen maintains highly consistent telemetry patterns across laps.

        - Speed profiles are nearly identical on straights
        - Throttle application is smooth and repeatable
        - Braking points are consistent, with slight variation in intensity

        This indicates a stable and optimized driving style.

        However, variability increases in certain technical sections of the track, where:
        - braking duration varies slightly
        - throttle reapplication differs across laps

        This indicates that while the **structure of the lap is consistent**, execution details adapt to conditions.

        ---

        ### Variability and outliers

        A small number of laps deviate significantly from the dominant pattern.  
        These likely correspond to:
        - pit laps
        - traffic influence
        - driver errors or corrections

        Such laps are automatically detected using a z-score filter on mean speed and excluded from analysis to preserve representative behaviour.

        ---

        ### Handling different lap lengths

        Laps are normalised to a [0, 1] axis to allow direct comparison despite differing durations.  
        This preserves the shape of each lap while enabling consistent overlay analysis.

        A limitation of this approach is that:
        - exact timing differences are not preserved
        - alignment is based on relative, not physical, track position

        ---

        ### Evaluation of the approach

        The time-domain overlay method proves highly effective in extracting:

        - driving phases (braking, cornering, acceleration)
        - consistency patterns
        - variability across laps

        Despite using only basic time-domain analysis, the method provides strong insight into driver behaviour.

        However, it is limited in explaining *why* variations occur.

        ---

        ### Potential improvements

        With more time or additional data, the analysis could be extended by:

        - Comparing fastest lap vs median behaviour
        - Segmenting the lap into corners and straights
        - Including tyre or stint information to explain variability

        ---
        
        ### Conclusion

        The analysis shows that Verstappen’s driving is highly structured, consistent, and efficient.  
        He combines aggressive braking with controlled throttle application, maintaining stable performance across the race while adapting slightly in more complex sections.
        """
    )

st.caption("DSP Exercise 1 · DAT25")
