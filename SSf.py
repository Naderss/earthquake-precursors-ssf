# =============================================================================
# Shakibay Senobari's frequency-domain (SSf) PIPELINE (Turkey_2023 earthquake example): Download → Frequency Features → Accumulate → Plot
# =============================================================================
# U.S. Patent Pending: Application No. 63/870,020, “Methods and Systems for 
# Detecting Earthquake Precursors via Stress-Sensitive Transformations of Seismic Noise”, 
# filed on August 25, 2025. The patent was filed by and is owned by the author, Nader Shakibay Senobari.
# =============================================================================

import os, glob, numpy as np
from scipy.signal import welch
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib import ticker, rcParams
from scipy.stats import zscore
import pandas as pd
from collections import defaultdict
from datetime import datetime

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# =========================
# CONFIG
# =========================
NETWORK      = "KO"
STATIONS     = ["GAZ", "KMRS", "DARE", "URFA"]
LOCATION     = "*"
CHANNEL      = "HH*"
CLIENT       = Client("KOERI")

STARTTIME    = UTCDateTime("2023-02-05T00:00:00")  # inclusive
NUM_DAYS     = 2
STEP_HOURS   = 1                                   # process per hour

SR_TARGET    = 100     # Hz after resample (if needed)
WINDOW_SEC   = 120     # 2-minute windows
FREQ_MIN     = 0.25    # Hz
FREQ_MAX     = 15.25   # Hz
NUM_BINS     = 150

BASE_DIR     = "./Turkey_2023/"
VISUALIZE    = False
OVERWRITE    = False

# Plot config / event times for the Turkey–Syria sequence
PLOT_SAVE    = "2023_Turkey–Syria_earthquakes_vn.pdf"
EQ1_TIME     = datetime(2023, 2, 6, 1, 16, 0)   # Mw 7.8 timing rounded to even minutes 
EQ2_TIME     = datetime(2023, 2, 6, 10, 24, 0)  # Mw 7.5
EQ3_TIME     = None
EQ4_TIME     = None

# =========================
# Derived constants
# =========================
FREQ_BINS     = np.linspace(FREQ_MIN, FREQ_MAX, NUM_BINS + 1)
FREQ_CENTERS  = 0.5 * (FREQ_BINS[:-1] + FREQ_BINS[1:])
CHUNK_SECONDS = int(STEP_HOURS * 3600)


# =============================================================================
# 1) DOWNLOAD & AMPLITUDE SPECTRAL DENSITY COMPUTATIONS
# =============================================================================
def out_dir(station: str) -> str:
    d = f"{BASE_DIR}{NETWORK}-{station}-{CHANNEL[0:2]}-features/"
    os.makedirs(d, exist_ok=True)
    return d

def out_path(dir_data: str, channel_tail: str, t0: UTCDateTime) -> str:
    return f"{dir_data}{channel_tail}-{t0.date}-{t0.hour}.mat"

def fetch_stream(station: str, t0: UTCDateTime, t1: UTCDateTime):
    st = CLIENT.get_waveforms(NETWORK, station, LOCATION, CHANNEL, t0, t1)
    inv = CLIENT.get_stations(
        network=NETWORK, station=station, location=LOCATION, channel=CHANNEL,
        starttime=t0, endtime=t1, level="response"
    )
    return st, inv

def preprocess(st, inv):
    if len(st) == 0:
        return st
    st.remove_sensitivity(inv)
    st.sort(keys=["starttime"])
    st.merge(method=0, fill_value="latest")
    for tr in st:
        if abs(tr.stats.sampling_rate - SR_TARGET) > 1e-6:
            tr.resample(SR_TARGET)
    return st

def compute_hourly_spectra(tr):
    """
    Split trace into non-overlapping WINDOW_SEC segments.
    Return array [N_windows x (1 + NUM_BINS)]: [unix_end, bin1..binN]
    """
    sr = tr.stats.sampling_rate
    npts = tr.stats.npts
    t0 = tr.stats.starttime
    data = tr.data.astype(np.float64, copy=False)

    samples_per_win = int(WINDOW_SEC * sr)
    num_windows = npts // samples_per_win
    if num_windows <= 0:
        return np.empty((0, 1 + NUM_BINS))

    rows = []
    for i in range(num_windows):
        s = i * samples_per_win
        e = s + samples_per_win
        seg = data[s:e]
        if len(seg) < samples_per_win:
            continue

        f, Pxx = welch(seg, fs=sr, nperseg=min(1024, len(seg)))
        valid = (f >= FREQ_MIN) & (f <= FREQ_MAX)
        f = f[valid]
        A = np.sqrt(Pxx[valid])  # amplitude spectrum

        avg_amp = np.zeros(NUM_BINS, dtype=float)
        for j in range(NUM_BINS):
            idx = np.where((f >= FREQ_BINS[j]) & (f < FREQ_BINS[j + 1]))[0]
            avg_amp[j] = float(np.mean(A[idx])) if idx.size else 0.0

        t_end = (t0 + (i + 1) * WINDOW_SEC).timestamp
        rows.append([t_end] + avg_amp.tolist())

    return np.asarray(rows)

def run_download_and_features():
    for station in STATIONS:
        dir_data = out_dir(station)
        t0 = STARTTIME
        t1 = STARTTIME + CHUNK_SECONDS

        for _ in range(0, 24 * NUM_DAYS, STEP_HOURS):
            try:
                print(f"[{station}] Fetching {t0} → {t1}")
                st, inv = fetch_stream(station, t0, t1)
                st = preprocess(st, inv)

                if len(st) == 0:
                    print(f"[{station}] No data for {t0}")
                    t0, t1 = t1, t1 + CHUNK_SECONDS
                    continue

                for tr in st:
                    tail = tr.id.replace(".", "_")[-3:]  # e.g., HHZ
                    out_file = out_path(dir_data, tail, t0)
                    if (not OVERWRITE) and os.path.exists(out_file):
                        print(f"  ↪ Skipping existing: {out_file}")
                        continue

                    arr = compute_hourly_spectra(tr)
                    savemat(out_file, {
                        f"features_{tail}": arr,
                        "freq_centers": FREQ_CENTERS,
                        "time_unix": arr[:, 0] if arr.size else np.array([]),
                    })
                    print(f"  ✓ Saved: {out_file}")

                    if VISUALIZE and arr.size:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        cax = ax.imshow(
                            arr[:, 1:1+NUM_BINS].T,
                            aspect="auto", origin="lower",
                            extent=[0, arr.shape[0], FREQ_CENTERS[0], FREQ_CENTERS[-1]],
                            cmap="viridis"
                        )
                        ax.set_title(f"Spectral Amplitude - {tr.id}")
                        ax.set_xlabel("Window index (2-min steps)")
                        ax.set_ylabel("Frequency (Hz)")
                        fig.colorbar(cax, label="Avg. amplitude")
                        plt.tight_layout()
                        plt.savefig(f"{dir_data}spectral_avg_{tail}.png", dpi=300)
                        plt.close(fig)

            except Exception as e:
                print(f"[{station}] Error at {t0}: {e}")

            t0, t1 = t1, t1 + CHUNK_SECONDS


# =============================================================================
# 2) ACCUMULATE (from MAT); SAVE/LOAD NPZ with META
# =============================================================================
def _extract_time_from_filename(filename):
    """Expect ...-YYYY-MM-DD-HH.mat"""
    try:
        base = os.path.basename(filename)
        parts = base.split('-')
        date_part = parts[-2]
        hour_part = parts[-1].split('.')[0]
        return f"{date_part} {hour_part}:00:00"
    except Exception:
        return None

def _key_for(station, channel): return f"{station}_{channel}"

def load_accumulated_data(base_dir=BASE_DIR):
    accumulated = defaultdict(lambda: defaultdict(list))
    meta = {}

    station_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.endswith('-features')
    ]
    print("📂 Folders:", station_dirs)

    for station_dir in station_dirs:
        full_path = os.path.join(base_dir, station_dir)
        parts = station_dir.split('-')
        if len(parts) < 3:
            continue
        _, station, _ = parts[:3]

        mat_files = glob.glob(os.path.join(full_path, '*.mat'))
        mat_files.sort(key=lambda f: _extract_time_from_filename(f) or f)

        for mat_file in mat_files:
            try:
                mat = loadmat(mat_file)
                feature_keys = [k for k in mat.keys() if k.startswith('features_')]
                if not feature_keys:
                    continue

                key = feature_keys[0]
                channel = key.replace('features_', '')

                data = np.asarray(mat[key])
                if data.size == 0 or data.shape[0] == 0:
                    continue
                data = data[np.argsort(data[:, 0])]

                accumulated[station][channel].append(data)

                fc = np.asarray(mat.get('freq_centers', [])).ravel()
                if fc.size:
                    meta[_key_for(station, channel)] = {"freq_centers": fc}

            except Exception as e:
                print(f"Error reading {mat_file}: {e}")

    for st in accumulated:
        for ch in accumulated[st]:
            combined = np.vstack(accumulated[st][ch])
            combined = combined[np.argsort(combined[:, 0])]
            accumulated[st][ch] = combined
            print(f"✅ Accumulated: {st}-{ch}, shape={combined.shape}")

    return accumulated, meta

def save_accumulated_data_npz(accumulated, meta, filename='accumulated_Turkey_2023.npz'):
    save_dict = {}
    for st, ch_dict in accumulated.items():
        for ch, arr in ch_dict.items():
            save_dict[_key_for(st, ch)] = arr
    for key, md in meta.items():
        if "freq_centers" in md:
            save_dict[f"{key}__freq_centers"] = md["freq_centers"]
    np.savez_compressed(filename, **save_dict)
    print(f"Saved accumulated data to {filename}")

def load_accumulated_data_npz(filename='accumulated_Turkey_2023.npz'):
    loaded = np.load(filename, allow_pickle=True)
    accumulated = defaultdict(dict)
    meta = {}
    for key in loaded.files:
        if key.endswith("__freq_centers"):
            base = key.replace("__freq_centers", "")
            meta[base] = {"freq_centers": loaded[key]}
        else:
            station, channel = key.split('_', 1)
            accumulated[station][channel] = loaded[key]
            print(f"Loaded: {station}-{channel}, shape={loaded[key].shape}")
    return accumulated, meta


# =============================================================================
# 3) PLOT SS_f (with true Hz labels from meta)
# =============================================================================
def plot_ssf_traces(
    accumulated_data, meta, *,
    Eq1_time: datetime,
    Eq2_time: datetime | None = None,
    Eq3_time: datetime | None = None,
    Eq4_time: datetime | None = None,
    stations: list[str] | None = None,
    channel_selector: dict[str, str] | None = None,   # e.g., {"URFA": "HHN"}
    freq_indices: range = range(6, 13),
    diff_steps: tuple[int, ...] = (1, 2),
    env_med_win: int = 10,
    env_mean_win: int = 30,
    ssf_med_win: int = 30,
    ssf_mean_win: int = 60,
    figsize=(10, 10),
    tick_interval_days: int = 7,
    y_offset_step: float = 5.0,
    ylim: tuple[float, float] | None = (-24, 18),
    xlim_days: tuple[float, float] | None = (-100, 10),
    title: str = "2023 Turkey–Syria earthquakes",
    save_path: str | None = PLOT_SAVE,
    show: bool = True,
):
    """Plot z-scored SS_f curves using real frequency labels in meta[key]['freq_centers']."""
    if Eq1_time is None:
        raise ValueError("Eq1_time (datetime) must be provided.")

    rcParams.update({
        "font.size": 12, "font.family": "serif",
        "axes.labelsize": 14, "axes.titlesize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "lines.linewidth": 1.5, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linestyle": "--"
    })

    station_keys = list(accumulated_data.keys()) if stations is None else list(stations)
    num_stations = len(station_keys)
    fig, axes = plt.subplots(num_stations, 1, figsize=figsize, sharex=True)
    if num_stations == 1:
        axes = [axes]

    def meta_key(station: str, channel: str) -> str:
        return f"{station}_{channel}"

    def choose_channel(station: str) -> str:
        if channel_selector and station in channel_selector:
            return channel_selector[station]
        ch_keys = list(accumulated_data[station].keys())
        if not ch_keys:
            raise ValueError(f"No channels for station {station}")
        return ch_keys[0]

    # global x-range (days since Eq1)
    global_min_day, global_max_day = float("inf"), float("-inf")

    for st in station_keys:
        ch = choose_channel(st)
        arr = accumulated_data[st][ch]
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        
        dtime = pd.to_datetime(arr[:, 0], unit="s", utc=True).to_pydatetime()
        x_days = (np.array([t.timestamp() for t in dtime]) - Eq1_time.timestamp()) / (3600 * 24)
        if np.isfinite(np.nanmin(x_days)): global_min_day = min(global_min_day, np.floor(np.nanmin(x_days)))
        if np.isfinite(np.nanmax(x_days)): global_max_day = max(global_max_day, np.ceil(np.nanmax(x_days)))

    for ax, st in zip(axes, station_keys):
        ch = choose_channel(st)
        arr = accumulated_data[st][ch]
        arr = arr[np.argsort(arr[:, 0])]
        dtime = pd.to_datetime(arr[:, 0], unit="s")
        x_days = (dtime - Eq1_time).total_seconds() / (3600 * 24)
        X = arr[:, 1:]
        num_bins = X.shape[1]

        # freq centers (Hz) from meta
        fc = None
        mk = meta_key(st, ch)
        if mk in meta and isinstance(meta[mk], dict) and "freq_centers" in meta[mk]:
            fc = np.asarray(meta[mk]["freq_centers"]).ravel()
        if fc is None or fc.size != num_bins:
            fc = np.arange(num_bins)

        for j in freq_indices:
            if j >= num_bins:
                continue
            curves = []
            for k in diff_steps:
                if j + k >= num_bins:
                    continue
                tm = np.log10(X[:, j] + 1e-20) - np.log10(X[:, j + k] + 1e-20)
                tm -= np.nanmin(tm)
                curves.append(tm)
            if not curves:
                continue

            # (optional) context envelope, computed but not plotted
            env_stack = []
            for i in range(1, min(3, num_bins)):
                env_stack.append(X[:, i])
            if env_stack:
                env_stack = np.stack(env_stack, axis=1)
                env_stack = 10 * (np.log10(env_stack / (np.nanmedian(env_stack, axis=0) + 1e-20)))
                env_sum = pd.Series(np.nansum(env_stack, axis=1))
                env_sum = env_sum.rolling(env_med_win, min_periods=1).median().rolling(env_mean_win, min_periods=1).mean()
                _ = zscore(env_sum.bfill())

            ssf_sum = pd.Series(np.nansum(np.stack(curves, axis=1), axis=1))
            ssf_med = ssf_sum.rolling(ssf_med_win, min_periods=1).median()
            ssf_smooth = ssf_med.rolling(ssf_mean_win, min_periods=1).mean()
            ssf_z = zscore(ssf_smooth.bfill())

            y = ssf_z - j * y_offset_step + 40.0
            yen=env_sum - j * y_offset_step + 40.0
            label_hz = f"{np.round(fc[j] * 10) / 10} Hz" if fc.ndim == 1 else f"bin {j}"
            ax.plot(x_days[ssf_mean_win::], y[ssf_mean_win::], linewidth=1.2, label=label_hz)

        # Event markers
        ax.axvline(0.0, color="k", linestyle="-", linewidth=1.1, label="M$_{w}$ 7.8")
        if Eq2_time:
            ax.axvline((Eq2_time - Eq1_time).total_seconds() / (3600 * 24), color="r", linestyle="--", linewidth=1.0)
        if Eq3_time:
            ax.axvline((Eq3_time - Eq1_time).total_seconds() / (3600 * 24), color="b", linestyle="--", linewidth=1.0)
        if Eq4_time:
            ax.axvline((Eq4_time - Eq1_time).total_seconds() / (3600 * 24), color="k", linestyle="--", linewidth=1.0)

        ax.set_ylabel(f"Z-scored SS$_{{f}}$ (offset)\n{st}-{ch}", fontsize=12)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval_days))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))
        ax.grid(True, which="major", axis="both", linestyle="-", alpha=0.5)
        ax.grid(True, which="minor", axis="both", linestyle="--", alpha=0.3)
        ax.set_yticklabels([])
        ax.legend(loc="upper left", ncol=6, fontsize=8, frameon=True)

    # limits & labels
    if xlim_days is not None:
        axes[0].set_xlim(*xlim_days)
    else:
        axes[0].set_xlim(global_min_day, global_max_day)
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(*ylim)

    axes[-1].set_xlabel(r"Days since the main shock (M$_{w}$ 7.8)")
    axes[0].set_title(title)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # 1) Download & compute hourly amplitude spectral densities using Welch method (comment out after first run)
    #run_download_and_features()

    # 2) Accumulate from hourly MATs → in-memory dicts (comment out after first run)
    #accumulated, meta = load_accumulated_data(BASE_DIR)
    # Optionally cache as NPZ for quick reload:
    #save_accumulated_data_npz(accumulated, meta, 'accumulated_Turkey_2023.npz')

    # 3) (Re)load from NPZ later:
    #accumulated, meta = load_accumulated_data_npz('accumulated_Turkey_2023.npz')

    # 4) Plot SS_f traces (Turkey–Syria example)
    plot_ssf_traces(
        accumulated, meta,
        Eq1_time=EQ1_TIME, Eq2_time=EQ2_TIME, 
        channel_selector={"URFA": "HHN", "KMRS": "HHZ", "GAZ": "HHZ", "DARE": "HHZ"},
        xlim_days=(-100, 5),
        ylim = None,
        ssf_med_win=30,
        ssf_mean_win=60*1,
        title="2023 Turkey–Syria earthquakes",
        save_path=PLOT_SAVE,
    )
