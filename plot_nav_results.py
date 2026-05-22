import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import glob

test_results_base = "/home/datvu/LeggedGym-Ex/test_results"

algos = ["DWA", "MPPI"]
states = ["Bật", "Tắt"]
colors = {"Bật": "#27AE60", "Tắt": "#E67E22"}


def smooth(series, window=12):
    if len(series) < window:
        return series
    return series.rolling(window=window, min_periods=1).mean()


# Tìm tất cả các thư mục mean_data trong test_results
print(f"Scanning for mean_data folders in {test_results_base}...", flush=True)
mean_data_dirs = glob.glob(
    os.path.join(test_results_base, "**", "mean_data"), recursive=True
)

for data_dir in mean_data_dirs:
    map_name = os.path.basename(os.path.dirname(data_dir))
    print(f"Processing map: {map_name} in {data_dir}", flush=True)

    for algo in algos:
        files = {}
        for state in states:
            f = os.path.join(data_dir, f"timeseries_{algo}_{map_name}_SF_{state}.csv")
            if os.path.exists(f):
                files[state] = pd.read_csv(f)

        if not files:
            continue

        print(f"  -> Generating plots for {algo} in {map_name}...", flush=True)
        plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})

        # --- FIG 1: VELOCITY METRICS ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
        v_metrics = [
            ("vx", "Linear Velocity $v_x$ (m/s)"),
            ("vy", "Lateral Velocity $v_y$ (m/s)"),
            ("vyaw", "Angular Velocity $\omega_{yaw}$ (rad/s)"),
        ]

        for idx, (key, ylabel) in enumerate(v_metrics):
            ax = axes[idx]
            for state, df in files.items():
                label = f"SF {'On' if state=='Bật' else 'Off'}"
                mean_col = f"mean_{key}"
                if mean_col not in df.columns:
                    continue

                val = df[mean_col]
                if key in ["vy", "vyaw"]:
                    val = val.abs()

                # Plot raw data (transparent)
                ax.plot(
                    df["time_bin"].values,
                    val.values,
                    color=colors[state],
                    alpha=0.15,
                    linewidth=1.0,
                )
                # Plot smoothed data
                ax.plot(
                    df["time_bin"].values,
                    smooth(val).values,
                    label=label,
                    color=colors[state],
                    linewidth=1.0,
                )

            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if idx == 0:
                ax.set_title(f"Velocity Profile Analysis ({algo})", fontsize=14, pad=20)
                ax.legend(frameon=False, loc="upper right", ncol=2)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(data_dir, f"plot_velocities_stacked_{algo}.png"),
            dpi=400,
            bbox_inches="tight",
        )  # High DPI for paper figures
        plt.savefig(
            os.path.join(data_dir, f"plot_velocities_stacked_{algo}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(data_dir, f"plot_velocities_stacked_{algo}.svg"),
            format="svg",
            bbox_inches="tight",
        )
        plt.close()

        # --- FIG 2: STABILITY METRICS ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        s_metrics = [("roll", "Roll Angle (rad)"), ("pitch", "Pitch Angle (rad)")]

        for idx, (key, ylabel) in enumerate(s_metrics):
            ax = axes[idx]
            for state, df in files.items():
                label = f"SF {'On' if state=='Bật' else 'Off'}"
                mean_col = f"mean_{key}"
                if mean_col not in df.columns:
                    continue

                # Plot raw data (transparent)
                ax.plot(
                    df["time_bin"].values,
                    df[mean_col].abs().values,
                    color=colors[state],
                    alpha=0.15,
                    linewidth=1.0,
                )
                # Plot smoothed data
                ax.plot(
                    df["time_bin"].values,
                    smooth(df[mean_col].abs()).values,
                    label=label,
                    color=colors[state],
                    linewidth=1.0,
                )

            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if idx == 0:
                ax.set_title(f"Body Stability Analysis ({algo})", fontsize=14, pad=20)
                ax.legend(frameon=False, loc="upper right", ncol=2)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(data_dir, f"plot_stability_stacked_{algo}.png"),
            dpi=400,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(data_dir, f"plot_stability_stacked_{algo}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(data_dir, f"plot_stability_stacked_{algo}.svg"),
            format="svg",
            bbox_inches="tight",
        )
        plt.close()

        # --- FIG 3: ADVANCED STABILITY & HEIGHT ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        adv_metrics = [
            ("rss", "Composite Stability (RSS Roll/Pitch) (rad)"),
            ("rel_height", "Body Height (m)"),
        ]

        for idx, (key, ylabel) in enumerate(adv_metrics):
            ax = axes[idx]
            for state, df in files.items():
                label = f"SF {'On' if state=='Bật' else 'Off'}"
                mean_col = f"mean_{key}"
                if mean_col not in df.columns:
                    continue

                # Plot raw data (transparent)
                ax.plot(
                    df["time_bin"].values,
                    df[mean_col].values,
                    color=colors[state],
                    alpha=0.15,
                    linewidth=1.0,
                )
                # Plot smoothed data
                ax.plot(
                    df["time_bin"].values,
                    smooth(df[mean_col]).values,
                    label=label,
                    color=colors[state],
                    linewidth=1.0,
                )

            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if idx == 0:
                ax.set_title(
                    f"Advanced Stability & Height Analysis ({algo})",
                    fontsize=14,
                    pad=20,
                )
                ax.legend(frameon=False, loc="upper right", ncol=2)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(data_dir, f"plot_adv_stability_{algo}.png"),
            dpi=400,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(data_dir, f"plot_adv_stability_{algo}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(data_dir, f"plot_adv_stability_{algo}.svg"),
            format="svg",
            bbox_inches="tight",
        )
        plt.close()

print("\nDone! Combined plots generated in map subfolders.", flush=True)
