import os
import csv
import matplotlib

matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# ─── CẤU HÌNH ĐƯỜNG DẪN ────────────────────────────────────────────────────────
# ==============================================================================

# 1. Locomotion Training Logs
file_reward = (
    "/home/datvu/LeggedGym-Ex/test_results/Apr19_12-55-01_ts_terrain_genesis (1).csv"
)
file_length = (
    "/home/datvu/LeggedGym-Ex/test_results/Apr19_12-55-01_ts_terrain_genesis.csv"
)
output_loco = "/home/datvu/LeggedGym-Ex/test_results/training_progress_plot.svg"

# 2. Safety Training Log
file_safety_reward = "/home/datvu/LeggedGym-Ex/test_results/go2_safety_terrain.csv"
output_safety = "/home/datvu/LeggedGym-Ex/test_results/safety_progress_plot.svg"

# ==============================================================================
# ─── HÀM HỖ TRỢ ───────────────────────────────────────────────────────────────
# ==============================================================================


def smooth(series, window=40):
    """Làm mượt dữ liệu bằng rolling mean sử dụng numpy."""
    series = np.array(series)
    if len(series) < window:
        return series
    smoothed = np.zeros(len(series))
    for i in range(len(series)):
        start = max(0, i - window + 1)
        end = i + 1
        smoothed[i] = np.mean(series[start:end])
    return smoothed


def load_csv_data(filepath):
    """Đọc dữ liệu từ file CSV chứa hai cột 'Step' và 'Value'."""
    steps = []
    values = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        step_idx = header.index("Step")
        val_idx = header.index("Value")
        for row in reader:
            if len(row) == len(header):
                steps.append(float(row[step_idx]))
                values.append(float(row[val_idx]))
    return np.array(steps), np.array(values)


def setup_style():
    """Cấu hình font Serif (Times New Roman style) và định dạng chung."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 12,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "legend.fontsize": 10,
        }
    )


# ==============================================================================
# ─── THỰC THI VẼ ĐỒ THỊ ───────────────────────────────────────────────────────
# ==============================================================================


def plot_all():
    setup_style()

    # --- PHẦN 1: LOCOMOTION TRAINING ---
    if os.path.exists(file_reward) and os.path.exists(file_length):
        print("Đang vẽ đồ thị Locomotion Training...", flush=True)
        steps_r, val_r = load_csv_data(file_reward)
        steps_l, val_l = load_csv_data(file_length)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # 1.1 Reward
        ax1 = axes[0]
        ax1.plot(
            steps_r,
            val_r,
            color="#1F77B4",
            alpha=0.3,
            label="mean reward (raw)",
        )
        ax1.plot(
            steps_r,
            smooth(val_r, window=60),
            color="#1F77B4",
            linewidth=1.5,
            label="mean reward (smooth)",
        )
        ax1.set_ylabel("mean reward")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="lower right", frameon=True)

        # 1.2 Length
        ax2 = axes[1]
        ax2.plot(
            steps_l,
            val_l,
            color="#2CA02C",
            alpha=0.3,
            label="mean episode length (raw)",
        )
        ax2.plot(
            steps_l,
            smooth(val_l, window=60),
            color="#2CA02C",
            linewidth=1.5,
            label="mean episode length (smooth)",
        )
        ax2.set_ylabel("mean episode length")
        ax2.set_xlabel("step")
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="lower right", frameon=True)

        plt.tight_layout()
        plt.savefig(output_loco, format="svg", bbox_inches='tight')
        print(f"✅ Đồ thị locomotion đã lưu tại: {output_loco}")
        plt.close()
    else:
        print("[!] Bỏ qua Locomotion (Không tìm thấy file).")

    # --- PHẦN 2: SAFETY TRAINING ---
    if os.path.exists(file_safety_reward):
        print("Đang vẽ đồ thị Safety Training...", flush=True)
        steps_s, val_s = load_csv_data(file_safety_reward)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            steps_s,
            val_s,
            color="#D62728",
            alpha=0.3,
            label="mean reward (raw)",
        )
        ax.plot(
            steps_s,
            smooth(val_s, window=40),
            color="#D62728",
            linewidth=1.5,
            label="mean reward (smooth)",
        )

        ax.set_ylabel("mean reward")
        ax.set_xlabel("step")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", frameon=True)

        plt.tight_layout()
        plt.savefig(output_safety, format="svg", bbox_inches='tight')
        print(f"✅ Đồ thị safety đã lưu tại: {output_safety}")
        plt.close()
    else:
        print("[!] Bỏ qua Safety (Không tìm thấy file).")


if __name__ == "__main__":
    plot_all()
