import os
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = {"Bật": "#27AE60", "Tắt": "#E67E22"}

def smooth(series, window=12):
    series = np.array(series)
    if len(series) < window:
        return series
    smoothed = np.zeros(len(series))
    for i in range(len(series)):
        start = max(0, i - window + 1)
        end = i + 1
        smoothed[i] = np.mean(series[start:end])
    return smoothed

def parse_case_csv(filepath):
    # Find header row
    header_idx = -1
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) > 0 and row[0] == 'time':
                header_idx = idx
                break
    
    if header_idx == -1:
        raise ValueError(f"Could not find timeseries header in {filepath}")
    
    # Parse the data
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
        headers = lines[header_idx]
        for row in lines[header_idx+1:]:
            if len(row) == len(headers):
                item = {}
                for h, val in zip(headers, row):
                    if h in ['time', 'vx', 'vy', 'vyaw', 'roll', 'pitch', 'rel_height', 'x', 'y', 'friction']:
                        item[h] = float(val)
                    else:
                        item[h] = val
                data.append(item)
                
    # Create dict of lists
    res = {h: [] for h in headers}
    for item in data:
        for h in headers:
            res[h].append(item[h])
            
    # Convert to numpy arrays
    for h in res:
        if h != 'terrain':
            res[h] = np.array(res[h])
        else:
            res[h] = list(res[h])
            
    # Compute composite stability (RSS of roll and pitch)
    res['rss'] = np.sqrt(res['roll']**2 + res['pitch']**2)
    
    # Compute cumulative distance traveled
    x = res['x']
    y = res['y']
    cum_dist = [0.0]
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        d = np.sqrt(dx**2 + dy**2)
        cum_dist.append(cum_dist[-1] + d)
    res['cum_dist'] = np.array(cum_dist)
    
    return res

def get_zone_intervals(df):
    cum_dist = df['cum_dist']
    frictions = df['friction']
    terrains = df['terrain']
    
    intervals = []
    if len(cum_dist) == 0:
        return intervals
        
    start_d = cum_dist[0]
    last_f = frictions[0]
    last_t = terrains[0]
    
    for i in range(1, len(cum_dist)):
        f = frictions[i]
        t_terr = terrains[i]
        d = cum_dist[i]
        
        if f != last_f or t_terr != last_t:
            intervals.append((start_d, d, last_f, last_t))
            start_d = d
            last_f = f
            last_t = t_terr
            
    intervals.append((start_d, cum_dist[-1], last_f, last_t))
    return intervals

def shade_zones_on_axes(ax, intervals, draw_labels, y_lims):
    # Shade styling
    # Friction 1.0 + Flat -> No shading (white)
    # Friction 0.1 + Flat -> Soft red (#FADBD8)
    # Friction 0.05 + Flat -> Soft dark red (#F5B7B1)
    # Friction 0.2 + Rough -> Soft blue (#D6EAF8)
    
    y_min, y_max = y_lims
    y_text = y_min + (y_max - y_min) * 0.85
    
    drawn_labels = set()
    
    for start, end, f, t in intervals:
        color = None
        label = None
        
        # Determine color and label based on zones
        if abs(f - 0.1) < 0.01:
            color = "#FADBD8"
            label = r"$\mu = 0.1$"
        elif abs(f - 0.05) < 0.01:
            color = "#F5B7B1"
            label = r"$\mu = 0.05$"
        elif t == "rough":
            color = "#D6EAF8"
            if abs(f - 0.2) < 0.01:
                label = r"rough, $\mu = 0.2$"
            else:
                label = "rough"
            
        if color:
            ax.axvspan(start, end, color=color, alpha=0.5, edgecolor="none", zorder=0)
            
            if draw_labels and label and (end - start) >= 0.1:
                if label not in drawn_labels:
                    drawn_labels.add(label)
                    # Add text label at the middle of the zone
                    mid_x = (start + end) / 2
                    ax.text(
                        mid_x, 
                        y_text, 
                        label, 
                        color="#5D6D7E", 
                        fontsize=8, 
                        ha="center", 
                        va="center",
                        rotation=0,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2),
                        zorder=10
                    )

def generate_case_plots(map_name, seed, df_on, df_off, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    
    # We use df_on (SF On) to define the reference environment zones
    intervals = get_zone_intervals(df_on)
    
    max_dist = max(df_on['cum_dist'].max(), df_off['cum_dist'].max())
    
    # --- FIGURE 1: VELOCITY PROFILE ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    v_metrics = [
        ("vx", "$|v_x|$ (m/s)"),
        ("vy", "$|v_y|$ (m/s)"),
    ]
    
    for idx, (key, ylabel) in enumerate(v_metrics):
        ax = axes[idx]
        
        # Plot SF On
        val_on = df_on[key]
        if key in ["vx", "vy"]:
            val_on = np.abs(val_on)
        # Raw data commented out for cleaner presentation
        # ax.plot(df_on["cum_dist"], val_on, color=colors["Bật"], alpha=0.15, linewidth=1.0, zorder=2)
        ax.plot(df_on["cum_dist"], smooth(val_on), label="SF On", color=colors["Bật"], linewidth=1.5, zorder=3)
        
        # Plot SF Off
        val_off = df_off[key]
        if key in ["vx", "vy"]:
            val_off = np.abs(val_off)
        # Raw data commented out for cleaner presentation
        # ax.plot(df_off["cum_dist"], val_off, color=colors["Tắt"], alpha=0.15, linewidth=1.0, zorder=2)
        ax.plot(df_off["cum_dist"], smooth(val_off), label="SF Off", color=colors["Tắt"], linewidth=1.5, zorder=3)
        
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Draw background zone shading
        ax.set_xlim(0, max_dist)
        y_lims = ax.get_ylim()
        
        # Only draw text labels on the first subplot
        draw_labels = (idx == 0)
        shade_zones_on_axes(ax, intervals, draw_labels, y_lims)
        
        if idx == 0:
            ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)

    axes[-1].set_xlabel("Cumulative Distance Traveled (m)")
    plt.tight_layout()
    fig_name = f"case_study_velocity_dist_{map_name}"
    plt.savefig(os.path.join(output_dir, f"{fig_name}.svg"), format="svg", bbox_inches="tight")
    plt.close()
    
    # --- FIGURE 2: STABILITY AND HEIGHT ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    s_metrics = [
        ("rss", "Stability Index (rad)"),
        ("rel_height", "Body Height (m)"),
    ]
    
    for idx, (key, ylabel) in enumerate(s_metrics):
        ax = axes[idx]
        
        # Plot SF On
        val_on = df_on[key]
        # Raw data commented out for cleaner presentation
        # ax.plot(df_on["cum_dist"], val_on, color=colors["Bật"], alpha=0.15, linewidth=1.0, zorder=2)
        ax.plot(df_on["cum_dist"], smooth(val_on), label="SF On", color=colors["Bật"], linewidth=1.5, zorder=3)
        
        # Plot SF Off
        val_off = df_off[key]
        # Raw data commented out for cleaner presentation
        # ax.plot(df_off["cum_dist"], val_off, color=colors["Tắt"], alpha=0.15, linewidth=1.0, zorder=2)
        ax.plot(df_off["cum_dist"], smooth(val_off), label="SF Off", color=colors["Tắt"], linewidth=1.5, zorder=3)
        
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Draw background zone shading
        ax.set_xlim(0, max_dist)
        y_lims = ax.get_ylim()
        
        draw_labels = (idx == 0)
        shade_zones_on_axes(ax, intervals, draw_labels, y_lims)
        
        if idx == 0:
            ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)

    axes[-1].set_xlabel("Cumulative Distance Traveled (m)")
    plt.tight_layout()
    fig_name = f"case_study_stability_dist_{map_name}"
    plt.savefig(os.path.join(output_dir, f"{fig_name}.svg"), format="svg", bbox_inches="tight")
    plt.close()

    # --- SAVE METRICS SUMMARY TO FILE ---
    summary_file = os.path.join(output_dir, "case_study_summary.txt")
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"CASE STUDY SUMMARY - MAP: {map_name.upper()} (Seed: {seed})\n")
        f.write("=" * 80 + "\n\n")
        
        metrics_list = [
            ("vx", "Velocity X |vx| (m/s)"),
            ("vy", "Velocity Y |vy| (m/s)"),
            ("rss", "Stability Index RSS (rad)"),
            ("rel_height", "Body Height (m)"),
        ]
        
        for name, df in [("Safety Filter On", df_on), ("Safety Filter Off", df_off)]:
            f.write(f"** Configuration: {name} **\n")
            for key, label in metrics_list:
                val = df[key]
                if key in ["vx", "vy"]:
                    val = np.abs(val)
                
                val_smooth = smooth(val)
                
                # Raw stats
                raw_max = np.max(val)
                raw_min = np.min(val)
                raw_mean = np.mean(val)
                
                # Smoothed stats
                smooth_max = np.max(val_smooth)
                smooth_min = np.min(val_smooth)
                smooth_mean = np.mean(val_smooth)
                
                f.write(f"- {label}:\n")
                f.write(f"  + Raw Data:\n")
                f.write(f"    * Max (Peak Top)   : {raw_max:.6f}\n")
                f.write(f"    * Min (Peak Bottom): {raw_min:.6f}\n")
                f.write(f"    * Mean (Average)   : {raw_mean:.6f}\n")
                f.write(f"  + Smoothed Data (Plotted):\n")
                f.write(f"    * Max (Peak Top)   : {smooth_max:.6f}\n")
                f.write(f"    * Min (Peak Bottom): {smooth_min:.6f}\n")
                f.write(f"    * Mean (Average)   : {smooth_mean:.6f}\n")
            f.write("\n")
        f.write("\n")

if __name__ == "__main__":
    output_dir = "/home/datvu/LeggedGym-Ex/test_results/case_studies"
    
    # Reset summary file at starting
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "case_study_summary.txt")
    if os.path.exists(summary_file):
        os.remove(summary_file)
        
    # Map 1: Seed 35
    print("Generating Map 1 plots...")
    df_on_map1 = parse_case_csv("/home/datvu/LeggedGym-Ex/test_results/map1/mppi/nav_data_mppi_with_SF#35.csv")
    df_off_map1 = parse_case_csv("/home/datvu/LeggedGym-Ex/test_results/map1/mppi/nav_data_mppi#35.csv")
    generate_case_plots("map1", 35, df_on_map1, df_off_map1, output_dir)
    
    # Warehouse (Map 3): Seed 56
    print("Generating Warehouse plots...")
    df_on_wh = parse_case_csv("/home/datvu/LeggedGym-Ex/test_results/warehouse/mppi/nav_data_mppi_with_SF#56.csv")
    df_off_wh = parse_case_csv("/home/datvu/LeggedGym-Ex/test_results/warehouse/mppi/nav_data_mppi#56.csv")
    generate_case_plots("warehouse", 56, df_on_wh, df_off_wh, output_dir)
    
    print("Done! Plots generated successfully in:", output_dir)
    print(f"Summary data saved in: {summary_file}")
