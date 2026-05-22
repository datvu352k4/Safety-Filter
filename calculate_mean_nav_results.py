import os
import glob
import csv
from collections import defaultdict
import numpy as np
import re


def parse_file(filepath):
    summary_data = {}
    history_data = []  # List of (t, vx, vy, vyaw, roll, pitch, rel_height)

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            # Đoạn 1: Parse summary
            header = next(reader)
            if header != ["Metric", "Value"]:
                return None, None
            for row in reader:
                if not row:  # Empty line signals end of summary
                    break
                if len(row) == 2:
                    summary_data[row[0]] = row[1]

            # Đoạn 2: Parse History (nếu có)
            for row in reader:
                if row and row[0] == "time":
                    break

            for row in reader:
                if row and len(row) >= 7:
                    try:
                        t = float(row[0])
                        vx = float(row[1])
                        vy = float(row[2])
                        vyaw = float(row[3])
                        roll = float(row[4])
                        pitch = float(row[5])
                        rel_height = float(row[6])
                        if len(row) >= 11:
                            x = float(row[7])
                            y = float(row[8])
                            friction = float(row[9])
                            terrain = row[10].strip()
                            history_data.append((t, vx, vy, vyaw, roll, pitch, rel_height, x, y, friction, terrain))
                        else:
                            history_data.append((t, vx, vy, vyaw, roll, pitch, rel_height))
                    except ValueError:
                        pass
                elif row and len(row) == 6:  # Backward compatibility
                    try:
                        history_data.append(
                            (
                                float(row[0]),
                                float(row[1]),
                                float(row[2]),
                                float(row[3]),
                                float(row[4]),
                                float(row[5]),
                                0.3, # Default height if missing
                            )
                        )
                    except ValueError:
                        pass
        except StopIteration:
            pass

    return summary_data, history_data


def main():
    test_results_dir = "/home/datvu/LeggedGym-Ex/test_results/"
    output_dir = os.path.join(test_results_dir, "mean_data")
    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(test_results_dir, "**", "*.csv"), recursive=True)
    csv_files = [f for f in csv_files if "mean_data" not in f]

    # Lọc data: với map1 và warehouse lấy từ 31 trở đi, map2 lấy từ 1 đến 30
    filtered_csv_files = []
    for f in csv_files:
        match = re.search(r'#(\d+)\.csv$', f)
        if match:
            idx = int(match.group(1))
            if "map1" in f or "warehouse" in f:
                if idx >= 31:
                    filtered_csv_files.append(f)
            elif "map2" in f:
                if 1 <= idx <= 30:
                    filtered_csv_files.append(f)
            else:
                filtered_csv_files.append(f)
        else:
            filtered_csv_files.append(f)
    csv_files = filtered_csv_files

    metrics_to_average = [
        "Remaining_Dist",
        "Total_Time",
        "Collisions",
        "Min_Dist",
        "Actual_Path",
        "AStar_Path",
        "RMS_Vx",
        "RMS_Vy",
        "RMS_Yaw",
        "Jerk_Vx",
        "Jerk_Vy",
        "Jerk_Yaw",
        "RMS_Roll",
        "RMS_Pitch",
    ]

    # grouped_data[algorithm][map_name][safety_filter_status] = (list_of_summaries, list_of_histories)
    grouped_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: ([], [])))
    )

    for filepath in csv_files:
        summary_data, history_data = parse_file(filepath)
        if (
            summary_data
            and "Algorithm" in summary_data
            and "Map" in summary_data
            and "Safety_Filter" in summary_data
        ):
            algo = summary_data["Algorithm"]
            map_name = summary_data["Map"]
            sf_status = summary_data["Safety_Filter"]

            grouped_data[algo][map_name][sf_status][0].append(summary_data)
            grouped_data[algo][map_name][sf_status][1].append(history_data)

    print(f"Bắt đầu phân tích {len(csv_files)} file dữ liệu csv...")

    import shutil

    old_output_dir = os.path.join(test_results_dir, "mean_data")

    for algo, maps in grouped_data.items():
        for map_name, sf_groups in maps.items():
            # Tạo đường dẫn output dựa trên tên map: test_results/<map_name>/mean_data
            map_output_dir = os.path.join(test_results_dir, map_name, "mean_data")
            os.makedirs(map_output_dir, exist_ok=True)

            # File xuất Summary
            summary_output = os.path.join(
                map_output_dir, f"mean_summary_{algo}_{map_name}.txt"
            )
            with open(summary_output, "w", encoding="utf-8") as f_out:
                f_out.write(
                    f"--- TỔNG HỢP TRUNG BÌNH SUMMARY: {algo} | MAP: {map_name} ---\n\n"
                )

                for sf_status, (records, histories) in sf_groups.items():
                    # --- Step 1: Filter only Success cases for the averages ---
                    success_records = [
                        r for r in records if r.get("Status", "") == "Success"
                    ]
                    # Map histories to their corresponding record status
                    success_histories = [
                        h
                        for r, h in zip(records, histories)
                        if r.get("Status", "") == "Success"
                    ]

                    n_total = len(records)
                    n_success = len(success_records)
                    success_rate = (n_success / n_total) * 100 if n_total > 0 else 0

                    f_out.write(
                        f"** Safety Filter: {sf_status} (Total runs: {n_total}, Successful: {n_success}) **\n"
                    )
                    f_out.write(f"- Success Rate: {success_rate:.1f}%\n")

                    if n_success == 0:
                        f_out.write(
                            "- [Warning] No successful runs in this category.\n\n"
                        )
                        continue

                    metric_values = defaultdict(list)
                    composite_stability_values = []

                    for r in success_records:
                        # Tính Composite Stability cho từng lượt chạy thành công
                        try:
                            r_roll = float(r.get("RMS_Roll", 0))
                            r_pitch = float(r.get("RMS_Pitch", 0))
                            comp_stab = np.sqrt(r_roll**2 + r_pitch**2)
                            composite_stability_values.append(comp_stab)
                        except:
                            pass

                        for m in metrics_to_average:
                            if m in r:
                                try:
                                    val = float(r[m])
                                    metric_values[m].append(val)
                                except ValueError:
                                    pass

                    for m in metrics_to_average:
                        vals = metric_values[m]
                        if vals:
                            mean_val = np.mean(vals)
                            std_val = np.std(vals)
                            f_out.write(f"- {m}: {mean_val:.4f} ± {std_val:.4f}\n")
                        else:
                            f_out.write(f"- {m}: N/A\n")

                    # Ghi thêm Composite Stability
                    if composite_stability_values:
                        mean_comp = np.mean(composite_stability_values)
                        std_comp = np.std(composite_stability_values)
                        f_out.write(
                            f"- Composite_Stability (RSS Roll/Pitch): {mean_comp:.4f} ± {std_comp:.4f} rad\n"
                        )
                        
                    # Calculate average timestamps of each terrain/friction change
                    all_change_timestamps = []
                    for hist in success_histories:
                        if not hist or len(hist[0]) < 11:
                            continue
                        last_fric = hist[0][9]
                        last_terr = hist[0][10]
                        current_run_changes = []
                        for row in hist[1:]:
                            t = row[0]
                            fric = row[9]
                            terr = row[10]
                            if fric != last_fric or terr != last_terr:
                                current_run_changes.append(t)
                                last_fric = fric
                                last_terr = terr
                        if current_run_changes:
                            all_change_timestamps.append(current_run_changes)
                                
                    if all_change_timestamps:
                        max_len = max(len(c) for c in all_change_timestamps)
                        f_out.write("- Terrain/Friction Change Timestamps (Average across runs):\n")
                        for i in range(max_len):
                            ith_changes = [c[i] for c in all_change_timestamps if i < len(c)]
                            mean_t = np.mean(ith_changes)
                            std_t = np.std(ith_changes)
                            count = len(ith_changes)
                            f_out.write(f"  + Change {i+1}: {mean_t:.4f} ± {std_t:.4f} s (from {count} runs)\n")
                    f_out.write("\n")

                    raw_bins = defaultdict(
                        lambda: {
                            "vx": [],
                            "vy": [],
                            "vyaw": [],
                            "roll": [],
                            "pitch": [],
                            "rel_height": [],
                            "rss": [],
                        }
                    )

                    for hist in success_histories:
                        for hist_row in hist:
                            t = hist_row[0]
                            vx = hist_row[1]
                            vy = hist_row[2]
                            vyaw = hist_row[3]
                            roll = hist_row[4]
                            pitch = hist_row[5]
                            rel_height = hist_row[6]
                            
                            t_bin = round(t, 1)
                            raw_bins[t_bin]["vx"].append(vx)
                            raw_bins[t_bin]["vy"].append(vy)
                            raw_bins[t_bin]["vyaw"].append(vyaw)
                            raw_bins[t_bin]["roll"].append(roll)
                            raw_bins[t_bin]["pitch"].append(pitch)
                            raw_bins[t_bin]["rel_height"].append(rel_height)
                            raw_bins[t_bin]["rss"].append(np.sqrt(roll**2 + pitch**2))

                    # Ghi kết quả timeseries ra file riêng bao gồm cả std
                    safe_sf = sf_status.replace(" ", "_")
                    ts_output = os.path.join(
                        map_output_dir, f"timeseries_{algo}_{map_name}_SF_{safe_sf}.csv"
                    )
                    with open(ts_output, "w", encoding="utf-8") as f_ts:
                        f_ts.write(
                            "time_bin,mean_vx,std_vx,mean_vy,std_vy,mean_vyaw,std_vyaw,mean_roll,std_roll,mean_pitch,std_pitch,mean_rel_height,std_rel_height,mean_rss,std_rss,sample_count\n"
                        )
                        for t_bin in sorted(raw_bins.keys()):
                            b = raw_bins[t_bin]
                            count = len(b["vx"])
                            if count > 0:
                                m_vx, s_vx = np.mean(b["vx"]), np.std(b["vx"])
                                m_vy, s_vy = np.mean(b["vy"]), np.std(b["vy"])
                                m_yaw, s_yaw = np.mean(b["vyaw"]), np.std(b["vyaw"])
                                m_roll, s_roll = np.mean(b["roll"]), np.std(b["roll"])
                                m_pitch, s_pitch = np.mean(b["pitch"]), np.std(
                                    b["pitch"]
                                )
                                m_h, s_h = np.mean(b["rel_height"]), np.std(b["rel_height"])
                                m_rss, s_rss = np.mean(b["rss"]), np.std(b["rss"])

                                f_ts.write(
                                    f"{t_bin:.1f},{m_vx:.4f},{s_vx:.4f},{m_vy:.4f},{s_vy:.4f},"
                                    f"{m_yaw:.4f},{s_yaw:.4f},{m_roll:.4f},{s_roll:.4f},"
                                    f"{m_pitch:.4f},{s_pitch:.4f},{m_h:.4f},{s_h:.4f},"
                                    f"{m_rss:.4f},{s_rss:.4f},{count}\n"
                                )

                    print(
                        f"[Timeseries] Done: {algo} - Map {map_name} - SF {sf_status}"
                    )

    # Xóa thư mục cũ nếu tồn tại
    if os.path.exists(old_output_dir) and os.path.isdir(old_output_dir):
        shutil.rmtree(old_output_dir)
        print(f"\nĐã xóa thư mục cũ: {old_output_dir}")

    print(
        f"\nThành công! Kết quả đã được phân loại theo từng Map trong thư mục: {test_results_dir}"
    )


if __name__ == "__main__":
    main()
