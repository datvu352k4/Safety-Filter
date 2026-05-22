# Walkthrough — Go2 Terrain-Aware Latent Representation

## Mục tiêu phiên làm việc
Cải thiện khả năng phân biệt địa hình của History Encoder (Student) trong hệ thống Teacher-Student, giải quyết vấn đề `z_t` bị "học vẹt" và không phân biệt được ma sát vs độ nhấp nhô địa hình.

---

## 1. Thay đổi Kiến trúc

### 1.1 Mở rộng Latent Space: 8D → 12D
**File**: `go2_ts_terrain_config.py`
- `num_latent_dims = 12`
- Phân công: `z0`=friction, `z1`=roughness, `z2-z11`=geometry
- **Lý do**: 8D không đủ chứa cả ma sát lẫn geometry mà không tranh giành nhau

### 1.2 Tăng Privileged Obs: 9D → 10D
**File**: `go2_ts_terrain_config.py`
```
num_privileged_obs = 10  # friction(1) + roughness_level(1) + heights_stats_ema(8)
```

### 1.3 EMA Coefficient: 0.8 → 0.6
**File**: `go2_ts_terrain.py`
```python
self._heights_stats_ema = 0.6 * self._heights_stats_ema + 0.4 * h_stats
```
- **Lý do**: EMA 0.8 làm mịn quá → mất tín hiệu tần cao của rough terrain

---

## 2. Thay đổi Encoding Terrain

### 2.1 is_rough (Binary) → roughness_level (Continuous)
**File**: `go2_ts_terrain.py`
```python
# Trước: Binary threshold (gradient gián đoạn)
# is_rough = (h_std.mean(dim=-1) > 0.025).float()

# Sau: Continuous regression target [0, 1]
roughness_level = (h_std.mean(dim=-1, keepdim=True) / 0.08).clamp(0.0, 1.0)
```
- **Lý do**: Binary threshold cứng tạo signal gián đoạn; regression mượt mà hơn và tự scale theo max_height training

---

## 3. Thay đổi Loss Architecture

### 3.1 Decoder Target: h_stats(8D) → h_std_per_foot(4D)
**File**: `rsl_rl/algorithms/ppo_ts.py`
```python
# Trước: priv_obs[:, 2:10] = h_mean(4) + h_std(4) — h_mean gait-DEPENDENT
# Sau:   priv_obs[:, 6:10] = h_std(4) ONLY — gait-INVARIANT
```
**Lý do toán học**: `std(foot_z - terrain_h) = std(terrain_h)` — foot ở độ cao nào cũng không ảnh hưởng std. Nhưng `mean(foot_z - terrain_h) = foot_z - mean(terrain_h)` thay đổi theo foot height → gait-dependent → confuse encoder.

### 3.2 TerrainDecoder: 8D output → 4D
Khớp với decoder target mới.

### 3.3 z1 Loss: BCEWithLogitsLoss → MSE Regression
```python
# Trước: BCE classification (binary)
# Sau: MSE regression trên roughness_level liên tục
roughness_gt = privileged_obs_batch[:, 1]   # [0, 1] continuous
loss_terrain = F.mse_loss(torch.sigmoid(z_t[:, 1]), roughness_gt)
```

### 3.4 Loss Weights (Rebalanced)
```python
# Trước: 1.0×PPO + 1.0×Friction + 0.5×Terrain + 0.2×Recon
# Sau:   1.0×PPO + 2.0×Friction + 1.0×Terrain + 0.3×Recon
total_loss = loss_ppo + 2.0 * loss_friction + 1.0 * loss_terrain + 0.3 * loss_recon
```
**Lý do**: `Loss_PPO` là MSE 12D nên magnitude lớn tự nhiên. `Loss_Friction` là MSE 1D scalar → cần tăng weight để gradient cạnh tranh được.

---

## 4. Feature Flag (Backward Compatibility)
**File**: `rsl_rl/algorithms/ppo_ts.py`
```python
class PPO_TS:
    def __init__(self, ..., use_aux_terrain_loss=False):
        ...
        if self.use_aux_terrain_loss:
            self.terrain_decoder = TerrainDecoder(...)
```
**File**: `go2_ts_terrain_config.py`
```python
class algorithm:
    use_aux_terrain_loss = True  # Chỉ task này mới bật
```
**Đảm bảo**: Các task khác (G1, go2 cơ bản) dùng `PPO_TS` vẫn hoạt động bình thường với `use_aux_terrain_loss=False` (default).

---

## 5. Đồng bộ hóa Safety Filter
Cập nhật dimensions do latent 8D→12D:
- `go2_safety_terrain_config.py`: `num_observations = 15` (12+3), `num_privileged_obs = 68`
- `go2_safety_terrain_env.py`: Updated docstrings
- `test_safety_terrain.py`: `num_obs = 15`, comments 15D

---

## 6. Semantic của Latent Space 12D (Sau training)

| Chiều | Vai trò | Loss | Target |
|-------|---------|------|--------|
| `z0` | Friction predictor | MSE | `(friction - 0.05) / 1.65` → [0,1] |
| `z1` | Roughness level | MSE | `h_std.mean() / 0.08` → [0,1] |
| `z2-z11` | Geometry latents | Reconstruction | `h_std_per_foot` (4D) |

---

## 7. Kết quả Test (model 600it — model CŨ chưa có BCE/regression)

Từ log `Apr09_04-22-33`, model 600it (8D/12D nhưng chưa có terrain loss đúng):
- `μ_pred` (friction): ✅ Hoạt động tốt — rớt xuống 0.4-0.5 khi μ=0.1
- `Terrain` (z1): ❌ Chưa học được — báo "Flat" ngay cả trên sỏi đá
- **Root cause**: Decoder dùng h_mean (gait-dependent) → encoder học gait thay vì terrain geometry

---

## 8. Files đã thay đổi trong phiên này

| File | Nội dung thay đổi |
|------|-------------------|
| `go2_ts_terrain_config.py` | `num_latent_dims=12`, `num_privileged_obs=10`, `use_aux_terrain_loss=True`, `save_interval=100` |
| `go2_ts_terrain.py` | EMA 0.6, `roughness_level` continuous thay `is_rough` binary |
| `rsl_rl/algorithms/ppo_ts.py` | `use_aux_terrain_loss` flag, TerrainDecoder 4D, MSE regression loss, rebalanced weights |
| `go2_safety_terrain_config.py` | `num_obs=15`, `num_privileged_obs=68` |
| `go2_safety_terrain_env.py` | Updated docstrings 12D/15D/68D |
| `test_safety_terrain.py` | Comments 15D |
| `play_terrain.py` | Hiển thị `Roughness: Xcm (Y%)` thay Flat/Rough |

---

## 9. Bước tiếp theo
1. **Train**: `python legged_gym/scripts/train.py --task go2_ts_terrain`
2. **Monitor** (logs): Kiểm tra 4 loss riêng biệt hội tụ đồng đều
3. **Verify** (`play_terrain.py`): Trên sỏi 8cm, `Roughness` phải hiển thị >50%; trên mặt phẳng <10%
4. **Validate Safety Filter**: Sau encoder converge, re-train Safety Filter với 12D input mới

---

## 10. Kiến thức Kỹ thuật Quan trọng

### Tại sao h_std gait-invariant?
```
std(foot_z - terrain_h_i, i=1..M) = std(terrain_h_i, i=1..M)
```
Vì `foot_z` là hằng số với tất cả M điểm quanh chân đó, `std(X + c) = std(X)`.
Foot nhấc lên hay đặt xuống đều không ảnh hưởng std → signal thuần túy từ terrain roughness.

### Tại sao h_mean gait-dependent?
```
mean(foot_z - terrain_h_i) = foot_z - mean(terrain_h_i)
```
`foot_z` thay đổi theo gait cycle → mean thay đổi → confuse encoder.

### Entanglement Problem
`z0` bị "chồng lấn" friction×roughness: sỏi trơn có `geometry_signal` push z0 lên, `friction_signal` pull z0 xuống → kết quả z0 giống mặt phẳng bám. Giải pháp: ép z1 học roughness riêng, z0 chỉ còn lo friction.
