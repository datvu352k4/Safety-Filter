# Báo Cáo Kỹ Thuật Chuyên Sâu: Tích hợp Mạng điều khiển (Locomotion) và Lọc an toàn (Safety Filter) cho Robot chân nhện Unitree Go2

> [!IMPORTANT]
> **Ghi chú Dành Cho Mạng Trí Tuệ Nhân Tạo (AI LLM Co-author Prompt)**:
> Tài liệu này cung cấp toàn bộ chi tiết logic, kiến trúc toán học (mathematical formulations), quy trình huấn luyện và thông tin nền tảng về thuật toán mới nhất của hệ thống điều khiển tự hành và độ an toàn. Hãy sử dụng những thông tin sau đây làm rường cột (ground truth) để phát triển nội dung chuyên sâu cho các phần: Mở đầu (Introduction), Phương pháp luận (Methodology/System Architecture), Hàm khen thưởng (Reward Shaping), Thiết kế Môi trường (Simulation Setup) và Kết quả thực nghiệm (Experimental Benchmarking) của một bài báo khoa học.

---

## 1. Tóm Tắt Dự Án (Executive Summary)

Dự án này giải quyết bài toán cốt lõi trong nghiên cứu kiểm soát robot legged động lực học cao: **Sự mâu thuẫn giữa hoạch định đường đi (Path Planning) cấp cao và giới hạn vật lý bám dính cấp thấp.** 

Khi các bộ lập kế hoạch cục bộ (DWA, MPPI) đưa ra lệnh vận tốc $(v_x, v_y, \omega_{yaw})$ theo thời gian thực dựa trên quét không gian (Lidar) và mục tiêu điểm (A*), chúng thường không nhận thức được (blind) giới hạn ma sát và cấu trúc địa hình bất định dưới chân robot. Việc mù quáng thực thi lệnh gốc trong môi trường trơn trượt có thể gây ra hiện tượng xòe chân, vấp ngã hoặc đánh võng va chạm OBB (Oriented Bounding Box).

Để giải quyết vấn đề, chúng tôi đề xuất một **Kiến trúc Phân cấp Cắt ghép (Decoupled Hierarchical Architecture)**. Hệ thống đưa vào một màng phản xạ thông minh **Safety Filter (Lớp lọc an toàn)** hoạt động ở giao diện giữa Điều hướng (Navigation) và Điều khiển (Locomotion). 

Bằng việc trích xuất **Biểu diễn ẩn (Latent Representation $z_t$)** từ lớp nơ-ron sâu của mạng Locomotion, Safety Filter có thể ước lượng độ suy giảm của bề mặt (ma sát) và sinh ra một dải hệ số nén $\alpha \in [0.2, 1.0]^3$. Lớp lọc tự động bóp nghẹt các vận tốc yêu cầu trên mọi trục, ép hệ thống cấp cao tuân thủ ranh giới an toàn động lực học (Feasible Velocity Bounds) mà không cần can thiệp quy tắc cứng (Hard-coded heuristics).

---

## 2. Luồng Xử Lý Dữ Liệu Tích Hợp (Data Flow Architecture)

Hệ thống hoạt động theo vòng lặp thời gian thực 50Hz, liên kết 3 thành phần chính:

### Bước 1: Mạng Locomotion (Student Policy)
- **Đầu vào**: Lịch sử 15 Frames quan sát thô từ cảm biến.
- **History Encoder**: Nén chuỗi lịch sử thành Trạng thái ẩn $z_t \in \mathbb{R}^3$.
- **Actor**: Tính toán 12 Lực/Góc Khớp $\tau$ trực tiếp tới động cơ.
- *Hệ thống "Hook" biến ẩn $z_t$ này để làm đầu vào cho Safety Filter.*

### Bước 2: Mạng Safety Filter (Meta-Controller)
- **Đầu vào (Obs 6D)**: $z_t$ (3D) + $\alpha_{t-1}$ (3D).
- **Đầu ra (Actor)**: Hành động thô $Raw\_{\alpha} \in \mathbb{R}^3$.
- **Kích hoạt End-to-End**: 
  $$\alpha_t = \text{clamp}\left(\frac{\tanh(Raw\_\alpha) + 1.0}{2.0}, 0.2, 1.0\right)$$
- **Ràng buộc biên**: $\alpha \in [0.2, 1.0]^3$ (Cho 3 kênh $v_x, v_y, \omega_{yaw}$).
- **Tính trơn tự nhiên**: Phạt Smoothness phi tuyến ngay trong RL Loss, giúp hệ thống tự học cách phanh/ga thay vì dùng bộ lọc LPF cứng.

### Bước 3: Hệ Thống Điều Hướng Cục Bộ (MPPI / DWA)
- Sinh quỹ đạo không gian từ Lidar (144 Raycast) để tránh vật cản.
- **Giới hạn vận tốc**: $V_{bounds} = \alpha \times V_{max\_ideal}$.
- Cấp lệnh mới $(v_x, v_y, \omega_{yaw})$ đã qua lọc tới tầng Locomotion.

---

## 3. Cấu Trúc Khối Tự Hành (Locomotion)

### 3.1 Không Gian Quan Sát & Kiến Trúc Neural Networks

Mạng Locomotion sử dụng giải thuật **PPO (Proximal Policy Optimization)**. Quá trình huấn luyện sử dụng kiến trúc Teacher-Student:

**Chi tiết các thành phần mạng (Input Breakdown):**

1.  **Base Observations (45D)**: Quan sát thô từ cảm biến tại một thời điểm $t$.
    - Lệnh vận tốc (Commands): 3D
    - Trọng lực chiếu (Projected Gravity): 3D
    - Vận tốc góc thân (Base Angular Velocity): 3D
    - Sai số vị trí khớp (DOF Position Error): 12D ($q_{current} - q_{default}$)
    - Vận tốc khớp (DOF Velocity): 12D
    - Lệnh khớp trước đó (Previous Actions): 12D

2.  **Teacher (Privilege Encoder) - 29D**: Nhận các thông tin đặc quyền (Privileged Observations) mà robot không thể đo trực tiếp bằng cảm biến tích hợp. Cấu trúc mạng là MLP `[256, 128]`.
    -   **Friction (1D)**: Hệ số ma sát bề mặt.
    -   **Height Statistics (12D)**: Thống kê địa hình quanh 4 bàn chân.
        -   Variance ($4 \times 1$): Độ biến thiên độ cao tức thời trên từng chân.
        -   Range ($4 \times 1$): Khoảng cách $\max(h) - \min(h)$.
        -   Mean ($4 \times 1$): Độ cao tương đối trung bình trên từng chân.
    -   **Height Var EMA (4D)**: Trung bình trượt lũy thừa (EMA với $\alpha=0.05$) của biến thiên độ cao: $EMA_t = 0.95 \times EMA_{t-1} + 0.05 \times Variance_t$.
    -   **Normal Vector (12D)**: Pháp tuyến bề mặt tại 4 vị trí chân $(n_x, n_y, n_z) \times 4$.

3.  **Student (History Encoder) - 675D**: Học cách phỏng đoán thông tin môi trường từ lịch sử.
    -   **Inputs**: Lịch sử $15$ khung hình liên tiếp của Base Observations ($15 \times 45D = 675D$).
    -   **Architecture**: Sử dụng kiến trúc mạng **MLP** với các lớp ẩn `[256, 128]`. Cấu trúc hệ thống hỗ trợ cả TCN (Temporal Convolutional Network) nhưng ở cấu hình hiện tại, MLP được ưu tiên.
    -   **Outputs**: Biểu diễn ẩn $z_t \in \mathbb{R}^3$. Mục tiêu là sao chép được đầu ra của Teacher thông qua Imitation Learning.

4.  **Actor (Policy) - 48D**: Mạng sinh hành động chính.
    -   **Inputs**: Base Observation (45D) + Biến ẩn ước lượng $\hat{z}_t$ (3D).
    -   **Outputs**: 12 tọa độ lệnh khớp (Target Actions).

5.  **Critic (Value) - 885D**: Đánh giá giá trị của trạng thái hiện tại. Cấu trúc mạng là MLP `[1024, 256, 128]`.
    - Sử dụng **Temporal Stack của 5 khung hình** ($5 \times 177D = 885D$). Mỗi khung hình 177D bao gồm:
    -   **Obs (45D)**: Base Observation.
    -   **Domain Randomization Info (31D)**: Bao gồm Friction (1D), Added Mass (1D), CoM Bias (3D), Push Velocities (2D), KP/KD Scales ($12D + 12D$). (Lưu ý: trên thực tế chỉ Friction được randomize để học).
    -   **Linear Velocity (3D)**: Vận tốc tịnh tiến thực của robot.
    -   **Contact States (17D)**: Trạng thái tiếp xúc của các liên kết.
    -   **Height Measurements (81D)**: Lưới độ cao xung quanh thân robot.

### 3.2 Hàm Khen Thưởng Locomotion (Reward Shaping)

Hệ thống reward được thiết kế để cân bằng giữa việc bám sát lệnh điều khiển, đảm bảo tính ổn định và bảo vệ phần cứng. Dưới đây là danh sách chi tiết các trọng số (scales) trong `Go2TSTerrainCfg`:

#### 1. Nhóm Bám Sát Lệnh (Command Tracking)
- **Tracking Linear Velocity** (Trọng số **1.5**): Khuyến khích robot bám sát lệnh vận tốc tịnh tiến $v_{xy}$.
  - $R_{lin\_vel} = \exp\left(-\frac{||v_{xy} - v^{cmd}_{xy}||^2}{\sigma_{tracking}}\right)$
- **Tracking Angular Velocity** (Trọng số **1.0**): Khuyến khích bám sát lệnh vận tốc góc $\omega_{yaw}$.
  - $R_{ang\_vel} = \exp\left(-\frac{(\omega_{yaw} - \omega^{cmd}_{yaw})^2}{\sigma_{tracking}}\right)$

#### 2. Nhóm Dáng Đi và Tiếp Xúc (Gait & Contact)
- **Feet Air Time** (Trọng số **1.0**): Thưởng cho thời gian chân ở trên không để tạo bước sải dài. Chỉ tính khi chân vừa chạm đất.
  - $R_{air\_time} = \sum (T_{air} - 0.25) \cdot \mathbb{I}_{first\_contact}$ (với lệnh di chuyển $||v_{cmd}|| > 0.1$)
- **Foot Clearance** (Trọng số **0.2**): Giữ bàn chân ở độ cao mục tiêu trong pha swing để tránh vấp ngã.
  - $R_{clearance} = \exp\left(-\frac{\sum ||v^{foot}_{xy}|| \cdot (z_{foot} - z_{target} - z_{offset})^2}{\sigma_{clearance}}\right)$
- **Feet Contact Stand Still** (Trọng số **0.5**): Khuyến khích các chân tiếp đất khi không có lệnh di chuyển ($||v_{cmd}|| < 0.2$).
  - $R_{stand\_still} = \mathbb{I}_{\text{all\_feet\_contact}} \cdot \mathbb{I}_{||v_{cmd}|| < 0.2}$

#### 3. Nhóm Độ Mượt và Ổn Định (Stability & Smoothness)
- **Lin Vel Z** (Trọng số **-0.5**): Phạt vận tốc theo trục Z của thân robot để giảm xóc nảy.
  - $R_{vel\_z} = -v_z^2$
- **Ang Vel XY** (Trọng số **-0.05**): Phạt vận tốc góc Pitch/Roll để giữ thân robot luôn phẳng.
  - $R_{ang\_vel\_xy} = -||\omega_{xy}||^2$
- **Action Rate / Smoothness** (Trọng số **-0.005**): Phạt sự thay đổi đột ngột và độ giật của lệnh điều khiển.
  - $R_{action\_rate} = -\sum (a_t - a_{t-1})^2$
  - $R_{action\_smooth} = -\sum (a_t - 2a_{t-1} + a_{t-2})^2$

#### 4. Nhóm Bảo Vệ Phần Cứng (Hardware Constraints)
- **Collision** (Trọng số **-1.0**): Phạt nặng khi các bộ phận không phải bàn chân va chạm với địa hình.
  - $R_{collision} = -\sum_{\text{links}} \mathbb{I}_{||\mathbf{F}_{link}|| > 0.1}$
- **DOF Position Limits** (Trọng số **-2.0**): Phạt khi các khớp tiến gần hoặc vượt quá giới hạn vật lý.
  - $R_{dof\_limits} = -\sum (\max(0, q - q_{max}) + \max(0, q_{min} - q))$
- **DOF Power** (Trọng số **-2.0e-5**): Phạt công suất tiêu thụ cơ học để tiết kiệm năng lượng.
  - $R_{power} = -\sum |\tau \cdot \dot{q}|$
- **DOF Acc** (Trọng số **-2.0e-8**): Phạt gia tốc khớp để bảo vệ động cơ và giảm rung.
  - $R_{acc} = -\sum (\ddot{q})^2$
- **DOF Close To Default** (Trọng số **-0.05**): Kéo các khớp về vị trí mặc định.
  - $R_{dof\_pos} = -\sum (q - q_{default})^2$
- **Hip Pos** (Trọng số **-0.05**): Phạt riêng cho việc lệch khớp háng.
  - $R_{hip\_pos} = -\sum_{i \in \text{hips}} (q_i - q_{i, default})^2$

### 3.3 Chiến Lược Huấn Luyện (Curriculum Learning)

- Sử dụng **Curriculum Terrain**: Địa hình tăng dần độ khó (Flat -> Rough), các thông số vận tốc lệnh cũng được mở rộng dần (lên tới tối đa $2.0 m/s$).
- Huấn luyện PPO với số Iterations tối đa là **3000**.
- Optimizer với Learning Rate của History Encoder là $2.0 \times 10^{-4}$ qua 2 epochs.

---

## 4. Mạng Lọc An Toàn (Safety Filter)

Safety Filter giải quyết khiếm khuyết của các hệ LPF truyền thống bằng một bộ điều khiển học sâu, đánh đổi tốc độ để lấy sự sống còn. Trong quá trình huấn luyện Safety Filter, **mạng Locomotion được đóng băng (frozen)**.

### 4.1 Kiến Trúc Mạng Safety Filter

Sử dụng thuật toán **PPO** với Learning Rate $3 \times 10^{-4}$, Entropy Coef $0.003$ trong vòng **5000 Iterations**.

**Chi tiết các module đầu vào (Input Breakdown):**

1.  **Safety Actor - 6D**: Nhỏ gọn để đảm bảo phản xạ thời gian thực (<1ms inference). Cấu trúc mạng MLP `[256, 128]`.
    -   **Latent State $\hat{z}_t$ (3D)**: Biểu diễn nén của môi trường (từ History Encoder của Locomotion).
    -   **Last Alpha $\alpha_{t-1}$ (3D)**: Hành động nén vận tốc ở frame trước.
    -   **Đầu ra**: 3 hệ số nén $\alpha \in [0.2, 1.0]$ tương ứng với các lệnh vận tốc $v_x, v_y, \omega_{yaw}$.

2.  **Safety Critic - 72D**: Chứa các thông tin đầy đủ để tính toán hàm giá trị. Cấu trúc mạng MLP `[512, 256, 128]`.
    -   **Physics Core (16D)**: $\hat{z}_t$ (3D), Ma sát thực tế từ simulator (1D), Vận tốc tuyến tính (3D), Vận tốc góc (3D), Trọng lực chiếu (3D), Độ cao thân (1D).
    -   **Control Info (6D)**: Hướng lệnh di chuyển mong muốn (3D) + Hệ số $\alpha_{t-1}$ hiện tại (3D).
    -   **Local Terrain Context (50D)**:
        -   **Height Var (4D)**: Biến thiên độ cao tức thời tại vị trí 4 chân.
        -   **Relative Height Map (36D)**: Bản đồ độ cao tương đối quanh 4 chân đã được flatten ($\text{clamp}(z_{foot} - z_{ground}, -1, 1)$).
        -   **Contact Normals (12D)**: Pháp tuyến bề mặt tại 4 điểm tiếp xúc chân.

### 4.2 Thiết Kế Hàm Khen Thưởng (Reward Shaping)

Hệ thống sử dụng các hàm thưởng sau:

1.  **Phạt Trượt Chân Động Lực ($R_{slip}$)** - Trọng số: `20.0`
    - Cảm biến lực đọc ngưỡng $10N$ để xác định chân chạm đất.
    - Sử dụng **Dynamic Deadzone** chỉ nới lỏng khi có lệnh xoay (Yaw), vì xoay yêu cầu chân phải trượt một chút:
      $$YAW\_ALLOWANCE = 0.05 \times |\omega_{yaw\_cmd\_mag}|$$
      $$DZ_X = 0.32 + YAW\_ALLOWANCE$$
      $$DZ_Y = 0.08 + YAW\_ALLOWANCE$$
    - Phạt tổng độ trượt trên các trục X, Y vượt quá Deadzone.

2.  **Khuyến Khích Tăng Tốc Alpha ($R_{\alpha}$)** - Trọng số: `1.5`
    - $\text{Reward} = \sum_{i} \alpha_i \times (1.5 / 3.0)$.
    - Khuyến khích alpha tối đa ($1.0$) nếu môi trường an toàn, tránh việc model lạm dụng $\alpha=0.2$ để đứng yên. Tính riêng rẽ từng trục để không "hi sinh vận tốc trục này bù điểm trục khác".

3.  **Bất Đối Xứng Chấn Động $\alpha$ ($R_{smooth}$)** - Trọng số: `1.5`
    - Đóng vai trò là **LPF học sâu phi tuyến**.
    - Gọi $\Delta \alpha = \alpha_t - \alpha_{t-1}$.
    - Phạt nếu **Tăng tốc** ($\Delta \alpha > 0$): $\Delta \alpha^2 \times 1.5$.
    - Phạt nếu **Phanh gấp** ($\Delta \alpha \leq 0$): $\Delta \alpha^2 \times 0.1$.
    - Điều này cho phép phanh gấp để tránh nguy hiểm nhưng yêu cầu tăng tốc từ tốn.

4.  **Chống Ngã/Lật ($R_{ori}$)** - Trọng số: `30.0`
    - Lỗi lật được phát hiện khi $Projected\_Gravity_z > 0.0$.
    - Hệ số phạt bị giảm bớt ($Scale = \max(1.0 - 0.5 \times |yaw|, 0.5)$) khi robot đang thực hiện lệnh xoay nhanh.

### 4.3 Chiến Lược Huấn Luyện Meta-Controller

-   **Mô phỏng ngẫu nhiên (Command Resampling)**: Tái lấy mẫu ngẫu nhiên hướng lệnh (25% X, 25% Y, 25% Yaw, 15% X+Y, 10% Đứng yên) với chu kỳ thay đổi từ 20 đến 150 bước.
-   **Grace Period**: Áp dụng thời gian "ân hạn" 8 steps đầu tiên khi lệnh thay đổi đột ngột, giảm nhẹ hình phạt trượt chân.
-   **Không dùng Curriculum**: Địa hình ngẫu nhiên được rải ngẫu nhiên ngay từ đầu, cấu hình $\text{curriculum = False}$ giúp Safety Filter học khái quát từ mọi điều kiện khắc nghiệt.

---

## 5. Đánh Giá Thực Nghiệm & Các Chỉ Số Định Lượng (Experimental Benchmarking & Metrics)

Trong quá trình chạy mô phỏng thực tế qua Genesis Simulator trên các địa hình trơn trượt phức tạp kết hợp chướng ngại vật tĩnh (bản đồ `map1`, `map2`, `warehouse`), hiệu năng của bộ điều hướng cục bộ kết hợp bộ lọc an toàn (Safety Filter + MPPI/DWA) được định lượng một cách nghiêm ngặt qua các nhóm chỉ số động học, độ ổn định và an toàn.

### 5.1 Tiêu Chí Đánh Giá Trạng Thái Hành Trình (Task Outcome Criteria)

Mỗi chu kỳ thử nghiệm (episode) được theo dõi liên tục ở tần số điều khiển $50\text{ Hz}$ ($\Delta t = 0.02\text{ s}$) và được phân loại kết quả dựa trên các điều kiện sau:

1. **Thành Công (Success)**:
   Robot tiếp cận vùng đích thành công khi khoảng cách Euclidean tới tọa độ đích $\mathbf{p}_g = (x_g, y_g)$ nhỏ hơn bán kính hội tụ $0.5\text{ m}$:
   $$d_{\text{goal}}(t) = \|\mathbf{p}_{xy}(t) - \mathbf{p}_g\|_2 < 0.5\text{ m}$$

2. **Thất Bại do Ngã/Lật (Fall Failure)**:
   Robot mất thăng bằng động học và xảy ra va chạm thân xe với mặt đất. Trạng thái này được phát hiện khi góc thái độ lệch quá lớn ($> 90^\circ$):
   $$\max(|\phi(t)|, |\theta(t)|) > 1.57\text{ rad}$$
   Trong đó, $\phi(t)$ và $\theta(t)$ lần lượt là góc Roll và Pitch tức thời của robot, được tính từ quaternion định hướng thân xe $\mathbf{q}_R(t)$ thông qua chuyển đổi Euler XYZ:
   $$[\phi(t), \theta(t), \psi(t)]^T = \text{EulerXYZ}(\mathbf{q}_R(t))$$

3. **Thất Bại do Kẹt (Stuck Failure)**:
   Khi robot không thể thoát khỏi chướng ngại vật hoặc vùng ma sát thấp mặc dù vẫn nhận lệnh di chuyển tiến từ hệ hoạch định cấp cao. Trạng thái này được kích hoạt khi robot di chuyển ít hơn $0.1\text{ m}$ trong vòng $200\text{ steps}$ liên tục (tương đương $4\text{ s}$):
   $$\|\mathbf{p}_{xy}(t) - \mathbf{p}_{xy}(t - 200)\|_2 < 0.1\text{ m} \quad \text{với} \quad |v_x^{\text{cmd}}(t)| > 0.1\text{ m/s}$$

4. **Thất Bại do Quá Giờ (Timeout)**:
   Hành trình vượt quá giới hạn số bước mô phỏng cực đại (mặc định $1,000,000\text{ steps}$) mà chưa đạt được bất kỳ điều kiện dừng nào ở trên.

---

### 5.2 Mô Hình Phát Hiện Va Chạm Hộp Định Hướng (OBB Collision Detection)

Để đánh giá chính xác độ an toàn tĩnh và động, hệ thống sử dụng cảm biến Lidar Spherical (quét $360^\circ$, gồm $144$ tia). Dữ liệu Lidar dạng điểm (Point Cloud) sau khi được thu thập sẽ qua hai bộ lọc tiền xử lý:
* **Bộ lọc độ cao (Height Filter)**: Loại bỏ các điểm quá cao hoặc quá thấp để chỉ giữ lại chướng ngại vật cản trở thân xe: $z_{\text{world}} \in [0.15\text{ m}, 1.0\text{ m}]$.
* **Bộ lọc khoảng cách hướng tâm (Radial Filter)**: Loại bỏ các điểm quét trúng chính chân robot khi sải chân mở rộng: $d_{\text{radial}} > 0.3\text{ m}$.

Mỗi điểm chướng ngại vật hợp lệ trong hệ tọa độ thế giới ${}^W\mathbf{p}_i = [x_i, y_i, z_i]^T$ được biến đổi ngược về hệ tọa độ gắn liền với thân robot (Local Robot Frame) thông qua vị trí thân robot ${}^W\mathbf{p}_R(t)$ và quaternion định hướng $\mathbf{q}_R(t)$:
$${}^R\mathbf{p}_i = \mathbf{q}_R^*(t) \otimes ({}^W\mathbf{p}_i - {}^W\mathbf{p}_R(t)) \otimes \mathbf{q}_R(t)$$

Sử dụng hộp va chạm định hướng OBB (Oriented Bounding Box) có kích thước thực tế bao quanh Go2 ($0.72\text{ m} \times 0.36\text{ m}$, tương ứng bán kính an toàn dọc trục $0.36\text{ m}$ và ngang trục $0.18\text{ m}$), va chạm được định nghĩa là sự xuất hiện của bất kỳ điểm Lidar nào rơi vào bên trong hộp này:
$$\text{Collision}(t) = \exists i \text{ s.t. } \left( |{}^R x_i| < 0.36\text{ m} \quad \text{và} \quad |{}^R y_i| < 0.18\text{ m} \right)$$

* **Số lần va chạm ($N_{\text{collisions}}$)** được đếm theo dạng sườn lên (edge-triggered): chỉ tăng lên $1$ khi hệ thống chuyển từ trạng thái an toàn sang va chạm, tránh việc đếm lặp khi robot cọ xát liên tục với chướng ngại vật trong nhiều chu kỳ kế tiếp.

---

### 5.3 Chỉ Số Động Học & Độ Mượt Điều Khiển (Control Kinematic & Smoothness Metrics)

Mục tiêu tối thượng của việc tích hợp Safety Filter là giảm thiểu rung lắc cơ học và các thay đổi đột ngột trong chỉ lệnh điều khiển cấp cao nhằm bảo vệ động cơ khớp và duy trì lực bám ổn định.

#### 1. Độ Mượt của Chỉ Lệnh Điều Khiển (Command Tracking Smoothness RMS)
Đo lường Root Mean Square (RMS) của độ biến thiên chỉ lệnh điều khiển giữa hai chu kỳ liên tiếp (tương đương với mức gia tốc chỉ lệnh trung bình):
* **RMS Vận tốc Dọc ($v_x$)**:
  $$RMS_{\Delta v_x} = \sqrt{\frac{1}{N} \sum_{k=1}^N \left(v_x[k] - v_x[k-1]\right)^2}$$
* **RMS Vận tốc Ngang ($v_y$)**:
  $$RMS_{\Delta v_y} = \sqrt{\frac{1}{N} \sum_{k=1}^N \left(v_y[k] - v_y[k-1]\right)^2}$$
* **RMS Vận tốc Góc Yaw ($\omega$)**:
  $$RMS_{\Delta \omega} = \sqrt{\frac{1}{N} \sum_{k=1}^N \left(\omega[k] - \omega[k-1]\right)^2}$$

#### 2. Chỉ Số Jerk của Chỉ Lệnh (Command Jerk Index)
Jerk Index đặc trưng cho độ giật và độ đột ngột trong thay đổi của gia tốc chỉ lệnh. Chỉ số Jerk càng cao chứng tỏ hệ thống bám đường A* / tránh vật cản bị dao động liên tục (chattering):
* **Jerk Dọc ($v_x$)**:
  $$Jerk_{v_x} = \frac{1}{N} \sum_{k=2}^N \left| (v_x[k] - v_x[k-1]) - (v_x[k-1] - v_x[k-2]) \right|$$
* **Jerk Ngang ($v_y$)**:
  $$Jerk_{v_y} = \frac{1}{N} \sum_{k=2}^N \left| (v_y[k] - v_y[k-1]) - (v_y[k-1] - v_y[k-2]) \right|$$
* **Jerk Góc Yaw ($\omega$)**:
  $$Jerk_{\omega} = \frac{1}{N} \sum_{k=2}^N \left| (\omega[k] - \omega[k-1]) - (\omega[k-1] - \omega[k-2]) \right|$$

#### 3. Độ Ổn Định Thái Độ Thân Xe (Body Attitude Stability RMS)
RMS sai lệch góc Roll ($\phi$) và Pitch ($\theta$) xung quanh điểm cân bằng ($0\text{ rad}$) để phản ánh mức độ lắc lư, nhấp nhô của robot khi di chuyển trên nền đất gồ ghề hoặc trơn trượt:
* **Stability RMS Roll**:
  $$RMS_{\phi} = \sqrt{\frac{1}{N} \sum_{k=1}^N \phi[k]^2}$$
* **Stability RMS Pitch**:
  $$RMS_{\theta} = \sqrt{\frac{1}{N} \sum_{k=1}^N \theta[k]^2}$$

---

### 5.4 Các Chỉ Số Hiệu Suất Điều Hướng (Navigation & Path Performance)

1. **Tổng Quãng Đường Thực Tế ($L_{\text{actual}}$)**:
   Quãng đường tích lũy thực tế mà tâm robot di chuyển trong suốt hành trình:
   $$L_{\text{actual}} = \sum_{k=1}^N \|\mathbf{p}_{xy}[k] - \mathbf{p}_{xy}[k-1]\|_2$$

2. **Quãng Đường Lý Thuyết A\* ($L_{A^*}$)**:
   Tổng chiều dài của đường đi tối ưu được hoạch định bởi thuật toán A* trên lưới bản đồ tĩnh không có chướng ngại vật động:
   $$L_{A^*} = \sum_{j=1}^{M-1} \|\mathbf{w}_{j+1} - \mathbf{w}_j\|_2$$
   Trong đó, $\mathbf{w}$ là các điểm waypoint được hoạch định và làm mịn thông qua bộ lọc Smooth Path.

3. **Chỉ Chỉ Số Khoảng Cách An Toàn Cực Tiểu (Minimum Distance to Obstacle - $D_{\text{min}}$)**:
   Khoảng cách gần nhất từ tâm robot đến bất kỳ chướng ngại vật nào được Lidar ghi nhận trong suốt quá trình chạy:
   $$D_{\text{min}} = \min_{k} \left( \min_{i} \| {}^W\mathbf{p}_R[k] - {}^W\mathbf{p}_i[k] \|_2 \right)$$

---

### 5.5 Phương Pháp Tổng Hợp & Thống Kê Nhiều Lượt Chạy (Multi-Run Statistical Aggregation Method)

Nhằm đảm bảo tính chính xác khoa học và loại bỏ tính ngẫu nhiên do sự thay đổi của các seed mô phỏng và địa hình sinh ra, hệ thống tích hợp bộ công cụ phân tích thống kê tự động tại [`calculate_mean_nav_results.py`](file:///home/datvu/LeggedGym-Ex/calculate_mean_nav_results.py). Bộ công cụ này thực hiện tổng hợp dữ liệu qua các bước chuẩn hóa nghiêm ngặt:

#### 1. Tỷ Lệ Thành Công Tổng Thể (Overall Success Rate - $SR$)
Được tính toán trên toàn bộ số lượt chạy thử nghiệm được cấu hình cho mỗi thuật toán (MPPI/DWA) và trạng thái của Safety Filter:
$$SR = \left( \frac{N_{\text{success}}}{N_{\text{total}}} \right) \times 100\%$$

#### 2. Bộ Lọc Lượt Chạy Thành Công (Success-Only Metric Filtering)
Khi robot gặp thất bại (ví dụ: ngã lật hoặc kẹt cứng giữa đường), các chỉ số như thời gian chạy, quãng đường thực tế và độ ổn định sẽ bị méo mó nghiêm trọng (robot bị ngã ngay lập tức sẽ có thời gian di chuyển rất ngắn và độ ổn định ảo sau ngã bằng 0). 

Để phản ánh chính xác hiệu năng vận hành ổn định, bộ công cụ tự động lọc bỏ các lượt chạy thất bại, chỉ tính toán giá trị trung bình ($\mu$) và độ lệch chuẩn ($\sigma$) của các chỉ số hiệu suất trên tập hợp các lượt chạy thành công ($\mathcal{S}$):
$$\mu_m = \frac{1}{N_{\text{success}}} \sum_{j \in \mathcal{S}} m_j$$
$$\sigma_m = \sqrt{\frac{1}{N_{\text{success}}} \sum_{j \in \mathcal{S}} (m_j - \mu_m)^2}$$
Trong đó, $m_j$ là chỉ số thứ $m$ đo được tại lượt chạy thứ $j$.

#### 3. Chỉ Số Độ Ổn Định Tổng Hợp (Composite Stability - $CS$)
Để có một chỉ số duy nhất đánh giá tổng quát mức độ lắc lư và mất thăng bằng của thân xe, bộ công cụ tổng hợp hai thành phần RMS của Roll và Pitch thành chỉ số Composite Stability theo công thức Root-Sum-Square (RSS):
$$CS_j = \sqrt{RMS_{\phi, j}^2 + RMS_{\theta, j}^2}$$
Chỉ số Composite Stability trung bình ($\mu_{CS} \pm \sigma_{CS}$) được tính toán trên toàn bộ tập hợp $\mathcal{S}$, cung cấp cái nhìn trực quan về chất lượng giữ thăng bằng của robot.

#### 4. Phân Tích Chuỗi Thời Gian Đồng Bộ (Synchronized Time-Series Binning)
Để so sánh đặc tính động lực học của hệ thống qua thời gian, lịch sử chuyển động của các lượt chạy thành công được phân hoạch đồng bộ vào các khoang thời gian (time bins) có độ rộng $\Delta t_{\text{bin}} = 0.1\text{ s}$:
$$t_{\text{bin}} = \text{round}(t, 1)$$

Tại mỗi khoang thời gian $t_{\text{bin}}$, hệ thống tính toán giá trị trung bình ($\mu$) và độ lệch chuẩn ($\sigma$) của tất cả các kênh dữ liệu:
* Lệnh vận tốc ($v_x$, $v_y$, $\omega_{yaw}$).
* Góc lệch thái độ ($\phi$, $\theta$) và độ lắc lư tức thời $RSS = \sqrt{\phi^2 + \theta^2}$.
* Độ cao thân xe tương đối so với bàn chân ($h_{\text{rel}}$).

Dữ liệu chuỗi thời gian tổng hợp này được xuất ra các tệp `.csv` riêng biệt (ví dụ: `timeseries_mppi_map2_SF_Bật.csv`) để làm cơ sở xây dựng đồ thị so sánh trực quan, minh chứng cho sự mượt mà và an toàn vượt trội khi tích hợp Safety Filter.

---

### 5.6 Công Cụ Trực Quan Hóa Đồ Thị Kết Quả (Result Visualization & Graphing Tool)

Để chuyển đổi các kết quả số học thô thành các đồ thị trực quan phục vụ cho công bố khoa học và phân tích chuyên sâu, hệ thống sử dụng module đồ họa chuyên dụng tại [`plot_nav_results.py`](file:///home/datvu/LeggedGym-Ex/plot_nav_results.py). Module này tự động quét các thư mục kết quả thống kê trung bình (`mean_data`) của từng bản đồ và kết xuất ra **3 nhóm hình vẽ chất lượng cao** tương thích chuẩn định dạng các bài báo (Paper-ready Figures) dưới 3 định dạng đồng thời: `.png` (độ phân giải cao 400 DPI), `.pdf` (định dạng vector cho LaTeX), và `.svg` (định dạng vector cho web).

#### 1. Phương Pháp Làm Mịn Dữ Liệu & Biểu Diễn Thống Kê (Smoothing & Styling)
Để loại bỏ nhiễu đo lường tần số cao của cảm biến mô phỏng mà không làm mất đi đặc tính động lực học của chuỗi thời gian, đồ thị áp dụng bộ lọc trung bình trượt (Moving Average) với cửa sổ lọc $W = 12$ bước thời gian:
$$y_{\text{smooth}}[k] = \frac{1}{W} \sum_{i=0}^{W-1} y[k-i]$$

Đồ thị biểu diễn song song hai thành phần:
* **Đường dữ liệu thô (Raw Data)**: Được vẽ nét mảnh với độ trong suốt cao (`alpha = 0.15`) nhằm phản ánh chính xác sự biến động tức thời của trạng thái cơ học.
* **Đường làm mịn (Smoothed Data)**: Đường nét đậm dày (`linewidth = 1.0`), sử dụng màu sắc phân biệt đặc trưng:
  * **Xanh lá (`#27AE60`)**: Cấu hình **Bật Safety Filter** (SF On).
  * **Cam đất (`#E67E22`)**: Cấu hình **Tắt Safety Filter** (SF Off).

---

#### 2. Chi Tiết Các Nhóm Hình Vẽ Đồ Thị (Figure Layout Breakdown)

##### **Hình 1: Phân Tích Lược Đồ Vận Tốc (Velocity Profile Analysis)**
Gồm 3 đồ thị xếp chồng đồng trục thời gian (3-row stacked subplots sharing X-axis), thể hiện trực quan khả năng kiểm soát tốc độ của Safety Filter:
* **Đồ thị 1**: Vận tốc tuyến tính dọc thân $v_x$ (m/s).
* **Đồ thị 2**: Trị tuyệt đối của vận tốc ngang $|v_y|$ (m/s).
* **Đồ thị 3**: Trị tuyệt đối của vận tốc góc Yaw $|\omega_{yaw}|$ (rad/s).
* *Ý nghĩa*: Minh chứng rõ ràng cơ chế "bóp nghẹt" (throttling) tức thời của Safety Filter khi đi qua vùng ma sát thấp hoặc sát vật cản để duy trì lực bám.

##### **Hình 2: Phân Tích Độ Ổn Định Thái Độ (Body Stability Analysis)**
Gồm 2 đồ thị xếp chồng đồng trục thời gian hiển thị sai lệch thái độ của thân xe:
* **Đồ thị 1**: Trị tuyệt đối của góc Roll $|\phi|$ (rad).
* **Đồ thị 2**: Trị tuyệt đối của góc Pitch $|\theta|$ (rad).
* *Ý nghĩa*: Phản ánh trực quan mức độ trượt, lắc lư và xóc nảy. Khi tắt Safety Filter, các đường Roll/Pitch thô sẽ có các đỉnh nhọn dao động cực lớn (thể hiện robot bị trượt mạnh hoặc vấp ngã).

##### **Hình 3: Phân Tích Thái Độ Nâng Cao & Độ Cao Thân Xe (Advanced Stability & Height Analysis)**
Gồm 2 đồ thị xếp chồng đồng trục thời gian đi sâu vào mối liên hệ giữa chiều cao và độ ổn định tổng hợp:
* **Đồ thị 1**: Độ ổn định thái độ tổng hợp $RSS = \sqrt{\phi^2 + \theta^2}$ (rad).
* **Đồ thị 2**: Độ cao tương đối của thân robot so với bàn chân $h_{\text{rel}}$ (m).
* *Ý nghĩa*: Cho phép đánh giá mối tương quan trực tiếp giữa việc hạ thấp trọng tâm thân xe của mạng điều khiển Locomotion cấp thấp và mức độ dao động thái độ tổng hợp của robot. Khi có Safety Filter, thân xe được duy trì phẳng ổn định ($RSS$ tiệm cận $0$) và độ cao $h_{\text{rel}}$ mượt mà, không bị sụt lún đột ngột.
