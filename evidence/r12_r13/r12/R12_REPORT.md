# R12: kết quả trên cohort mới (Claude, 29/10/2026)

## Thiết kế, theo lock (07:54 UTC ngày 28/10, trước mọi kết quả R12)
- **Cohort:** 32 start từ các trajectory khác nhau của training split, không trùng mọi cohort trước. Seed khóa là
  42000000.
- **Planner:** JEPA-WM PushT, CEM 64 × 8 với 10 elite, horizon 6. Thực thi mean cuối theo kiểu open loop, mỗi start một
  seed.
- **Bốn arm,** dùng chung quần thể đầu tiên Q0:

  | Arm | Cost |
  |---|---|
  | raw | cost chính thức |
  | veldrop_z | z(vis) + z(pos) |
  | fixed_full | vis + α0·full |
  | fixed_pos | vis + α0·pos |

  α0 = sd(vis)/sd(full) tính trên Q0 rồi cố định suốt quá trình tìm kiếm.
- **Thu thập:** 9 chunk, 128/128 run, không crash. Log chỉ in thời gian và α0. Kết quả được xem lần đầu khi chạy phân
  tích đã khóa, đúng một lần (hash `r12_analyze.py` b4e46785… khớp lock).
- **Tính toàn vẹn:**
  - không có issue;
  - margin tính lại từ final_state lệch 0,0;
  - α0 và Q0 trùng giữa các arm của mỗi start;
  - α0 không đổi qua các vòng lặp.
- **Kiểm độc lập:** `verify_r12_independent.py` (cho Q1, Q2) và kiểm mô tả (4 contrast, 4 ô) đều cho ALL.

## Kết quả

| Arm | Margin TB | Success | Block-only | Lỗi agent | Lỗi block |
|---|---:|---:|---:|---:|---:|
| Cost chính thức | 3,057 | 13/32 | 21 | 52,9 | 19,3 |
| full, α0 cố định | 2,597 | 14/32 | 23 | 43,9 | 18,9 |
| position, α0 cố định | 2,429 | 14/32 | 23 | 39,9 | 17,4 |
| **VELDROP-z** | **1,562** | **20/32** | 22 | 16,5 | 23,4 |

- **Q2, phụ, khai báo trước (VELDROP-z − chính thức): VELDROP_Z_HELPS.**
  - D = −1,495 [−2,445; −0,724], median −0,539.
  - 26 start tốt hơn, 6 xấu hơn; sign test 0,0005.
  - Bỏ start thuận lợi nhất còn −1,189.
  - Success 13 → 20, thêm 7 mất 0; McNemar chính xác 0,0156.
  - Đây là phép lặp lại R9 trên dữ liệu mới, với bằng chứng mạnh hơn R9: CI không chứa 0 và không mất success nào.
- **Q1, chính (fixed_pos − fixed_full): POSITION_HELPS, yếu.**
  - D = −0,169 [−0,430; +0,091], median −0,136.
  - 22 tốt hơn, 10 xấu hơn; sign test 0,0501.
  - Bỏ start thuận lợi nhất còn −0,087.
  - Success 14 = 14 (thêm 1, mất 1).
  - Đạt quy tắc hoạt động nhưng CI chứa 0.
- **Mô tả (không phải kiểm định):**

  | Contrast | D [CI 95%] | Success thêm/mất |
  |---|---|---|
  | fixed_full − chính thức | −0,460 [−0,857; −0,145] | 2/1 |
  | fixed_pos − chính thức | −0,629 [−0,991; −0,328] | 2/1 |
  | veldrop_z − fixed_pos | −0,867 [−1,590; −0,294] | 6/0 |
  | veldrop_z − fixed_full | −1,035 [−1,721; −0,439] | 6/0 |

- **Trọng số:**
  - α0 trung bình 1,64 (từ 0,34 đến 4,07), khoảng 16 lần mức 0,1.
  - Trọng số ngầm của VELDROP-z trên phần position, tức sd(vis)/sd(pos), có trung vị 3,74 ở quần thể đầu và 23,47 ở
    quần thể cuối (số liệu quá trình, hậu nghiệm).

## Diễn giải
- **Đã quan sát (thăm dò):** trên 32 goal mới, VELDROP-z dùng làm objective của CEM cải thiện cả margin lẫn success so
  với cost chính thức. Kết quả R9 lặp lại được trên dữ liệu mới.
- **Về cơ chế:** ở cùng một trọng số cố định trung gian, bỏ các thành phần liên quan vận tốc chỉ cho lợi ích nhỏ và bất
  định. Contrast lớn nhất nằm giữa trọng số thích nghi (tăng dần trong quá trình tìm kiếm) và trọng số cố định. Đây là
  so sánh hai contrast cụ thể, không phải phân bổ nhân quả.
- **Chưa trả lời được:**
  - bỏ vận tốc có quan trọng khi trọng số lớn hoặc thích nghi không, vì cohort này không có arm z(full);
  - kết quả với nhiều seed;
  - kết quả ở ngân sách đầy đủ hoặc MPC;
  - kết quả với goal ngoài training split hay task khác.
- **Gợi ý cho bước sau:** lặp lại R11 C01 (z(vis) + z(full)) trên cohort mới này, chỉ cần 32 run. Phép thử này sẽ
  tách được vai trò của vận tốc dưới trọng số thích nghi.

## Tệp
- **Kết quả và phân tích:**
  - `r12/out/r12_inloop.json` (sha256 2ba48a56…);
  - `r12/r12_analysis.json` (d788c0ca…);
  - `logs/r12_analyze.log`, `logs/r12_verify_independent.log`, `logs/r12_verify_descriptive.log`;
  - các log `r12_chunk1–9`.
- **Bài v2:** `paper_icai_r12/ICAI2026_LeVanManh_v2.pdf` (7/8 trang) và `CHANGELOG_v2.md`.

## Cập nhật
Gợi ý ở trên đã thành R13: khóa lúc 23:41 UTC ngày 28/10 (06:41 giờ VN ngày 29/10) và đang chạy (`r13/R13_LOCK.md`).
