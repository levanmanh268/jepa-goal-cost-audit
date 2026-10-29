# RUNBOOK R12 (28/10/2026). Đọc tệp này ở mỗi lần tự kiểm tra

Mục tiêu: chạy hết 128 run của R12 (Q1 trước, Q2 sau), phân tích đúng một lần, báo cáo và gửi Mạnh.

Quy tắc chung:
- Mỗi lúc chỉ chạy một tiến trình nạp mô hình.
- Mỗi chunk chạy bằng Bash `run_in_background`, timeout 7200000 ms, script tự dừng sau 7000 s.
- Mỗi chunk ghi log riêng.
- Không sửa `r12_frozen_scale.py`, `r12_analyze.py`, `r12_cohort.json`. Nếu bắt buộc phải sửa vì lỗi hạ tầng, ghi
  `R12_DEVIATIONS.md` trước khi chạy tiếp.
- Không đọc kết quả trong `r12_inloop.json` trước khi phân tích; chỉ đếm số khóa.
- Chỉ báo trạng thái từ ps, log và số khóa trong JSON.

## 1. Kiểm trạng thái

```
date -u; ps aux | grep -E "r12_frozen_scale|run_r12" | grep -v grep
ls -la /home/claude/wm/logs/r12_chunk*.log; grep -E "exit code|Traceback|Error|Killed|refusing|mismatch" /home/claude/wm/logs/r12_chunk*.log | tail -5
python3 -I -c "import json;R=json.load(open('/home/claude/wm/r12/out/r12_inloop.json'))['runs'];print(len(R), sum(k.endswith(('fixed_full','fixed_pos')) for k in R), sum(k.endswith(('raw','veldrop_z')) for k in R))"
ls /home/claude/wm/r12/r12_analysis.json 2>/dev/null
```

## 2. Xử lý theo trạng thái

- **A. Chưa đủ 128 run, không có tiến trình nào chạy.** Chạy chunk tiếp theo, với N là số thứ tự chunk kế tiếp:
  `/home/claude/wm/r12/run_r12.sh > /home/claude/wm/logs/r12_chunkN.log 2>&1`
- **B. Đủ 128 run, chưa có `r12_analysis.json`.** Phân tích đúng một lần:
  ```
  cd /home/claude/wm/r12 && sha256sum r12_analyze.py out/r12_inloop.json && python3 -I r12_analyze.py r12_cohort.json out/r12_inloop.json r12_analysis.json | tee /home/claude/wm/logs/r12_analyze.log
  ```
  Sau đó tự kiểm lại bằng mã độc lập, viết `R12_REPORT.md`, cập nhật bài (bản mới, có changelog), đóng gói và gửi Mạnh.
- **C. Đến 20:00 ngày 29/10 giờ Việt Nam mà chưa đủ.** Chạy phân tích để nhận nhãn INCOMPLETE cho câu chưa xong, rồi báo
  trung thực.
- **D. Đã gửi xong.** Không làm gì thêm; báo ngắn trạng thái.
