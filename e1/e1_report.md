# Báo cáo E1 (2026-10-01)

**Nguồn:** `e1_raw.jsonl` và `e1_acc.jsonl`, đã loại các dòng superseded. Mọi số trong báo cáo này được tính bằng `summarize_e1.py`, kết quả nằm ở `e1_summary.csv`.

**Tiến độ:**
- Thời gian chạy xong lúc 04:00; đo lại run-or-walk xong lúc 05:03; độ chính xác xong lúc 06:31.
- Không thiếu đơn vị nào, không có dòng trùng.
- Các sửa đổi giao thức trong lúc chạy (bộ nhớ, tạm dừng, đo lại): xem `e1_prereg.md` mục 10.

## 1. Tính chính xác (K1): ĐẠT
- Có 332 nhóm (bộ × tiêu chí × tham số × fold) mà ít nhất hai phương pháp chính xác chạy xong. **Cả 332 nhóm cho cùng dãy thuộc tính theo đúng thứ tự chọn.**
- `e1_k1.jsonl` không phát sinh, nghĩa là eval không thấy nhóm nào lệch.
- 13 nhóm chỉ có một phương pháp chạy xong (gas-drift FRS, vehicle-sensit), nên không so được.

## 2. Tốc độ: KD-PA+B3 so với đối thủ chính xác tốt nhất đang có
Đối thủ "đang có" gồm FNC, HASH, CABIN, BUCKET, BF-PA cho NRS, và FA-FPR cho FRS. Đối thủ tốt nhất là phương pháp nhanh nhất trong số chạy xong cả 5 fold. Nếu không đối thủ nào chạy xong, tỉ số ghi là > 900/ours (cận dưới).

"Dùng được" nghĩa là KD-PA+B3 chạy xong đủ 5 fold và cấu hình không suy biến (|R| > 0): có 67/76 cấu hình.

| | NRS | FRS |
|---|---|---|
| Cấu hình dùng được | 43 (12 bộ) | 24 (12 bộ) |
| ≥ 2× | 40 | 23 |
| ≥ 1× | 42 | 24 |
| < 1× | creditcard δ = 0,05: 0,78× (FNC 1,7 s so với 2,2 s) | — |
| Trung vị | 10,8× | 56,7× |
| Số tỉ số là cận dưới | 8 | 10 |

**Phần do cận (KD-PA+B3 so với KD-PA):**
- Trung vị: NRS 1,2×, FRS 1,1×.
- Lợi lớn khi d lớn, ví dụ NRS:
  - gas-drift (d = 128): 7,3–12,1×;
  - miniboone (d = 50): 12–13×;
  - creditcard: khoảng 3×.
- **Cận làm chậm 6–33% khi |R| = d và d nhỏ:** run-or-walk 0,78–0,80×; hls4ml-jets 0,67–0,94×; magic 0,80–0,91×; htru2, mammography, fried, electrical-grid 0,81–0,94×.
- Kết luận: **thông điệp 3b được xác nhận.**
  - Khi d nhỏ, tăng tốc so với đối thủ đến từ cây KD, còn cận là chi phí thêm nhỏ.
  - Khi d lớn, cận là nguồn tăng tốc chính.

## 3. Giới hạn thấy trong E1
- **vehicle-sensit (d = 100):**
  - Chỉ NRS δ = 0,05 chạy xong: KD-PA+B3 140 ± 21 s; mọi đối thủ quá 900 s hoặc memout (> 6,4×).
  - δ ≥ 0,10 và FRS: mọi phương pháp, kể cả của mình, đều quá 900 s.
  - Chạy 4 tiến trình song song có thể đã làm chậm; đề xuất đo lại một mình (mục 5).
- **jannis FRS:** mọi phương pháp quá 900 s.
- **Suy biến theo từng fold:** hls4ml-jets NRS δ = 0,20 có |R| = 0 ở cả 5 fold; δ = 0,15 có |R| = 0 ở fold 1. Cách xử lý chưa đăng ký trước, **chờ user quyết**.
- **FNC memout** (vượt 3e8 cặp) trên internet-firewall, run-or-walk, miniboone, vehicle-sensit, hls4ml-jets (δ nhỏ). Đây là giới hạn bộ nhớ của FNC, không phải "thua về thời gian".

## 4. Độ chính xác (SVM-RBF, trung bình 5 fold; chi tiết 3-NN trong csv)
- Reduct chính xác và FARA thường cho cùng độ chính xác, vì |R| bằng nhau hoặc bằng d.
- FARA cho |R| lớn hơn trên gas-drift (72–124 so với 52–108) và cho SVM thấp hơn (0,879–0,915 so với 0,894–0,937).
- **Ngoại lệ phải nêu:** trên jannis, FARA cho SVM cao hơn rõ (0,61–0,63 so với 0,54–0,59) với |R| tương đương. Bài không claim độ chính xác, nhưng phải báo con số này.
- house-16h δ = 0,20: một fold có |R| = 1, kéo SVM của reduct chính xác xuống 0,835 (FARA 0,858).

## 5. Việc tiếp
1. User quyết cách xử lý các cấu hình suy biến của hls4ml-jets.
2. Đo lại một mình (đã đăng ký ở giới hạn 8): một nhóm nhỏ cấu hình, gồm vehicle-sensit KD-PA ± B3 δ = 0,10–0,20.
3. Kiểm định: Friedman/Nemenyi trên các phương pháp; Wilcoxon cho từng cặp E với E+B3.
4. Hình tách tăng tốc theo d: phần do KD và phần do cận.
5. Cập nhật `manuscript_prep.md` mục 4b bằng các số ở đây, thay cho các số sơ bộ lấy từ log.

## 6. Đo lại một mình (pre-reg §11.2; xong 2026-10-01; file `remeasure_idle.jsonl`)
- **Nhiễu do chạy 4 tiến trình:** 20 đơn vị có thời gian E1 > 1 s. Tỉ lệ thời gian một mình / E1 có trung vị **0,90** (0,85–0,98), gần như đều ở mọi phương pháp: KD-PA ± B3, FNC ± B3, HASH. Vì vậy các tỉ số tăng tốc giữa phương pháp trong E1 không bị lệch đáng kể. Các trường hợp memout giữ nguyên.
- **Ngoại lệ vehicle-sensit (d = 100):**

  | δ | KD-PA+B3 một mình | KD-PA một mình | E1 |
  |---|---|---|---|
  | 0,10 | **443 s** | > 900 s | cả hai > 900 s |
  | 0,15 | **735 s** | > 900 s | cả hai > 900 s |
  | 0,20 | > 900 s | > 900 s | cả hai > 900 s |

  Chạy song song làm KD-PA+B3 trên bộ này chậm hơn ít nhất khoảng 2×, nhiều khả năng do tranh chấp băng thông bộ nhớ ở d lớn (giả thuyết, chưa kiểm). Báo cáo trong phân tích độ nhạy; bảng chính giữ số E1.

## 7. Chạy lại engine liệt kê cặp sau khi sửa O(P) (pre-reg §12; xong 2026-10-01 16:21)
- **Tính chính xác:** kiểm tra lại đạt 12/12. Số lượt đánh giá của các bản +B3 trùng hoàn toàn với trước khi sửa. K1 sau chạy lại: 332/332 nhóm, 1482 lần chạy.
- **HASH/CABIN/BUCKET/BF-PA +B3 so với bản không cận (δ = 0,10):**
  - 20 tổ hợp cả hai chạy xong: trung vị 1,03× (0,89–3,88); chậm hơn ở 7 tổ hợp, tất cả có m ≤ 16. Wilcoxon một phía p = 0,066.
  - gas-drift: chỉ bản +B3 chạy xong (347–495 s).
  - 7 tổ hợp chỉ bản không cận chạy xong vì bản +B3 vượt 32 GB: internet-firewall (HASH, CABIN, BUCKET), run-or-walk (CABIN, BUCKET, BF-PA), hls4ml-jets (HASH).
- **Kết luận cũ "cận làm chậm 0,39–0,68×" là SAI**, do cài O(P log P). Kết luận mới: cái giá là bộ nhớ Θ(P), không phải thời gian. KD-PA+B3 ≤ 2,0 GB footprint (≤ 0,6 GB RSS); footprint chỉ được ghi cho các dòng chạy theo §10.
- Bảng: `paper/manuscript/tab_engines.tex` (`make_tab_engines.py`).
