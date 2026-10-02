# E1: đăng ký trước lưới thực nghiệm đầy đủ — **KHOÁ 2026-09-29** (trước khi chạy bộ đầu tiên)

**User chốt 2026-09-29:**
- thêm khoảng 8–10 bộ vào commons;
- tiêu chí NRS + FRS-min (không có k-NRS);
- δ ∈ {0.05, 0.10, 0.15, 0.20};
- có FARA, báo độ lệch reduct.

Trần n = 10⁵ đã chốt. **FRS chốt ngày 2026-09-29** (xem mục 2). Lưới chỉ dùng **v1** (user chốt 2026-09-29). Danh sách bộ đã chốt ở mục 1 (14 bộ dùng được).

Theo quy tắc chốt từng bộ ([[feedback-finalize-per-dataset]]), **lưới phải khoá trước khi chạy bộ đầu tiên**. Sau khi khoá, chỉ được thêm bộ dữ liệu, không thêm cấu hình. Các mục **[CHỐT?]** cần user quyết.

## 0. Điều kiện vào E1
- **G1:** đạt (`g1/g1_report.md`).
- **G2:** đề xuất coi là đạt dựa trên bằng chứng sẵn có.
  - Quy tắc dung sai đã thống nhất trong code: bộ lọc nới rộng thêm WIN = 1e-9, sau đó mọi engine đều xác nhận bằng cùng một phép kiểm float64 d² ≤ δ².
  - Kết quả: K1 trùng khớp 11/11 cặp trong G1 và 28/28 cặp khi đối chiếu với code FNC gốc.
- **A2:** không đạt, bỏ khỏi thuật toán chính (`pilot_A2/a2_report.md`).

## 1. Bộ dữ liệu **[CHỐT?]**
Quy tắc (giữ như G1):
- mọi thuộc tính là số, mỗi thuộc tính có ≥ 100 giá trị khác nhau;
- N ≥ 10⁴;
- lấy từ commons, dùng cả `standardized/v1` và `v2` (A3).

**Danh sách đạt quy tắc (2026-09-29).** Lọc 2 vòng, danh sách ứng viên cố định trước khi tải; chi tiết ở `screen_datasets.log`.

| Nhóm | Bộ | Ghi chú |
|---|---|---|
| Đã có (G1) | gas-drift, miniboone, magic-telescope, jannis, creditcard | |
| ~~Chờ v2~~ LOẠI | eeg-eye-state, numerai28-6 | kiểm tra v2 (`v2_degeneracy_check.log`, tiêu chí đặt trước \|R\| ≥ 2 ở δ = 0,1): không đạt. numerai có \|R\| = 0 ở mọi δ, cả v1 lẫn v2 (nhãn gần ngẫu nhiên); eeg v2 có \|R\| = 0 từ δ ≥ 0,1. Nêu như giới hạn của NRS khi nhãn nhiễu. |
| Mới vào commons | electrical-grid-stability, mammography, htru2, house-16h, fried, run-or-walk, internet-firewall, vehicle-sensit, hls4ml-jets | n từ 10⁴ đến 8,3·10⁵ |

Tổng cộng **14 bộ dùng được**. **[CHỐT?]** Có dùng v2 trong lưới không: đề xuất không, chỉ dùng v1, vì v2 không cứu được bộ nào mà làm gấp đôi chi phí chạy.

**Loại theo quy tắc:**
- letter và covertype (giá trị nguyên hoặc cột nhị phân);
- vòng 1: pendigits, ailerons, sylva, pol, elevators, california, default-credit, electricity, volkert, otto, spoken-arabic, codrna, atlas-higgs;
- vòng 2: bitcoinheist, mozilla4, microaggregation2, news_channel, bank-marketing, 2dplanes, risk-level, college-scorecard.

**Không lọc được:**
- higgs và santander: file trên OpenML lỗi checksum;
- microsoft-ltr: tải quá chậm nên đã dừng; bộ này bị loại trước khi xem dữ liệu, nên không gây thiên lệch.

**Lưu ý cần báo thẳng trong bài:**
- fried là dữ liệu tổng hợp;
- internet-firewall gồm giá trị đếm hoặc số cổng;
- run-or-walk và eeg có tương quan thời gian (random CV cho kết quả lạc quan).

## 2. Lưới cấu hình **[CHỐT?]**

| Trục | Giá trị đề xuất |
|---|---|
| Tiêu chí | **NRS** (Euclid, δ cố định). **FRS-min**: Dubois–Prade với implicator Kleene–Dienes, t-norm min, R_a = max(0, 1 − \|Δ_a\|/σ_a), quyết định rõ, nên μ_POS(x) = min(1, ne^∞_σ(x)) (L83, L85). k-NRS và FRS-Gauss không đưa vào E1 (FRS-Gauss chỉ nêu trong lý thuyết) |
| σ (FRS) | (i) σ_a = 0.25 cho mọi a trên dữ liệu v1 [0, 1], đúng thiết lập của Qian 2015; (ii) σ_a = std(a) tính trên fold huấn luyện (Jensen & Shen 2009) |
| Quy tắc dừng FRS | dừng khi lợi tốt nhất ≤ 1e-9 (lợi là số thực; dung sai cố định, dùng chung cho mọi phương pháp) |
| Đối thủ FRS | BF-DAR (brute force), **FA-FPR** (Qian 2015, bỏ các x có μ_POS = 1), KD-PA, KD-B3. PARA (L86) dùng greedy khác (Sig2), nên chỉ nêu trong Related Work |
| δ (NRS) | {0.05, 0.10, 0.15, 0.20} |
| n | toàn bộ tập, trần **10⁵** (user chốt 2026-09-29): bộ lớn hơn lấy mẫu phân tầng seed 0 một lần, lưu cố định, rồi mới chia fold |
| Engine × cận (NRS) | {BF, FNC/HNAR, CABIN, HASH, BUCKET, KD} × {không có B3, có B3} |
| Engine × cận (FRS) | {BF, FA-FPR, KD} × {không có B3, có B3} |
| Accelerator ngoài dòng | positive approximation (đã có trong mọi engine); **FARA** (`fara.py`, L71 Alg. 2, cùng engine KD). Ba điểm bài để mở được chốt trước: t = 1; k tăng mỗi vòng; C_ρ: γ(A) ≥ γ(AT). Báo \|R\|, γ, độ trùng với reduct chính xác, thời gian, độ chính xác, cờ stall |
| Fold | 5-fold StratifiedKFold, seed 0 (`eval_harness/protocol.py`) |
| Giới hạn | 900 s mỗi (fold, method); FNC dừng với MemOut khi vượt 3e8 cặp |

## 3. Tiết kiệm tính toán (thiết kế, không làm thay đổi số liệu)
Mọi phương pháp chính xác trả cùng một reduct (K1). Vì vậy:
- 3-NN và SVM-RBF chỉ đánh giá **một lần** cho mỗi (bộ, tiêu chí, δ, fold), rồi gán cho mọi phương pháp chính xác có reduct trùng.
- Code tự kiểm tra việc trùng reduct. Nếu không trùng thì đánh giá riêng và ghi cờ.
- Thời gian rút gọn vẫn đo riêng cho từng phương pháp, ở từng fold.
- SVM-RBF trên n ≥ 10⁵: dùng mặc định harness. Có thể rất chậm; nếu cần thì báo thời gian ước tính trước khi chạy.

## 4. Ghi nhận (một lần chạy duy nhất)
- **Mỗi fold:** acc_3nn, acc_svm_rbf, time_reduct_sec, và |R|. Tổng hợp thành mean ± std.
- **Kho reduct:** `results/reducts/<dataset>__<criterion>__<delta>__fold<k>__<method>.json`. Mỗi file gồm:
  - thứ tự thuộc tính và lợi của từng bước;
  - số lượt đánh giá và số ứng viên bị cận loại ở từng bước;
  - UB(a*)/gain(a*);
  - thời gian từng bước, maxrss, và phiên bản code.
- **File thô:** `e1_raw.csv` (append-only). Mỗi (bộ, tiêu chí) chạy xong thì ghi vào `e1_locked.txt`.

## 5. Phân tích (định trước)
- **Tốc độ:** Friedman + Nemenyi trên thời gian. Wilcoxon cho từng cặp E với E + B3.
- **Ablation cận:** tỉ số thời gian (E)/(E + B3) cho mỗi engine E. Đây là bằng chứng cho luận điểm "cận cộng dồn với mọi bộ tính".
- **Khả năng mở rộng:** đường cong thời gian theo n, trên 2–3 bộ lớn, n ∈ {10⁴, 3·10⁴, 10⁵, 3·10⁵}.
- **Giới hạn báo thẳng:**
  - các bộ có lớp thiểu số cực nhỏ (creditcard), nơi FNC thắng;
  - các bộ có \|R\| ≈ d (Magic), nơi cận không cắt được.

## 6. Kiểm tra nhanh FARA (2026-09-29, 4 mẫu G1, δ = 0,1; không phải kết quả E1)
- **Magic:** FARA cho reduct trùng greedy chính xác và nhanh hơn một chút (0,9 s so với 1,1 s).
- **jannis, MiniBooNE, creditcard:** reduct khác. \|R\| lớn hơn: 14/24/16 so với 12/14/10. FARA cũng chậm hơn KD-B3 từ 1,4 đến 4 lần.
- **Phát hiện:** greedy chính xác dừng khi lợi bằng 0, nên không phải lúc nào cũng đạt γ(AT).
  - MiniBooNE: 551 so với 553. creditcard: 19997 so với 20000.
  - FARA cũng bị "stall" trên hai bộ này.
  - Cần báo trong bài: quy tắc dừng khi lợi bằng 0 là quy tắc chuẩn, và cả greedy lẫn FARA đều không bảo đảm đạt γ(AT).

## 7. Quy trình chạy (khoá 2026-09-29, user chốt)
- **Quy tắc suy biến theo từng tiêu chí.** Một bộ được giữ cho tiêu chí c nếu \|R\| ≥ 2 theo c, trên toàn bảng đã cắt trần (NRS: δ = 0,1; FRS: σ = 0,25). Kết quả ở `degeneracy_screen.log` và `degeneracy_screen_frs.log`.
  - NRS có 12 bộ; electrical-grid và fried bị loại vì \|R\| = 0.
- **Đơn vị chạy** = (bộ, tiêu chí, tham số, fold, phương pháp).
  - Phép rút gọn chạy trên phần huấn luyện của fold, chia bằng StratifiedKFold(5, shuffle=True, random_state=0), đúng cách chia của harness.
  - Mỗi đơn vị chạy trong một tiến trình riêng, với 1 luồng BLAS và giới hạn 900 s.
  - 4 đơn vị chạy song song. Bài phải ghi rõ điều này.
  - Mỗi lúc chỉ chạy tối đa một đơn vị họ FNC, để tránh hết bộ nhớ.
- **Quy tắc bỏ qua.** Nếu một phương pháp quá giờ hoặc hết bộ nhớ ở fold 1 của một (bộ, tiêu chí, tham số), thì các fold 2–5 được ghi `skipped-after-fold1` và không chạy.
- **Cận trên mọi engine.** Cận chỉ cần một kẻ thù bất kỳ trong δ làm nhân chứng, nên vẫn hợp lệ với mọi nhân chứng.
  - BF, CABIN, BUCKET, FNC và KD lấy kẻ thù gần nhất; HASH lấy nhân chứng đầu tiên tìm được.
  - Kiểm tra tính chính xác (`verify_nrs_e1.log`): 12 phương pháp NRS chính xác trùng khớp trên 12/12 cấu hình.
  - Với FRS (`verify_frs.log`): 6 phương pháp trùng khớp trên 10/10 cấu hình.
- **K1 trong E1.** Mọi phương pháp chính xác đã chạy xong phải cho cùng R. Nếu lệch: ghi cờ, báo user, **không tự chạy lại**.
- **Độ chính xác** chạy tách riêng và song song, bằng `eval_harness/protocol.py::evaluate_reduct`.
  - Reduct tham chiếu cho mỗi fold là reduct chung của nhóm phương pháp chính xác. FARA được đánh giá bằng reduct riêng của nó.
  - Nếu \|R\| = 0 thì không đánh giá, và ghi cờ.
- **Đầu ra:** `e1_raw.jsonl`, `results/reducts/*.json` (có ghi nhận từng bước), `e1_locked.txt`, `run_e1.log`.

## 8. Sửa đổi lưới — lưới gọn (2026-09-29 16:36, user chốt; chưa có bộ NRS nào khoá)
**Lý do:** ước lượng 2–3 ngày, phần lớn là các đối thủ chậm chạy hết 900 s ở mọi δ, trong khi G1 đã cho thấy chúng quá giờ ở n lớn. Lưới gọn vẫn đủ bằng chứng cho ba luận điểm: KD-B3 so với đối thủ tốt nhất; cận cộng dồn với mọi engine; kết quả đúng trên nhiều δ và nhiều bộ.

**Lưới gọn:**
- **NRS, cả 4 δ:** KD-PA, KD-PA+B3, FNC, FNC+B3, HASH, FARA.
- **NRS, chỉ δ = 0,10 (ablation engine):** thêm HASH+B3, CABIN ± B3, BUCKET ± B3, BF-PA ± B3.
- **FRS, 2 σ:** KD-PA ± B3, FA-FPR ± B3. Bỏ BF-DAR, vì FA-FPR đã là BF cộng positive approximation (Qian 2015).
- Fold, giới hạn giờ, quy tắc bỏ qua và song song giữ nguyên như mục 7.

**Dữ liệu đã chạy trước khi đổi:**
- electrical-grid (chỉ FRS, đã khoá lúc 16:33, gồm cả BF-DAR) và mammography (156 dòng, chưa khoá) được giữ trong `e1_raw.jsonl`.
- Các dòng ngoài lưới gọn không bị xoá, nhưng bảng chính và kiểm định chỉ dùng các dòng thuộc lưới gọn, để mọi bộ dữ liệu được so trên cùng một tập phương pháp.

## 9. Sửa lỗi hiệu năng FNC+B3 (2026-09-29 19:15, user chốt; ghi trước khi chạy lại)
- **Phát hiện:** trên house-16h, FNC+B3 chậm hơn FNC từ 2 đến 4 lần ở mọi δ. Nguyên nhân là bước tìm nhân chứng dùng `np.lexsort` trên toàn bộ danh sách cặp mỗi bước, tốn O(P log P). Một lần đánh giá của FNC chỉ tốn O(P).
- **Sửa:** tìm nhân chứng bằng `np.minimum.at` cộng một lần lọc, tốn O(P). Trên house-16h (fold 1, δ = 0,10), FNC+B3 giảm từ 158 s xuống 56 s (FNC: 55 s). Reduct không đổi.
  - Kiểm tra tính chính xác được chạy lại: `verify_nrs_e1_fix.log`.
- **Áp dụng:**
  - Mọi dòng FNC+B3 cũ được liệt kê trong `e1_superseded.jsonl`. Chúng **không bị xoá** khỏi `e1_raw.jsonl`, nhưng bị loại khỏi phân tích.
  - FNC+B3 được chạy lại trên các bộ đã khoá bằng `run_e1.py rerun FNC+B3 <bộ>`.
  - Các phương pháp khác không bị động đến.
- **Nguyên tắc:** tối ưu đều cho mọi phương pháp áp dụng cho cả phần của mình. Cách cài chậm sẽ làm bài đánh giá thấp luận điểm "cận cộng dồn với FNC".

## 10. Giới hạn bộ nhớ (2026-09-30 ~09:40, user chốt; ghi trước khi chạy tiếp)
**Phát hiện:** máy sập 2 lần (06:58 kernel panic do watchdog, 07:32 shutdown stall), cả hai lần ngay sau các sự kiện Jetsam.
- Theo `/Library/Logs/DiagnosticReports/JetsamEvent-2026-09-30-*.ips`, 4 tiến trình lớn nhất đều là worker Python của E1.
- RAM đỉnh đã ghi trong `e1_raw.jsonl`: HASH/BF-PA/BUCKET/CABIN +B3 25–29 GB; FNC+B3 15,5 GB; FNC 9,4 GB; KD-PA+B3 0,6 GB. Máy có 48 GB.
- Nguyên nhân: phần thu nhân chứng của các engine liệt kê cặp (mixin Collect) ghi mọi cặp trong bán kính δ, và 4 tiến trình như vậy chạy cùng lúc.

**Sửa đổi (chỉ áp cho 5 bộ chưa khoá: run-or-walk, vehicle-sensit, miniboone, creditcard, hls4ml-jets):**
1. Các phương pháp HEAVY chạy từng cái một: họ FNC (FNC, FNC+B3) và HASH+B3, CABIN+B3, BUCKET+B3, BF-PA+B3. Các phương pháp khác vẫn chạy song song, tổng 4 worker như cũ.
2. Tiến trình cha theo dõi mỗi 2 s. Nếu RSS của tiến trình con > 32 GB, hoặc áp lực bộ nhớ hệ thống ở mức critical (`kern.memorystatus_vm_pressure_level` ≥ 4), thì giết tiến trình con và ghi `status = memout`, có `note`. Xử lý như timeout: bỏ qua fold 2–5.
   - Trần 32 GB cao hơn mức đỉnh đã thấy trên mọi bộ đã khoá (29,3 GB), nên luật nhất quán với các kết quả đã khoá.
   - Trần thấp hơn mức 40 GB em đề xuất ban đầu: 32 GB + 3 worker nhẹ (≤ 5 GB mỗi cái) vẫn nằm trong 48 GB.
3. Không chạy lại bộ đã khoá. run-or-walk chưa khoá: các đơn vị đã có trong `e1_raw.jsonl` được giữ, phần còn lại chạy theo luật mới.

**Hệ quả khi báo cáo:**
- Thời gian của các phương pháp HEAVY trên 5 bộ này đo khi không có phương pháp HEAVY nào khác chạy song song. Phải ghi rõ trong bài, cùng với giới hạn 8 của manuscript_prep.
- Thêm cột RAM đỉnh (`maxrss_mb`) vào bảng kết quả.
- Code: `run_e1.py` (`run_capped`, `is_heavy`); bản trước lưu ở `run_e1.py.bak_mem_20260930`.
- **Bổ sung (09:23):** chia worker thành 1 luồng cho phương pháp HEAVY và 3 luồng cho phương pháp nhẹ (`run_units`); tổng vẫn 4. Trước đó các đơn vị HEAVY chiếm hết 4 luồng trong lúc chờ nhau, nên chỉ 1 tiến trình thật sự chạy. Chỉ đổi lịch chạy, không đổi phép đo.
- **Sửa (09:34), sau khi user thấy trên Activity Monitor: 42,9/48 GB đã dùng, 25 GB nén, swap 11 GB.**
  - Trần theo RSS **không có tác dụng**: CABIN+B3 (run-or-walk, δ = 0,10, fold 2) có RSS 6,5 GB nhưng phys_footprint 89 GB, trong đó 83 GB là bộ nhớ nén (`top`). Em đã dừng runner; đơn vị đó không được ghi.
  - Trần nay đo **phys_footprint** (proc_pid_rusage, gồm cả phần nén; cùng con số Activity Monitor hiển thị), vẫn 32 GB. Mỗi dòng mới có thêm `peak_footprint_gb`, lấy mẫu mỗi 2 s.
  - **Đính chính câu ở điểm 2 phía trên:** "trần cao hơn mức đỉnh đã thấy trên các bộ đã khoá" là SAI. `maxrss_mb` của các dòng cũ là RSS, không tính phần nén. Footprint thật của HASH/CABIN/BUCKET/BF-PA +B3 trên các bộ đã khoá (ví dụ jannis) có thể vượt 32 GB mà vẫn chạy xong nhờ nén và swap.
  - Do đó trên 5 bộ còn lại, các engine này có thể bị `memout` ở những cấu hình tương tự nơi chúng từng chạy xong trên bộ đã khoá. Bài phải nêu rõ sự không đồng nhất này, và không so trực tiếp tỉ lệ memout giữa hai nhóm bộ.
  - Luật này không ảnh hưởng KD-PA ± B3, FA-FPR ± B3 và FARA (footprint < 1 GB).
- **run-or-walk, NRS δ = 0,10, CABIN+B3, fold 1:**
  - Dòng `status = error`, stderr rỗng (dòng 1609 của `e1_raw.jsonl`) là do tiến trình bị macOS giết vì thiếu bộ nhớ trong lần sập máy đầu. User xác nhận 2026-09-30.
  - Dòng này được liệt kê trong `e1_superseded.jsonl`, không xoá.
  - Đơn vị này sẽ được chạy lại dưới luật mục 10 **sau khi E1 xong**, không chạy song song với runner chính, để không có 2 đơn vị HEAVY cùng lúc.
- **Đo lại toàn bộ run-or-walk (user đồng ý chạy lại, 2026-09-30 10:35):**
  - Mọi dòng run-or-walk đo trước luật mục 10 (code ≠ `41a9cd109ecd`: 117 dòng, gồm cả dòng bỏ qua sau fold 1) được liệt kê trong `e1_superseded.jsonl`, không xoá.
  - Lý do: các dòng này đo khi nhiều đơn vị HEAVY chạy cùng lúc, có swap, và qua 2 lần sập máy. Ví dụ HASH+B3 δ = 0,10: fold 2 timeout 900 s, trong khi fold 3–5 chỉ 485–557 s.
  - Sẽ đo lại **sau khi runner E1 chính dừng**, dưới luật mục 10, bằng `rerun` cho từng phương pháp trong lưới. Như vậy cả bộ đo theo một giao thức.
  - Trước khi chạy lại: chuyển `rerun_method` sang `run_units` (1 luồng HEAVY + 3 luồng nhẹ).
  - Các dòng MemOut theo số cặp của FNC là tất định. Chúng vẫn được đo lại cho đồng nhất, và kết quả dự kiến không đổi.
- **Tạm dừng E1 (2026-09-30 ~14:00, user yêu cầu):**
  - Một chương trình của dự án khác chạy trên máy từ 13:50 (1 tiến trình Python, khoảng 1 nhân).
  - Để thời gian đo không bị nhiễm, toàn bộ các dòng miniboone đã có (bắt đầu 13:15) được liệt kê superseded, và miniboone được đo lại từ đầu khi chạy tiếp.
  - Chạy tiếp khi máy không có việc nặng khác.
- **Lỗi đánh dấu superseded cho miniboone (phát hiện 2026-10-01 04:10, đã sửa):**
  - 23 mục superseded của miniboone khớp theo (dataset, criterion, param, fold, method, code). Các dòng đo lại có **cùng mã code**, nên cũng bị che.
  - Hệ quả trong lúc chạy: với các khoá bị che, luật bỏ qua sau fold 1 không kích hoạt. Fold 2–5 của một số cấu hình timeout hoặc memout ở fold 1 (NRS δ = 0,05 và 0,10: HASH, BF-PA, BUCKET, CABIN, CABIN+B3, HASH+B3, FNC ±B3) **đã thật sự chạy**, và cũng timeout hoặc memout.
  - Kết quả vẫn tương đương với "bỏ qua" (đều > 900 s hoặc memout), chỉ tốn thêm thời gian máy. Không có dòng trùng.
  - Sửa: mục superseded có thêm trường `time` để khớp đúng dòng cũ (`superseded()` / `is_superseded()` trong `run_e1.py`). Bản cũ lưu ở `e1_superseded.jsonl.bak_20261001`.
  - Sau khi sửa: 195 dòng miniboone hợp lệ, 0 khoá trùng, không thiếu đơn vị nào (ngoài run-or-walk đang đo lại).

## 11. Quyết định sau E1 (2026-10-01, user chốt; ghi trước khi phân tích và đo lại)
1. **Cấu hình suy biến theo fold** (exact |R| = 0 ở ít nhất một fold; E1: hls4ml-jets NRS δ = 0,15 và 0,20):
   - vẫn báo trong bảng, gắn nhãn "degenerate (|R| = 0)";
   - loại khỏi mọi thống kê tăng tốc và kiểm định;
   - không đánh giá độ chính xác (eval đã ghi not-evaluated).
2. **Đo lại một mình (giới hạn 8, phân tích độ nhạy; KHÔNG thay số E1):**
   - Script `remeasure_idle.py`, chạy tuần tự từng đơn vị một (1 tiến trình, 1 luồng BLAS), cùng trần 900 s và 32 GB, fold 1. Đầu ra `remeasure_idle.jsonl`, tách khỏi `e1_raw.jsonl`.
   - (a) Nhiễu do chạy song song: NRS δ = 0,10 với KD-PA+B3, KD-PA, FNC, FNC+B3 trên gas-drift, creditcard, house-16h, magic-telescope, jannis, miniboone; thêm HASH trên house-16h và magic-telescope.
   - (b) vehicle-sensit: KD-PA+B3 và KD-PA ở δ = 0,10, 0,15 và 0,20.
   - Báo cáo: tỉ số thời gian (một mình so với E1) theo từng đơn vị. Với vehicle-sensit, ghi "run alone" trong phân tích độ nhạy; bảng chính giữ nguyên số E1.
3. **Kiểm định:**
   - Friedman + Nemenyi trên thời gian trung bình theo cấu hình (các cấu hình dùng được). Timeout và memout được xếp hạng như 900 s; hạng hoà khi trùng.
   - Wilcoxon một phía cho từng cặp E với E+B3 trên các cấu hình mà cả hai chạy xong đủ 5 fold.

## 12. Sửa hiệu năng bước nhân chứng của các engine liệt kê cặp (ghi 2026-10-01, TRƯỚC khi sửa và chạy lại)
- **Phát hiện** (khi viết Section 4 của bản thảo): `Collect.witness` trong `engines_e1.py`, dùng cho HASH+B3, CABIN+B3, BUCKET+B3 và BF-PA+B3, rút các cặp đã lưu về một nhân chứng cho mỗi đối tượng bằng `np.lexsort`, tốn O(P log P).
  - Đây đúng là lỗi đã sửa cho FNC+B3 ở mục 9; FNC+B3 nay dùng `np.minimum.at` + lọc, tốn O(P).
  - Như vậy bốn engine kia chưa được tối ưu ngang FNC+B3, trái với nguyên tắc ở mục 9.
- **Sửa:** dùng cùng cách O(P) như FNC+B3 (cặp có d² nhỏ nhất cho mỗi đối tượng; khi hoà lấy chỉ số kẻ thù nhỏ nhất, để giữ đúng thứ tự cũ của `lexsort((e, d2, q))`).
  - Trước khi chạy lại phải kiểm tra lại tính chính xác, như mục 9.
- **Áp dụng:**
  - mọi dòng cũ của 4 phương pháp này (lưới gọn: chỉ NRS δ = 0,10) được liệt kê superseded, không xoá;
  - chạy lại 4 phương pháp trên 14 bộ bằng `run_e1.py rerun <phương pháp> <bộ>`, dùng luật mục 10 (1 luồng HEAVY, trần 32 GB footprint), **sau khi** đo lại một mình (mục 11) xong;
  - các phương pháp khác không bị động đến.
- **Lý do:** bài kết luận "cận làm chậm các engine liệt kê cặp". Kết luận đó phải dựa trên cách cài tối ưu ngang nhau.

## 13. Phân tích bổ sung: co giãn theo số đối tượng n (ghi 2026-10-01, TRƯỚC khi chạy; user đồng ý)
- **Mục đích:** đo thời gian rút gọn theo n để đối chiếu với Proposition 5 (độ phức tạp). Đây là phân tích bổ sung, ngoài lưới E1; KHÔNG thay số nào của E1. Báo cáo ở Appendix, có ghi rõ là bổ sung.
- **Dữ liệu:** hls4ml-jets (m = 16) và miniboone (m = 50), bảng đã chặn 10⁵ của E1 (`data/<slug>.npz`).
  - n ∈ {12 500, 25 000, 50 000, 100 000}. Với n < 10⁵: một mẫu con phân tầng (seed 0) của bảng; với n = 10⁵: bảng nguyên vẹn (để fold 1 trùng E1).
  - Rút gọn trên phần huấn luyện của fold 1 (StratifiedKFold 5, shuffle, seed 0, như E1), tức n_train = 0,8 n.
- **Tiêu chí và phương pháp:**
  - NRS δ = 0,10: KD-PA+B3, KD-PA, FNC, HASH;
  - FRS σ = 0,25: KD-PA+B3, KD-PA, FA-FPR.
- **Chạy:** tuần tự, mỗi đơn vị một tiến trình, 1 luồng BLAS, trần 900 s và 32 GB footprint như E1 (§10), máy không chạy việc khác.
  - Luật bỏ qua: nếu (bộ, tiêu chí, phương pháp) timeout hoặc memout ở một n thì không chạy các n lớn hơn, ghi "skipped-larger-n".
- **Kiểm tra:** các phương pháp chính xác phải trả cùng reduct có thứ tự ở mỗi n. Thời gian ở n = 10⁵ so với E1 fold 1 (E1 chạy 4 tiến trình song song).
- **Đầu ra:** `scale_n.py` → `scale_n.jsonl`; hình `make_fig_scale.py` → `fig_scale.tex` (log–log, thời gian theo n).

## 14. Phân tích bổ sung: co giãn theo số thuộc tính m trên cùng một bảng (ghi 2026-10-02, TRƯỚC khi chạy; user đồng ý)
- **Mục đích:** tách ảnh hưởng của m lên lợi ích của cận khi giữ nguyên dữ liệu và n (phép so giữa các dataset ở 5.4 lẫn m với các yếu tố khác). Phân tích bổ sung, KHÔNG thay số E1; báo cáo ở Section 5.4 và Appendix.
- **Dữ liệu:** gas-drift (m = 128, n = 13 910), bảng E1 (`data/gas-drift.npz`). Phần huấn luyện của fold 1 (StratifiedKFold 5, shuffle, seed 0, như E1).
- **Tập thuộc tính:** m ∈ {16, 32, 64, 128}; với m < 128, m thuộc tính đầu của một hoán vị ngẫu nhiên (seed s ∈ {0, 1, 2}; các tập lồng nhau theo m với mỗi s). m = 128 là bảng đầy đủ, chạy một lần.
- **Tiêu chí và phương pháp:** NRS δ = 0,10 và FRS σ = 0,25; KD-PA và KD-PA+B3.
- **Chạy:** tuần tự, mỗi đơn vị một tiến trình, 1 luồng BLAS, giới hạn 900 s và 32 GB như E1 (§10), máy không chạy việc khác. Luật bỏ qua: nếu (tiêu chí, phương pháp, s) quá giờ hoặc hết bộ nhớ ở một m thì không chạy m lớn hơn, ghi "skipped-larger-m".
- **Ghi:** thời gian, |R|, số lượt đánh giá; kiểm tra KD-PA và KD-PA+B3 trả cùng reduct có thứ tự.
- **Báo cáo định trước:** tỉ số thời gian KD-PA / KD-PA+B3 và tỉ lệ lượt đánh giá còn lại theo m (trung bình hình học qua s); quá giờ vẽ như cận dưới 900/t.
- **Ghi chú thăm dò (không báo cáo):** thăm dò ngày 2026-10-02 trên dilbert (m = 2 000) cho thấy greedy FRS chọn toàn bộ thuộc tính và KD-PA+B3 quá giờ ở m lớn; Santander (m = 200) chỉ NRS chạy được, cận bỏ 26% lượt. Hai bộ này không đưa vào E1 hay §14.
- **Đầu ra:** `scale_m.py` → `scale_m.jsonl`; hình `make_fig_scale_m.py` → `fig_scale_m.tex`.
