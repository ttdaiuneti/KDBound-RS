# Schema output chuẩn — bộ đánh giá reduct dùng chung

Mọi script thực nghiệm (trong `baselines/` hoặc trong E1 của một paper) khi đánh giá
một tập rút gọn thuộc tính phải gọi `protocol.py::evaluate_reduct()` và ghi kết quả
theo schema này vào `raw_results.csv` của paper đó — **ghi ở mức per-fold, không lưu
sẵn mean/std**. Mean/std tính ở bước gộp bảng (E3), giữ đúng nguyên tắc "raw_results.csv
là nguồn sự thật duy nhất, không gõ tay/suy ngược số đã làm tròn" của quy trình 12-skill.

## Policy "single-run capture" cho E1 (bat buoc)

- Mot cau hinh thuc nghiem (dataset, method, seed) **chi can 1 lan chay E1** de thu
  du thong tin can thiet.
- Trong lan chay do, script **MUST** ghi du 5 dong fold voi day du cot bat buoc:
  `selected_features`, `n_features_reduced`, `acc_3nn`, `acc_svm_rbf`,
  `time_reduct_sec` (va `time_eval_sec` neu ghi).
- Mean/std accuracy cua 3-NN va SVM-RBF **MUST** duoc tinh tu chinh 5 dong fold da ghi,
  **MUST NOT** rerun chi de lay std.
- Dau vao danh gia **MUST** la du lieu `standardized/v<N>/data.csv`.

## Cột bắt buộc

| Cột | Ý nghĩa |
|---|---|
| `dataset` | slug dataset, khớp `datasets/<slug>/` |
| `method` | tên phương pháp rút gọn thuộc tính |
| `seed` | seed dùng để sinh 5-fold (StratifiedKFold, shuffle=True, random_state=seed) |
| `fold` | 1..5 |
| `n_features_original` | số thuộc tính của bản `standardized/v<N>/data.csv` (không tính trên raw) |
| `n_features_reduced` | \|R\| — số thuộc tính trong tập rút gọn |
| `reduction_ratio` | `n_features_reduced / n_features_original` |
| `selected_features` | chỉ số cột (0-based, tính trên `standardized/v<N>/data.csv`) của các thuộc tính được giữ lại trong `R`, nối bằng `\|` (vd `"0\|3\|7"`) — bắt buộc để reviewer/rerun sau này biết chính xác thuộc tính nào được chọn, không chỉ số lượng |
| `acc_3nn` | accuracy của fold này, phân lớp bằng KNN(n_neighbors=3), hyperparameter mặc định — KHÔNG tune |
| `acc_svm_rbf` | accuracy của fold này, phân lớp bằng SVC(kernel="rbf"), hyperparameter mặc định sklearn (C, gamma mặc định) — KHÔNG tune |
| `time_reduct_sec` | thời gian CHỈ của thuật toán rút gọn thuộc tính cho fold này — KHÔNG tính thời gian huấn luyện/đánh giá 3-NN hay SVM-RBF |
| `time_eval_sec` | (tuỳ chọn, khuyến nghị ghi) thời gian huấn luyện + đánh giá cả 3-NN và SVM-RBF cho fold này — tách riêng khỏi `time_reduct_sec` theo bài học MHG (gộp chung reduct-time và eval-time là lỗi so tốc độ kinh điển) |
| `git_commit_of_method` | commit hash của `baselines/<algo>/v<N>/impl.py` đã dùng (hoặc "own" nếu là phương pháp mới của chính paper) |

## Quyết định đã chốt (không tự đổi khi chưa hỏi lại)

- **3-NN / SVM-RBF dùng hyperparameter mặc định, không nested-CV tuning.** Vai trò của
  hai classifier này là "probe chất lượng reduct" — thước đo nhất quán để so sánh các
  phương pháp rút gọn với nhau — không phải baseline cạnh tranh. Nếu một paper cần so
  sánh với một SVM đã tune như một baseline thật sự, đó là một baseline riêng trong
  `baselines/`, không phải một phần của `eval_harness`.
- **`time_reduct_sec` chỉ đo thuật toán rút gọn**, vì đây là con số cạnh tranh có ý
  nghĩa giữa các phương pháp; thời gian huấn luyện/đánh giá classifier là chi phí chung
  cho mọi phương pháp nên không dùng để so sánh tốc độ.
- **Fold không lưu index tĩnh** — cùng `seed` → cùng fold cho mọi phương pháp được so
  sánh, nhờ dùng chung `StratifiedKFold(shuffle=True, random_state=seed)` trong
  `protocol.py`. Điều này tương thích với cơ chế multi-seed rerun đã có trong các paper
  (vd `run_main_block_seeded.py` của ASC-pp2).
