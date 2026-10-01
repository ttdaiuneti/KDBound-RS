"""Bo danh gia reduct dung chung cho research-commons.

Xem schema.md cho dinh nghia day du cac cot output va cac quyet dinh thiet ke
(hyperparameter mac dinh, khong tune; time_reduct_sec chi do thuat toan rut gon).

Cach dung tu mot paper (E1 experiment-pipeline):

    from protocol import evaluate_reduct

    rows = evaluate_reduct(
        X=X_standardized, y=y, reduct_fn=my_reduction_algorithm,
        dataset="iris", method="deg-gb", seed=0,
    )
    # rows la list[dict] theo dung schema.md, moi dict = 1 fold
    # append vao raw_results.csv cua paper
"""

import time
from statistics import mean, pstdev

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def evaluate_reduct(X, y, reduct_fn, dataset, method, seed, git_commit_of_method="own"):
    """Chay 5-fold CV, danh gia reduct_fn bang 3-NN va SVM-RBF (hyperparameter mac dinh).

    reduct_fn(X_train, y_train) -> (X_train_reduced, transform_fn, selected_indices)
        Phai tra ve ham/callable de ap dung cung phep rut gon len X_test, tranh leakage:
        transform_fn(X_test) -> X_test_reduced.
        selected_indices: chi so cot (0-based, ung voi X_train goc/da chuan hoa) cua
        cac thuoc tinh duoc giu lai, thu tu tuy y cua thuat toan.

    Tra ve: list[dict], moi dict la mot dong theo schema.md (mot fold).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n_features_original = X.shape[1]
    rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        t0 = time.perf_counter()
        X_train_reduced, transform_fn, selected_indices = reduct_fn(X_train, y_train)
        time_reduct_sec = time.perf_counter() - t0
        X_test_reduced = transform_fn(X_test)

        n_features_reduced = X_train_reduced.shape[1]

        t0 = time.perf_counter()
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_reduced, y_train)
        acc_3nn = knn.score(X_test_reduced, y_test)

        svm = SVC(kernel="rbf")
        svm.fit(X_train_reduced, y_train)
        acc_svm_rbf = svm.score(X_test_reduced, y_test)
        time_eval_sec = time.perf_counter() - t0

        rows.append({
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "fold": fold,
            "n_features_original": n_features_original,
            "n_features_reduced": n_features_reduced,
            "reduction_ratio": n_features_reduced / n_features_original,
            "selected_features": "|".join(str(i) for i in selected_indices),
            "acc_3nn": acc_3nn,
            "acc_svm_rbf": acc_svm_rbf,
            "time_reduct_sec": time_reduct_sec,
            "time_eval_sec": time_eval_sec,
            "git_commit_of_method": git_commit_of_method,
        })

    return rows


def summarize_5fold_rows(rows):
    """Tong hop mean/std tu rows da ghi, KHONG can rerun.

    Dung cho chinh sach single-run capture:
    - E1 ghi raw per-fold mot lan.
    - E2/E3 chi tong hop lai tu rows co san.

    Tra ve dict gom mean/std cho accuracy, time, va kich thuoc reduct.
    Su dung population std (`statistics.pstdev`, tuong duong np.std ddof=0).
    """
    if not rows:
        raise ValueError("rows must be non-empty")

    acc_3nn_vals = [r["acc_3nn"] for r in rows]
    acc_svm_vals = [r["acc_svm_rbf"] for r in rows]
    reduct_time_vals = [r["time_reduct_sec"] for r in rows]
    eval_time_vals = [r["time_eval_sec"] for r in rows if "time_eval_sec" in r]
    reduct_size_vals = [r["n_features_reduced"] for r in rows]
    ratio_vals = [r["reduction_ratio"] for r in rows]

    summary = {
        "n_folds": len(rows),
        "acc_3nn_mean": mean(acc_3nn_vals),
        "acc_3nn_std": pstdev(acc_3nn_vals),
        "acc_svm_rbf_mean": mean(acc_svm_vals),
        "acc_svm_rbf_std": pstdev(acc_svm_vals),
        "time_reduct_sec_mean": mean(reduct_time_vals),
        "time_reduct_sec_std": pstdev(reduct_time_vals),
        "n_features_reduced_mean": mean(reduct_size_vals),
        "n_features_reduced_std": pstdev(reduct_size_vals),
        "reduction_ratio_mean": mean(ratio_vals),
        "reduction_ratio_std": pstdev(ratio_vals),
    }

    if eval_time_vals:
        summary["time_eval_sec_mean"] = mean(eval_time_vals)
        summary["time_eval_sec_std"] = pstdev(eval_time_vals)

    return summary
