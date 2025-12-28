import pandas as pd
import numpy as np
from scipy.optimize import minimize

SEED = 322
np.random.seed(SEED)


def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    return train, test


def create_submission(predictions):
    submission = predictions[["row_id", "price_p05", "price_p95"]].copy()
    submission.to_csv("results/submission.csv", index=False)
    print("Submission сохранен: submission.csv")


def calc_iou(y_true_p05, y_true_p95, y_pred_p05, y_pred_p95):
    epsilon = 1e-6
    y_pred_p05 = np.maximum(y_pred_p05, 0)
    y_pred_p95 = np.maximum(y_pred_p95, 0)
    y_pred_p05 = np.minimum(y_pred_p05, y_pred_p95)
    y_pred_p95 = np.maximum(y_pred_p05, y_pred_p95)
    width_true = np.maximum(y_true_p95 - y_true_p05, epsilon)
    width_pred = np.maximum(y_pred_p95 - y_pred_p05, epsilon)
    intersection = np.maximum(
        0, np.minimum(y_true_p95, y_pred_p95) - np.maximum(y_true_p05, y_pred_p05)
    )
    union = width_true + width_pred - intersection
    return np.mean(intersection / (union + epsilon))


def calculate_volatility(train):
    """Рассчитывает волатильность для каждого продукта"""
    vol = train.groupby("product_id")["price_p05"].std().reset_index()
    vol.columns = ["product_id", "volatility"]
    vol_median = vol["volatility"].median()
    if vol_median > 0:
        vol["vol_norm"] = vol["volatility"] / vol_median
    else:
        vol["vol_norm"] = 1.0
    # Ограничиваем диапазон
    vol["vol_norm"] = np.clip(vol["vol_norm"], 0.8, 1.6)
    return vol[["product_id", "vol_norm"]]


def apply_dynamic_interval_widening(predictions, train, test):
    """Применяет dynamic interval widening на основе волатильности"""
    vol = calculate_volatility(train)

    # Объединяем с предсказаниями через test для получения product_id
    predictions_with_prod = predictions.merge(
        test[["row_id", "product_id"]], on="row_id", how="left"
    )
    predictions_with_prod = predictions_with_prod.merge(
        vol, on="product_id", how="left"
    )
    predictions_with_prod["vol_norm"] = predictions_with_prod["vol_norm"].fillna(1.0)

    # Применяем адаптивное расширение интервала
    k = predictions_with_prod["vol_norm"].values
    # Для стабильных продуктов (k < 1) - сужаем интервал
    # Для волатильных продуктов (k > 1) - расширяем интервал
    alpha = 0.02  # коэффициент расширения

    # Сохраняем центр интервала
    center = (
        predictions_with_prod["price_p05"] + predictions_with_prod["price_p95"]
    ) / 2
    width = predictions_with_prod["price_p95"] - predictions_with_prod["price_p05"]

    # Адаптивно изменяем ширину
    new_width = width * (1 + alpha * (k - 1))

    # Пересчитываем границы
    predictions_with_prod["price_p05"] = center - new_width / 2
    predictions_with_prod["price_p95"] = center + new_width / 2

    # Убеждаемся, что интервал валиден
    predictions_with_prod["price_p05"] = np.maximum(
        predictions_with_prod["price_p05"], 0
    )
    predictions_with_prod["price_p95"] = np.maximum(
        predictions_with_prod["price_p95"], predictions_with_prod["price_p05"]
    )

    return predictions_with_prod[["row_id", "price_p05", "price_p95"]]


def get_model_6_opt_pred(train, test, known_products):
    """Model 6 с оптимизированными параметрами"""
    q_low, q_high = 0.18, 0.62
    n_days, w = 21, 0.45  # оптимизировано
    new_q_low, new_q_high = 0.27, 0.66  # оптимизировано

    last_date = train["dt"].max()
    recent = train[train["dt"] >= last_date - pd.Timedelta(days=n_days)]

    all_stats = (
        train.groupby("product_id")
        .agg(
            {
                "price_p05": lambda x: x.quantile(q_low),
                "price_p95": lambda x: x.quantile(q_high),
            }
        )
        .reset_index()
    )
    all_stats.columns = ["product_id", "p05_all", "p95_all"]

    recent_stats = (
        recent.groupby("product_id")
        .agg(
            {
                "price_p05": lambda x: x.quantile(q_low),
                "price_p95": lambda x: x.quantile(q_high),
            }
        )
        .reset_index()
    )
    recent_stats.columns = ["product_id", "p05_recent", "p95_recent"]

    prod_stats = all_stats.merge(recent_stats, on="product_id", how="left")
    prod_stats["p05_recent"] = prod_stats["p05_recent"].fillna(prod_stats["p05_all"])
    prod_stats["p95_recent"] = prod_stats["p95_recent"].fillna(prod_stats["p95_all"])
    prod_stats["p05"] = w * prod_stats["p05_recent"] + (1 - w) * prod_stats["p05_all"]
    prod_stats["p95"] = w * prod_stats["p95_recent"] + (1 - w) * prod_stats["p95_all"]

    cat_stats = (
        train.groupby(["management_group_id", "first_category_id"])
        .agg(
            {
                "price_p05": lambda x: x.quantile(new_q_low),
                "price_p95": lambda x: x.quantile(new_q_high),
            }
        )
        .reset_index()
    )
    cat_stats.columns = [
        "management_group_id",
        "first_category_id",
        "p05_cat",
        "p95_cat",
    ]

    cat_stats_1 = (
        train.groupby(["first_category_id"])
        .agg(
            {
                "price_p05": lambda x: x.quantile(new_q_low),
                "price_p95": lambda x: x.quantile(new_q_high),
            }
        )
        .reset_index()
    )
    cat_stats_1.columns = ["first_category_id", "p05_c1", "p95_c1"]

    global_p05 = train["price_p05"].quantile(new_q_low)
    global_p95 = train["price_p95"].quantile(new_q_high)

    test_known = test[test["product_id"].isin(known_products)].merge(
        prod_stats[["product_id", "p05", "p95"]], on="product_id", how="left"
    )
    test_known = test_known.rename(columns={"p05": "price_p05", "p95": "price_p95"})

    test_new = test[~test["product_id"].isin(known_products)].copy()
    test_new = test_new.merge(
        cat_stats, on=["management_group_id", "first_category_id"], how="left"
    )
    test_new = test_new.merge(cat_stats_1, on="first_category_id", how="left")
    test_new["price_p05"] = (
        test_new["p05_cat"].fillna(test_new["p05_c1"]).fillna(global_p05)
    )
    test_new["price_p95"] = (
        test_new["p95_cat"].fillna(test_new["p95_c1"]).fillna(global_p95)
    )

    predictions = pd.concat(
        [
            test_known[["row_id", "price_p05", "price_p95"]],
            test_new[["row_id", "price_p05", "price_p95"]],
        ]
    ).sort_values("row_id")

    return predictions["price_p05"].values, predictions["price_p95"].values


def get_model_20_pred(train, test, known_products):
    """Model 20: оптимизированные параметры + activity_flag"""
    q_low, q_high = 0.18, 0.62
    n_days = 21
    w_recent = 0.4  # оптимизировано
    w_activity = 0.4  # оптимизировано
    new_q_low, new_q_high = 0.27, 0.66

    last_date = train["dt"].max()
    recent = train[train["dt"] >= last_date - pd.Timedelta(days=n_days)]

    all_stats = (
        train.groupby("product_id")
        .agg(
            {
                "price_p05": lambda x: x.quantile(q_low),
                "price_p95": lambda x: x.quantile(q_high),
            }
        )
        .reset_index()
    )
    all_stats.columns = ["product_id", "p05_all", "p95_all"]

    recent_stats = (
        recent.groupby("product_id")
        .agg(
            {
                "price_p05": lambda x: x.quantile(q_low),
                "price_p95": lambda x: x.quantile(q_high),
            }
        )
        .reset_index()
    )
    recent_stats.columns = ["product_id", "p05_recent", "p95_recent"]

    prod_stats = all_stats.merge(recent_stats, on="product_id", how="left")
    prod_stats["p05_recent"] = prod_stats["p05_recent"].fillna(prod_stats["p05_all"])
    prod_stats["p95_recent"] = prod_stats["p95_recent"].fillna(prod_stats["p95_all"])
    prod_stats["p05_base"] = (
        w_recent * prod_stats["p05_recent"] + (1 - w_recent) * prod_stats["p05_all"]
    )
    prod_stats["p95_base"] = (
        w_recent * prod_stats["p95_recent"] + (1 - w_recent) * prod_stats["p95_all"]
    )

    prod_act_recent = (
        recent.groupby(["product_id", "activity_flag"])
        .agg(
            {
                "price_p05": lambda x: x.quantile(q_low),
                "price_p95": lambda x: x.quantile(q_high),
            }
        )
        .reset_index()
    )
    prod_act_recent.columns = ["product_id", "activity_flag", "p05_act", "p95_act"]

    cat_stats = (
        train.groupby(["management_group_id", "first_category_id"])
        .agg(
            {
                "price_p05": lambda x: x.quantile(new_q_low),
                "price_p95": lambda x: x.quantile(new_q_high),
            }
        )
        .reset_index()
    )
    cat_stats.columns = [
        "management_group_id",
        "first_category_id",
        "p05_cat",
        "p95_cat",
    ]

    cat_stats_1 = (
        train.groupby(["first_category_id"])
        .agg(
            {
                "price_p05": lambda x: x.quantile(new_q_low),
                "price_p95": lambda x: x.quantile(new_q_high),
            }
        )
        .reset_index()
    )
    cat_stats_1.columns = ["first_category_id", "p05_c1", "p95_c1"]

    global_p05 = train["price_p05"].quantile(new_q_low)
    global_p95 = train["price_p95"].quantile(new_q_high)

    test_known = test[test["product_id"].isin(known_products)].copy()
    test_known = test_known.merge(
        prod_stats[["product_id", "p05_base", "p95_base"]], on="product_id", how="left"
    )
    test_known = test_known.merge(
        prod_act_recent, on=["product_id", "activity_flag"], how="left"
    )

    has_act = test_known["p05_act"].notna()
    test_known["price_p05"] = test_known["p05_base"].copy()
    test_known["price_p95"] = test_known["p95_base"].copy()

    test_known.loc[has_act, "price_p05"] = (
        w_activity * test_known.loc[has_act, "p05_act"]
        + (1 - w_activity) * test_known.loc[has_act, "p05_base"]
    )
    test_known.loc[has_act, "price_p95"] = (
        w_activity * test_known.loc[has_act, "p95_act"]
        + (1 - w_activity) * test_known.loc[has_act, "p95_base"]
    )

    test_new = test[~test["product_id"].isin(known_products)].copy()
    test_new = test_new.merge(
        cat_stats, on=["management_group_id", "first_category_id"], how="left"
    )
    test_new = test_new.merge(cat_stats_1, on="first_category_id", how="left")
    test_new["price_p05"] = (
        test_new["p05_cat"].fillna(test_new["p05_c1"]).fillna(global_p05)
    )
    test_new["price_p95"] = (
        test_new["p95_cat"].fillna(test_new["p95_c1"]).fillna(global_p95)
    )

    predictions = pd.concat(
        [
            test_known[["row_id", "price_p05", "price_p95"]],
            test_new[["row_id", "price_p05", "price_p95"]],
        ]
    ).sort_values("row_id")

    return predictions["price_p05"].values, predictions["price_p95"].values


def get_model_2_pred(train, test, known_products):
    """Model 2: простые квантили"""
    q_low, q_high = 0.20, 0.65

    prod_stats = (
        train.groupby("product_id")
        .agg(
            {
                "price_p05": lambda x: x.quantile(q_low),
                "price_p95": lambda x: x.quantile(q_high),
            }
        )
        .reset_index()
    )
    prod_stats.columns = ["product_id", "p05", "p95"]

    cat_stats = (
        train.groupby(["management_group_id", "first_category_id"])
        .agg(
            {
                "price_p05": lambda x: x.quantile(q_low),
                "price_p95": lambda x: x.quantile(q_high),
            }
        )
        .reset_index()
    )
    cat_stats.columns = [
        "management_group_id",
        "first_category_id",
        "p05_cat",
        "p95_cat",
    ]

    test_known = test[test["product_id"].isin(known_products)].merge(
        prod_stats, on="product_id", how="left"
    )
    test_new = test[~test["product_id"].isin(known_products)].merge(
        cat_stats, on=["management_group_id", "first_category_id"], how="left"
    )
    test_new["p05"] = test_new["p05_cat"].fillna(train["price_p05"].quantile(q_low))
    test_new["p95"] = test_new["p95_cat"].fillna(train["price_p95"].quantile(q_high))

    predictions = pd.concat(
        [
            test_known[["row_id", "p05", "p95"]],
            test_new[["row_id", "p05", "p95"]],
        ]
    ).sort_values("row_id")

    return predictions["p05"].values, predictions["p95"].values


def optimize_weights_on_split(
    train_val, val_part, val_known_products, use_3_models=False
):
    """Оптимизирует веса на одном split"""
    val_test = val_part[
        ["product_id", "management_group_id", "first_category_id", "activity_flag"]
    ].copy()
    val_test["dt"] = val_part["dt"].values
    val_test["row_id"] = range(len(val_test))

    p05_m6, p95_m6 = get_model_6_opt_pred(train_val, val_test, val_known_products)

    if use_3_models:
        p05_m20, p95_m20 = get_model_20_pred(train_val, val_test, val_known_products)
        p05_m2, p95_m2 = get_model_2_pred(train_val, val_test, val_known_products)
    else:
        p05_m2, p95_m2 = get_model_2_pred(train_val, val_test, val_known_products)

    y_true_p05 = val_part["price_p05"].values
    y_true_p95 = val_part["price_p95"].values

    if use_3_models:

        def objective(weights):
            w1, w2, w3 = weights
            w1, w2, w3 = max(0, w1), max(0, w2), max(0, w3)
            total = w1 + w2 + w3
            if total == 0:
                return 1e6
            w1, w2, w3 = w1 / total, w2 / total, w3 / total

            p05_pred = w1 * p05_m6 + w2 * p05_m20 + w3 * p05_m2
            p95_pred = w1 * p95_m6 + w2 * p95_m20 + w3 * p95_m2

            iou = calc_iou(y_true_p05, y_true_p95, p05_pred, p95_pred)
            return -iou

        result = minimize(
            objective,
            x0=[0.5, 0.3, 0.2],
            method="L-BFGS-B",
            bounds=[(0, 1), (0, 1), (0, 1)],
        )

        w1, w2, w3 = result.x
        w1, w2, w3 = max(0, w1), max(0, w2), max(0, w3)
        total = w1 + w2 + w3
        if total == 0:
            return np.array([0.5, 0.3, 0.2])
        w1, w2, w3 = w1 / total, w2 / total, w3 / total
        return np.array([w1, w2, w3])
    else:

        def objective(weights):
            w1, w2 = weights
            w1, w2 = max(0, w1), max(0, w2)
            total = w1 + w2
            if total == 0:
                return 1e6
            w1, w2 = w1 / total, w2 / total

            p05_pred = w1 * p05_m6 + w2 * p05_m2
            p95_pred = w1 * p95_m6 + w2 * p95_m2

            iou = calc_iou(y_true_p05, y_true_p95, p05_pred, p95_pred)
            return -iou

        result = minimize(
            objective,
            x0=[0.7, 0.3],
            method="L-BFGS-B",
            bounds=[(0, 1), (0, 1)],
        )

        w1, w2 = result.x
        w1, w2 = max(0, w1), max(0, w2)
        total = w1 + w2
        if total == 0:
            return np.array([0.7, 0.3])
        w1, w2 = w1 / total, w2 / total
        return np.array([w1, w2])


def main():
    print("=" * 60)
    print("Model 24: Model 23 + Dynamic interval widening + Volatility-aware")
    print("=" * 60)

    train, test = load_data()
    train["dt"] = pd.to_datetime(train["dt"])
    train = train.sort_values("dt")

    print(f"Train: {train.shape}, Test: {test.shape}")

    known_products = set(train["product_id"].unique())

    print("\nТестирование ensemble из 2 моделей...")
    splits = [0.6, 0.65, 0.7, 0.75]
    all_weights_2 = []
    all_ious_2 = []

    for split in splits:
        cutoff = int(len(train) * split)
        train_val = train.iloc[:cutoff].copy()
        val_part = train.iloc[cutoff:].copy().reset_index(drop=True)
        val_known_products = set(train_val["product_id"].unique())

        weights = optimize_weights_on_split(
            train_val, val_part, val_known_products, use_3_models=False
        )
        all_weights_2.append(weights)

        val_test = val_part[
            ["product_id", "management_group_id", "first_category_id"]
        ].copy()
        val_test["dt"] = val_part["dt"].values
        val_test["row_id"] = range(len(val_test))

        p05_m6, p95_m6 = get_model_6_opt_pred(train_val, val_test, val_known_products)
        p05_m2, p95_m2 = get_model_2_pred(train_val, val_test, val_known_products)

        p05_ens = weights[0] * p05_m6 + weights[1] * p05_m2
        p95_ens = weights[0] * p95_m6 + weights[1] * p95_m2

        iou = calc_iou(
            val_part["price_p05"].values, val_part["price_p95"].values, p05_ens, p95_ens
        )
        all_ious_2.append(iou)

        print(
            f"Split {split:.0%}: weights=[{weights[0]:.3f}, {weights[1]:.3f}], IoU={iou:.4f}"
        )

    weights_array_2 = np.array(all_weights_2)
    mean_weights_2 = weights_array_2.mean(axis=0)
    std_weights_2 = weights_array_2.std(axis=0)
    mean_iou_2 = np.mean(all_ious_2)
    std_iou_2 = np.std(all_ious_2)

    print(f"\n2 модели - Стабильность:")
    print(f"  Mean: [{mean_weights_2[0]:.3f}, {mean_weights_2[1]:.3f}]")
    print(f"  Std:  [{std_weights_2[0]:.3f}, {std_weights_2[1]:.3f}]")
    print(f"  IoU:  {mean_iou_2:.4f} ± {std_iou_2:.4f}")

    # Теперь пробуем 3 модели
    print("\nТестирование ensemble из 3 моделей...")
    all_weights_3 = []
    all_ious_3 = []

    for split in splits:
        cutoff = int(len(train) * split)
        train_val = train.iloc[:cutoff].copy()
        val_part = train.iloc[cutoff:].copy().reset_index(drop=True)
        val_known_products = set(train_val["product_id"].unique())

        weights = optimize_weights_on_split(
            train_val, val_part, val_known_products, use_3_models=True
        )
        all_weights_3.append(weights)

        val_test = val_part[
            ["product_id", "management_group_id", "first_category_id", "activity_flag"]
        ].copy()
        val_test["dt"] = val_part["dt"].values
        val_test["row_id"] = range(len(val_test))

        p05_m6, p95_m6 = get_model_6_opt_pred(train_val, val_test, val_known_products)
        p05_m20, p95_m20 = get_model_20_pred(train_val, val_test, val_known_products)
        p05_m2, p95_m2 = get_model_2_pred(train_val, val_test, val_known_products)

        p05_ens = weights[0] * p05_m6 + weights[1] * p05_m20 + weights[2] * p05_m2
        p95_ens = weights[0] * p95_m6 + weights[1] * p95_m20 + weights[2] * p95_m2

        iou = calc_iou(
            val_part["price_p05"].values, val_part["price_p95"].values, p05_ens, p95_ens
        )
        all_ious_3.append(iou)

        print(
            f"Split {split:.0%}: weights=[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}], IoU={iou:.4f}"
        )

    weights_array_3 = np.array(all_weights_3)
    mean_weights_3 = weights_array_3.mean(axis=0)
    std_weights_3 = weights_array_3.std(axis=0)
    mean_iou_3 = np.mean(all_ious_3)
    std_iou_3 = np.std(all_ious_3)

    print(f"\n3 модели - Стабильность:")
    print(
        f"  Mean: [{mean_weights_3[0]:.3f}, {mean_weights_3[1]:.3f}, {mean_weights_3[2]:.3f}]"
    )
    print(
        f"  Std:  [{std_weights_3[0]:.3f}, {std_weights_3[1]:.3f}, {std_weights_3[2]:.3f}]"
    )
    print(f"  IoU:  {mean_iou_3:.4f} ± {std_iou_3:.4f}")

    # Выбираем лучший вариант
    use_3_models = mean_iou_3 > mean_iou_2
    if use_3_models:
        print("\n✓ Используем ensemble из 3 моделей")
        final_weights = mean_weights_3
        if std_weights_3.max() > 0.15:
            print("⚠️  Веса нестабильны, но используем средние")
    else:
        print("\n✓ Используем ensemble из 2 моделей")
        final_weights = mean_weights_2
        if std_weights_2.max() > 0.15:
            print("⚠️  Веса нестабильны, используем только model_6")
            use_3_models = False
            final_weights = None

    # Применяем к test
    print("\nГенерация финальных предсказаний...")
    # В test уже есть activity_flag, используем его
    test_with_act = test.copy()
    if "activity_flag" not in test_with_act.columns:
        # Если нет, используем последний известный из train
        last_activity = (
            train.sort_values("dt")
            .groupby("product_id")["activity_flag"]
            .last()
            .reset_index()
        )
        last_activity.columns = ["product_id", "activity_flag"]
        test_with_act = test_with_act.merge(last_activity, on="product_id", how="left")
        test_with_act["activity_flag"] = test_with_act["activity_flag"].fillna(0)

    p05_m6, p95_m6 = get_model_6_opt_pred(train, test, known_products)

    if final_weights is None:
        p05_ens = p05_m6
        p95_ens = p95_m6
    elif use_3_models:
        p05_m20, p95_m20 = get_model_20_pred(train, test_with_act, known_products)
        p05_m2, p95_m2 = get_model_2_pred(train, test, known_products)
        p05_ens = (
            final_weights[0] * p05_m6
            + final_weights[1] * p05_m20
            + final_weights[2] * p05_m2
        )
        p95_ens = (
            final_weights[0] * p95_m6
            + final_weights[1] * p95_m20
            + final_weights[2] * p95_m2
        )
    else:
        p05_m2, p95_m2 = get_model_2_pred(train, test, known_products)
        p05_ens = final_weights[0] * p05_m6 + final_weights[1] * p05_m2
        p95_ens = final_weights[0] * p95_m6 + final_weights[1] * p95_m2

    predictions = test[["row_id"]].copy()
    predictions["price_p05"] = p05_ens
    predictions["price_p95"] = p95_ens

    # Ensure valid
    predictions["price_p05"] = np.maximum(predictions["price_p05"], 0)
    predictions["price_p95"] = np.maximum(predictions["price_p95"], 0)
    predictions["price_p05"] = np.minimum(
        predictions["price_p05"], predictions["price_p95"]
    )
    predictions["price_p95"] = np.maximum(
        predictions["price_p05"], predictions["price_p95"]
    )

    min_width = 0.001
    interval_width = predictions["price_p95"] - predictions["price_p05"]
    too_narrow = interval_width < min_width
    predictions.loc[too_narrow, "price_p95"] = (
        predictions.loc[too_narrow, "price_p05"] + min_width
    )

    print(
        f"Stats до widening: p05={predictions['price_p05'].mean():.4f}, p95={predictions['price_p95'].mean():.4f}"
    )

    # Применяем dynamic interval widening
    print("\nПрименение dynamic interval widening...")
    predictions = apply_dynamic_interval_widening(predictions, train, test)

    # Повторная проверка валидности после widening
    predictions["price_p05"] = np.maximum(predictions["price_p05"], 0)
    predictions["price_p95"] = np.maximum(predictions["price_p95"], 0)
    predictions["price_p05"] = np.minimum(
        predictions["price_p05"], predictions["price_p95"]
    )
    predictions["price_p95"] = np.maximum(
        predictions["price_p05"], predictions["price_p95"]
    )

    interval_width = predictions["price_p95"] - predictions["price_p05"]
    too_narrow = interval_width < min_width
    predictions.loc[too_narrow, "price_p95"] = (
        predictions.loc[too_narrow, "price_p05"] + min_width
    )

    print(
        f"Stats после widening: p05={predictions['price_p05'].mean():.4f}, p95={predictions['price_p95'].mean():.4f}"
    )

    create_submission(predictions)
    print("Done!")


if __name__ == "__main__":
    main()
