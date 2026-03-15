from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bi_regional_inflation_synthetic.csv"
OUTPUT = ROOT / "data" / "outputs_session2"
OUTPUT.mkdir(exist_ok=True)


def main() -> None:
    df = pd.read_csv(DATA)
    df["month"] = pd.to_datetime(df["month"])
    df["high_inflation_pressure"] = (df["cpi_inflation_yoy"] >= 2.5).astype(int)
    df["month_num"] = df["month"].dt.month
    df["quarter"] = df["month"].dt.quarter

    features = [
        "province", "food_price_index", "exchange_rate_change_pct",
        "qris_transactions_growth_pct", "online_sentiment_score",
        "unemployment_rate", "hotel_occupancy_rate", "rainfall_index",
        "commodity_price_index", "month_num", "quarter"
    ]
    target = "high_inflation_pressure"

    X = df[features]
    y = df[target]

    categorical_features = ["province"]
    numeric_features = [
        "food_price_index", "exchange_rate_change_pct", "qris_transactions_growth_pct",
        "online_sentiment_score", "unemployment_rate", "hotel_occupancy_rate",
        "rainfall_index", "commodity_price_index", "month_num", "quarter"
    ]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=300, max_depth=6),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["model"], "predict_proba") else None
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        })
        fitted[name] = pipe

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    print(results_df)
    results_df.to_csv(OUTPUT / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["Model"]
    best_pipe = fitted[best_name]
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"ROC Curve - {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT / "roc_curve.png", dpi=160)
    plt.close()


    # Comparison chart
    plt.figure(figsize=(8, 5))
    ordered = results_df.sort_values("F1", ascending=True)
    plt.barh(ordered["Model"], ordered["F1"])
    plt.title("Model Comparison by F1 Score")
    plt.xlabel("F1 Score")
    plt.tight_layout()
    plt.savefig(OUTPUT / "model_comparison_f1.png", dpi=160)
    plt.close()

    # Driver plot for the best model
    if best_name == "Random Forest":
        ohe = best_pipe.named_steps["prep"].named_transformers_["cat"]
        feature_names = list(ohe.get_feature_names_out(["province"])) + numeric_features
        importances = best_pipe.named_steps["model"].feature_importances_
        feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
        feat_imp.to_csv(OUTPUT / "feature_drivers.csv", index=False)
        plt.figure(figsize=(10, 6))
        top10 = feat_imp.head(10).iloc[::-1]
        plt.barh(top10["feature"], top10["importance"])
        plt.title("Top 10 Feature Drivers")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(OUTPUT / "feature_drivers.png", dpi=160)
        plt.close()
    else:
        ohe = best_pipe.named_steps["prep"].named_transformers_["cat"]
        feature_names = list(ohe.get_feature_names_out(["province"])) + numeric_features
        coefs = best_pipe.named_steps["model"].coef_[0]
        feat_imp = pd.DataFrame({"feature": feature_names, "importance": abs(coefs), "signed_coef": coefs}).sort_values("importance", ascending=False)
        feat_imp.to_csv(OUTPUT / "feature_drivers.csv", index=False)
        plt.figure(figsize=(10, 6))
        top10 = feat_imp.head(10).iloc[::-1]
        plt.barh(top10["feature"], top10["signed_coef"])
        plt.title("Top 10 Feature Drivers (signed coefficients)")
        plt.xlabel("Coefficient")
        plt.tight_layout()
        plt.savefig(OUTPUT / "feature_drivers.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
