from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bi_regional_inflation_synthetic.csv"
OUTPUT = ROOT / "data" / "outputs_session1"
OUTPUT.mkdir(exist_ok=True)


def main() -> None:
    df = pd.read_csv(DATA)
    df["month"] = pd.to_datetime(df["month"])

    print("Dataset shape:", df.shape)
    print(df.head())
    print("\nSummary statistics:\n", df.describe())
    print("\nMissing values:\n", df.isna().sum())

    province_avg = df.groupby("province")["cpi_inflation_yoy"].mean().sort_values(ascending=False)
    print("\nAverage inflation by province:\n", province_avg)

    monthly_avg = df.groupby("month")["cpi_inflation_yoy"].mean()
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_avg.index, monthly_avg.values)
    plt.title("Average YoY Inflation Across Provinces")
    plt.xlabel("Month")
    plt.ylabel("Inflation (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT / "avg_inflation_trend.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["cpi_inflation_yoy"], bins=18)
    plt.title("Distribution of YoY Inflation")
    plt.xlabel("Inflation (%)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT / "inflation_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["food_price_index"], df["cpi_inflation_yoy"])
    plt.title("Food Price Index vs Inflation")
    plt.xlabel("Food Price Index")
    plt.ylabel("Inflation (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT / "food_vs_inflation.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["qris_transactions_growth_pct"], df["cpi_inflation_yoy"])
    plt.title("QRIS Growth vs Inflation")
    plt.xlabel("QRIS Transaction Growth (%)")
    plt.ylabel("Inflation (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT / "qris_vs_inflation.png", dpi=160)
    plt.close()

    corr = df.select_dtypes(include="number").corr()
    print("\nCorrelation matrix:\n", corr)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT / "correlation_matrix.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
