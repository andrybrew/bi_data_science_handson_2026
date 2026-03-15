import math
import numpy as np
import pandas as pd
from pathlib import Path


def build_dataset(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    provinces = [
        "DKI Jakarta", "Jawa Barat", "Jawa Tengah", "Jawa Timur", "Bali",
        "Sumatera Utara", "Sumatera Selatan", "Kalimantan Timur",
        "Sulawesi Selatan", "NTB", "Riau", "Lampung"
    ]
    months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
    rows = []

    for prov in provinces:
        prov_effect = np.random.normal(0, 0.22)
        tourism_bias = 1 if prov in ["Bali", "NTB", "DKI Jakarta"] else 0
        manuf_bias = 1 if prov in ["Jawa Barat", "Jawa Timur", "Riau"] else 0

        for month in months:
            m = month.month
            festive = 1 if m in [3, 4, 11, 12] else 0
            wet = 1 if m in [1, 2, 11, 12] else 0
            season = 0.35 * math.sin((m / 12) * 2 * math.pi)

            food_price_index = 100 + np.random.normal(0, 3.5) + season * 4 + festive * 2.1 + prov_effect * 1.8
            exchange_rate_change_pct = np.random.normal(0.15, 1.1) + festive * 0.08
            qris_transactions_growth_pct = np.random.normal(17, 5.2) + tourism_bias * 2.8 + prov_effect * 2.5
            online_sentiment_score = np.clip(np.random.normal(0.12, 0.22) - festive * 0.03, -1, 1)
            unemployment_rate = np.clip(np.random.normal(5.3, 1.0) - prov_effect - manuf_bias * 0.2, 2.5, 9.0)
            hotel_occupancy_rate = np.clip(np.random.normal(57, 9.5) + tourism_bias * 9 + season * 6, 28, 92)
            rainfall_index = np.clip(np.random.normal(95, 22) + wet * 18 + season * 6, 25, 185)
            commodity_price_index = 100 + np.random.normal(0, 4) + manuf_bias * 1.5 + festive * 1.2

            inflation = (
                2.15
                + 0.038 * (food_price_index - 100)
                + 0.17 * exchange_rate_change_pct
                - 0.012 * qris_transactions_growth_pct
                - 0.33 * online_sentiment_score
                + 0.024 * unemployment_rate
                - 0.010 * hotel_occupancy_rate
                + 0.0045 * rainfall_index
                + 0.016 * (commodity_price_index - 100)
                + festive * 0.25
                + prov_effect
                + np.random.normal(0, 0.28)
            )
            core_inflation_yoy = (
                1.8
                + 0.45 * inflation
                + 0.08 * exchange_rate_change_pct
                + 0.01 * (commodity_price_index - 100)
                + np.random.normal(0, 0.18)
            )

            rows.append([
                prov,
                month.strftime("%Y-%m-%d"),
                round(inflation, 2),
                round(core_inflation_yoy, 2),
                round(food_price_index, 2),
                round(exchange_rate_change_pct, 2),
                round(qris_transactions_growth_pct, 2),
                round(online_sentiment_score, 3),
                round(unemployment_rate, 2),
                round(hotel_occupancy_rate, 2),
                round(rainfall_index, 2),
                round(commodity_price_index, 2),
            ])

    df = pd.DataFrame(rows, columns=[
        "province", "month", "cpi_inflation_yoy", "core_inflation_yoy",
        "food_price_index", "exchange_rate_change_pct", "qris_transactions_growth_pct",
        "online_sentiment_score", "unemployment_rate", "hotel_occupancy_rate",
        "rainfall_index", "commodity_price_index"
    ])
    df["high_inflation_pressure"] = (df["cpi_inflation_yoy"] >= 2.5).astype(int)
    df["month_num"] = pd.to_datetime(df["month"]).dt.month
    df["quarter"] = pd.to_datetime(df["month"]).dt.quarter
    return df


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_csv = root / "data" / "bi_regional_inflation_synthetic.csv"
    out_xlsx = root / "data" / "bi_regional_inflation_synthetic.xlsx"
    df = build_dataset()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="dataset")
        pd.DataFrame([
            ["province", "Province name", "categorical"],
            ["month", "Observation month", "date"],
            ["cpi_inflation_yoy", "Synthetic YoY CPI inflation (%)", "numeric"],
            ["core_inflation_yoy", "Synthetic YoY core inflation (%)", "numeric"],
            ["food_price_index", "Food price pressure index (base around 100)", "numeric"],
            ["exchange_rate_change_pct", "Monthly exchange-rate change (%)", "numeric"],
            ["qris_transactions_growth_pct", "YoY QRIS transaction growth proxy (%)", "numeric"],
            ["online_sentiment_score", "Online public sentiment proxy (-1 to 1)", "numeric"],
            ["unemployment_rate", "Unemployment rate proxy (%)", "numeric"],
            ["hotel_occupancy_rate", "Hotel occupancy proxy (%)", "numeric"],
            ["rainfall_index", "Rainfall and weather shock proxy", "numeric"],
            ["commodity_price_index", "Commodity price index proxy (base around 100)", "numeric"],
            ["high_inflation_pressure", "1 if inflation >= 2.5, else 0", "binary"],
            ["month_num", "Month number 1-12", "integer"],
            ["quarter", "Quarter number 1-4", "integer"],
        ], columns=["field", "description", "type"]).to_excel(writer, index=False, sheet_name="data_dictionary")
    print(f"Saved {out_csv} and {out_xlsx}")


if __name__ == "__main__":
    main()
