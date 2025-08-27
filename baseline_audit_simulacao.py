#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline & Emissions Audit — Simulation Pipeline
- Simulates hourly telemetry for multiple sites/meters (12 months)
- Adds weather, occupancy, production signals
- Computes emissions (tCO2e) from variable grid factors
- Audits data quality
- Fits linear baseline models (CDD/HDD + occupancy + production)
- Reconciles monthly totals with simulated invoices (acceptance ≤5% MAPE)
- Exports CSVs and simple charts (matplotlib)
"""
import os
import math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

BASE_DIR = os.path.join(os.getcwd(), "baseline_audit_outputs")
os.makedirs(BASE_DIR, exist_ok=True)
TZ = "America/Sao_Paulo"
start = pd.Timestamp("2024-09-01 00:00:00", tz=TZ)
end   = pd.Timestamp("2025-08-31 23:00:00", tz=TZ)
sites = ["SITE_A", "SITE_B"]
meters_per_site = 2
temp_base_cooling = 18.0
temp_base_heating = 15.0

def simulate_weather_index(ts_index):
    days = pd.date_range(ts_index.min().normalize(), ts_index.max().normalize(), freq="D", tz=TZ)
    day_of_year = days.dayofyear.values
    temp_daily = 24 + 6*np.sin(2*np.pi*(day_of_year/365.0)) + np.random.normal(0, 1.0, size=len(days))
    temp_df = pd.DataFrame({"date": days, "temp_c_daily": temp_daily}).set_index("date")
    temp_series = temp_df.reindex(ts_index.tz_convert(TZ).floor("D")).reset_index(drop=True)["temp_c_daily"]
    temp_series.index = ts_index
    hum_daily = 70 - 10*np.sin(2*np.pi*(day_of_year/365.0)) + np.random.normal(0, 3.0, size=len(days))
    hum_series = pd.Series(hum_daily).reindex(ts_index.tz_convert(TZ).floor("D")).reset_index(drop=True)
    hum_series.index = ts_index
    hour = ts_index.hour.values
    month = ts_index.month.values
    diurnal = 0.08*np.sin(2*np.pi*(hour/24.0)) + 0.08
    seasonal = 0.05*np.sin(2*np.pi*((month-1)/12.0)) + 0.15
    ef = np.clip(diurnal + seasonal + np.random.normal(0, 0.01, size=len(ts_index)), 0.05, 0.35)
    return pd.DataFrame({"temp_c": temp_series.values, "humidity": hum_series.values, "grid_kgco2_per_kwh": ef}, index=ts_index)

def simulate_occupancy(idx):
    is_weekend = idx.weekday >= 5
    hour = idx.hour
    base = np.where(is_weekend, 0.4, 0.7)
    peak = np.where((hour >= 8) & (hour <= 18), 1.0, 0.5)
    occ = base * peak + np.random.normal(0, 0.05, size=len(idx))
    return np.clip(occ, 0, 1.2)

def simulate_production(idx):
    w = 1.0 + 0.1*np.sin(2*np.pi*(idx.dayofyear/7.0))
    noise = np.random.normal(0, 0.05, size=len(idx))
    return np.clip(w + noise, 0.8, 1.3)

def degree_days(temp_c, base_c, mode="cooling"):
    return np.maximum(temp_c - base_c, 0.0) if mode=="cooling" else np.maximum(base_c - temp_c, 0.0)

def simulate_meter_kwh(wdf, base_load, alpha_cdd, beta_hdd, gamma_occ, delta_prod, noise_sd, index):
    kwh = (base_load
           + alpha_cdd * wdf.loc[index, "cdd"].values
           + beta_hdd  * wdf.loc[index, "hdd"].values
           + gamma_occ * wdf.loc[index, "occupancy"].values
           + delta_prod* wdf.loc[index, "production_index"].values
           + np.random.normal(0, noise_sd, size=len(index)))
    hour = index.hour
    kwh *= 0.9 + 0.2*np.where((hour >= 8) & (hour <= 18), 1.0, 0.7)
    return np.clip(kwh, 0.05, None)

def audit_data(df):
    results = []
    for (site, meter), g in df.groupby(["site", "meter_id"]):
        total = len(g)
        missing = g["kwh"].isna().sum()
        missing_pct = 100*missing/total if total else 0.0
        dups = g.index.duplicated().sum()
        expected = pd.date_range(g.index.min(), g.index.max(), freq="H", tz=g.index.tz)
        gaps = len(set(expected).difference(set(g.index)))
        x = g["kwh"].dropna()
        if len(x) > 0:
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5*iqr
            lower = max(q1 - 1.5*iqr, 0)
            outliers = ((x > upper) | (x < lower)).sum()
            roll = x.rolling(24*30, min_periods=24*7).mean().diff().abs().median()
            drift = float(0 if np.isnan(roll) else roll)
        else:
            outliers = 0
            drift = 0.0
        results.append({"site": site, "meter_id": meter, "rows": int(total), "missing_pct": round(missing_pct, 2),
                        "duplicates": int(dups), "gaps": int(gaps), "iqr_outliers": int(outliers),
                        "median_monthly_roll_change_kwh": drift})
    return pd.DataFrame(results)

def build_baseline_models(df):
    records = []
    preds_list = []
    for (site, meter), g in df.groupby(["site", "meter_id"]):
        g2 = g.dropna(subset=["kwh", "cdd", "hdd", "occupancy", "production_index"])
        if len(g2) < 500:
            continue
        X = g2[["cdd", "hdd", "occupancy", "production_index"]].values
        y = g2["kwh"].values
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        mape = float(np.mean(np.abs((y - y_pred)/np.clip(y, 1e-3, None))) * 100.0)
        r2 = model.score(X, y)
        coefs = dict(zip(["cdd","hdd","occupancy","production_index"], model.coef_))
        records.append({"site": site, "meter_id": meter, "intercept": model.intercept_,
                        "coef_cdd": coefs["cdd"], "coef_hdd": coefs["hdd"],
                        "coef_occupancy": coefs["occupancy"], "coef_production": coefs["production_index"],
                        "MAE_kWh": mae, "MAPE_pct": mape, "R2": r2})
        tmp = g2.copy()
        tmp["kwh_pred"] = y_pred
        preds_list.append(tmp[["site","meter_id","kwh","kwh_pred"]])
    return pd.DataFrame(records), (pd.concat(preds_list) if preds_list else pd.DataFrame())

def monthly_summaries(df, preds_df):
    act = df.groupby([pd.Grouper(key="timestamp", freq="M"), "site", "meter_id"]).agg(
        kwh=("kwh", "sum"),
        tco2e=("tco2e", "sum")
    ).reset_index().rename(columns={"timestamp": "month", "kwh": "kwh_actual", "tco2e": "tco2e_actual"})
    pred = preds_df.groupby([pd.Grouper(key="timestamp", freq="M"), "site", "meter_id"]).agg(
        kwh_pred=("kwh_pred", "sum")
    ).reset_index().rename(columns={"timestamp": "month"})
    merged = pd.merge(act, pred, on=["site","meter_id","month"], how="left")
    return merged

def save_plot(fig, name):
    path = os.path.join(BASE_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def main():
    idx = pd.date_range(start, end, freq="H", tz=TZ)
    wdf = simulate_weather_index(idx)
    wdf["cdd"] = degree_days(wdf["temp_c"].values, temp_base_cooling, "cooling")
    wdf["hdd"] = degree_days(wdf["temp_c"].values, temp_base_heating, "heating")
    wdf["occupancy"] = simulate_occupancy(idx)
    wdf["production_index"] = simulate_production(idx)

    rows = []
    for site in sites:
        for m in range(1, meters_per_site+1):
            base_load = np.random.uniform(15, 30)
            alpha_cdd = np.random.uniform(0.2, 0.6)
            beta_hdd  = np.random.uniform(0.1, 0.3)
            gamma_occ = np.random.uniform(3.0, 6.0)
            delta_prod= np.random.uniform(2.0, 5.0)
            noise_sd  = np.random.uniform(0.8, 1.5)
            kwh = simulate_meter_kwh(wdf, base_load, alpha_cdd, beta_hdd, gamma_occ, delta_prod, noise_sd, idx)
            df_tmp = pd.DataFrame({"site": site, "meter_id": f"{site}_M{m}", "kwh": kwh}, index=idx)
            rows.append(df_tmp)
    telemetry = pd.concat(rows).sort_index()

    # Missingness and outliers (by position to avoid duplicate label issues)
    miss_mask = np.random.choice([True, False], size=len(telemetry), p=[0.01, 0.99])
    telemetry.loc[miss_mask, "kwh"] = np.nan
    n_rows = len(telemetry)
    n_outliers = max(1, int(n_rows * 0.002))
    pos_idx = np.random.choice(np.arange(n_rows), size=n_outliers, replace=False)
    vals = telemetry["kwh"].values
    vals[pos_idx] = vals[pos_idx] * np.random.uniform(2.0, 4.0, size=n_outliers)
    telemetry["kwh"] = vals

    df = telemetry.join(wdf, how="left")
    df["tco2e"] = df["kwh"] * df["grid_kgco2_per_kwh"] / 1000.0
    df["timestamp"] = df.index

    audit_df = audit_data(df)
    models_df, preds_df = build_baseline_models(df)
    preds_df["timestamp"] = preds_df.index
    monthly = monthly_summaries(df, preds_df)

    invoices = (
        monthly.groupby(["month","site"])["kwh_actual"]
        .sum().reset_index().rename(columns={"kwh_actual":"kwh_invoice"})
    )
    bias = np.random.uniform(-0.01, 0.01, size=len(invoices))
    noise = np.random.normal(0, 0.02, size=len(invoices))
    invoices["kwh_invoice"] = np.clip(invoices["kwh_invoice"] * (1 + bias + noise), 0, None)

    recon = (
        monthly.groupby(["month","site"])["kwh_actual"].sum().reset_index()
        .merge(invoices, on=["month","site"], how="left")
    )
    recon["abs_pct_error_vs_invoice"] = 100*np.abs(recon["kwh_actual"] - recon["kwh_invoice"]) / np.clip(recon["kwh_invoice"], 1e-6, None)
    recon_summary = recon.groupby("site")["abs_pct_error_vs_invoice"].mean().reset_index().rename(columns={"abs_pct_error_vs_invoice":"mean_abs_pct_error_vs_invoice"})
    recon_summary["acceptance_met_<=5pct"] = recon_summary["mean_abs_pct_error_vs_invoice"] <= 5.0

    # Exports
    df.reset_index().to_csv(os.path.join(BASE_DIR, "telemetry_hourly_simulada.csv"), index=False)
    audit_df.to_csv(os.path.join(BASE_DIR, "auditoria_qualidade_dados.csv"), index=False)
    models_df.to_csv(os.path.join(BASE_DIR, "modelos_baseline_coeficientes.csv"), index=False)
    preds_df.reset_index().to_csv(os.path.join(BASE_DIR, "baseline_predicoes_hourly.csv"), index=False)
    monthly.to_csv(os.path.join(BASE_DIR, "resumo_mensal_kwh_tco2e.csv"), index=False)
    invoices.to_csv(os.path.join(BASE_DIR, "faturas_simuladas_mensais.csv"), index=False)
    recon.to_csv(os.path.join(BASE_DIR, "reconciliacao_vs_faturas.csv"), index=False)
    recon_summary.to_csv(os.path.join(BASE_DIR, "criterio_aceite_por_site.csv"), index=False)

    # Charts (separate figures; default colors)
    site0 = telemetry["site"].unique()[0]
    ts_plot_df = df[df["site"]==site0].groupby(pd.Grouper(freq="D"))["kwh"].sum()
    fig1 = plt.figure(); ts_plot_df.plot()
    plt.title(f"Energia diária — {site0}"); plt.xlabel("Data"); plt.ylabel("kWh/dia")
    p1 = save_plot(fig1, f"plot_energia_diaria_{site0}.png")

    ef_plot = df.groupby(pd.Grouper(freq="D"))["grid_kgco2_per_kwh"].mean()
    fig2 = plt.figure(); ef_plot.plot()
    plt.title("Fator de emissão da rede — média diária (kgCO2e/kWh)"); plt.xlabel("Data"); plt.ylabel("kgCO2e/kWh")
    p2 = save_plot(fig2, "plot_fator_emissao_rede.png")

    m_site = monthly[monthly["site"]==site0].groupby("month")[["kwh_actual","kwh_pred"]].sum()
    fig3 = plt.figure(); m_site.plot()
    plt.title(f"Resumo mensal — Real vs. Baseline (previsto) — {site0}"); plt.xlabel("Mês"); plt.ylabel("kWh")
    p3 = save_plot(fig3, f"plot_mensal_real_vs_previsto_{site0}.png")

    rec_site = recon[recon["site"]==site0].set_index("month")
    fig4 = plt.figure(); rec_site["abs_pct_error_vs_invoice"].plot(kind="bar")
    plt.title(f"Erro abs. % vs. faturas — {site0}"); plt.xlabel("Mês"); plt.ylabel("Erro absoluto (%)")
    p4 = save_plot(fig4, f"plot_erro_vs_faturas_{site0}.png")

    meter0 = telemetry["meter_id"].unique()[0]
    sc = df[df["meter_id"]==meter0].dropna(subset=["kwh","cdd"])
    fig5 = plt.figure(); plt.scatter(sc["cdd"], sc["kwh"], s=3)
    plt.title(f"Dispersão kWh vs CDD — {meter0}"); plt.xlabel("CDD (°C·h acima de 18°C)"); plt.ylabel("kWh")
    p5 = save_plot(fig5, f"plot_disp_kwh_vs_cdd_{meter0}.png")

    print("Pronto. Arquivos em:", BASE_DIR)
    print(recon_summary)

if __name__ == "__main__":
    main()
