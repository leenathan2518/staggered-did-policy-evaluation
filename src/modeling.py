import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def run_twfe_regression(df, results_dir):
    """Estimate Staggered Two-Way Fixed Effects Model (TWFE) with Robust SE"""
    print("⏳ 1. Fitting Two-Way Fixed Effects (TWFE) Panel Regression...")

    # linearmodels requires a multi-index of [Entity, Time]
    df_reg = df.set_index(['province', 'date'])

    dependent = df_reg['search_index']
    exog = df_reg['did']

    # Model with entity (province) and time (date) fixed effects
    model = PanelOLS(dependent, exog, entity_effects=True, time_effects=True)
    results = model.fit(cov_type='robust')

    print("\n📊 === Two-Way Fixed Effects (TWFE) Estimation Results (Robust SE) ===")
    print(results.summary)

    output_txt = os.path.join(results_dir, "regression_report.txt")
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(str(results.summary))
    print(f"\n💾 Regression report exported to: {output_txt}")


def plot_event_study(df, charts_dir, window_size):
    """Generate raw data trend plot around policy enactment within the truncated window"""
    print(f"\n⏳ 2. Plotting Raw Data Trend (Event Time Window: +/-{window_size} Days)...")

    plt.figure(figsize=(11, 6), dpi=150)

    # Use the same truncated window for visualization consistency
    df_window = df[(df['days_to_policy'] >= -window_size) & (df['days_to_policy'] <= window_size)].copy()

    sns.lineplot(
        data=df_window,
        x='days_to_policy',
        y='search_index',
        errorbar=('ci', 95),
        color='#1f77b4',
        linewidth=2.5,
        marker='o',
        label='Mean Search Index (95% CI)'
    )

    # Mark policy enactment date line
    plt.axvline(x=0, color='#d62728', linestyle='--', linewidth=2, label='Policy Enactment (Day 0)')

    # Professional chart labeling
    plt.title(f'Dynamic Causal Shocks of Trade-in Policy (+/-{window_size} Days Window)', fontsize=13, fontweight='bold',
              pad=15)
    plt.xlabel('Relative Days to Policy Enactment (Event Time)', fontsize=11)
    plt.ylabel('Baidu Search Index', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)

    output_img = os.path.join(charts_dir, "event_study_plot.png")
    plt.tight_layout()
    plt.savefig(output_img, bbox_inches='tight')
    plt.close()
    print(f"💾 Parallel trend visualization chart saved to: {output_img}")


def main():
    print("📊 Initializing Econometric Empirical Pipeline...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_file = os.path.join(current_dir, "..", "data", "processed", "final_panel_data.csv")
    charts_dir = os.path.join(current_dir, "..", "output", "charts")
    results_dir = os.path.join(current_dir, "..", "output", "results")

    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(processed_file):
        print(f"❌ Cleaned panel data not found at: {processed_file}. Please run etl.py first.")
        return

    df = pd.read_csv(processed_file)
    df['date'] = pd.to_datetime(df['date'])

    # CRITICAL TUNING: Truncate the time window to eliminate long-term noise and avoid effect dilution
    # You can change 30 to 14 if you want to focus exclusively on ultra-short-term shock
    WINDOW_DAYS = 21
    df_truncated = df[(df['days_to_policy'] >= -WINDOW_DAYS) & (df['days_to_policy'] <= WINDOW_DAYS)].copy()
    print(f"✂️ Sample filtered from {len(df)} to {len(df_truncated)} observations using a +/-{WINDOW_DAYS} days window.")

    # Execute empirical steps on the truncated clean dataset
    run_twfe_regression(df_truncated, results_dir)
    plot_event_study(df_truncated, charts_dir, WINDOW_DAYS)

    print("\n🎉 Staggered DiD Baseline Empirical Pipeline successfully executed!")


if __name__ == "__main__":
    main()