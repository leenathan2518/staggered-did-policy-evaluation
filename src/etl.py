import os
import json
import pandas as pd


def main():
    print("🔄 Initializing ETL Data Cleansing Pipeline...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(current_dir, "..", "data", "raw")
    processed_dir = os.path.join(current_dir, "..", "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Policy implementation dates mapping for the Staggered DiD setup
    policy_map = {
        "Beijing": "2024-08-26",
        "Guangdong": "2024-08-29",
        "Shanghai": "2024-09-07",
        "Sichuan": "2024-08-26",
        "Shandong": "2024-09-09",
        "Zhejiang": "2024-08-22",
        "Hubei": "2024-08-10",
        "Henan": "2024-09-10"
    }

    all_dfs = []

    for province, p_date in policy_map.items():
        # Corrected file naming to target the correct output_{Province}.json files exactly
        raw_file = os.path.join(raw_dir, f"output_{province}.json")
        if not os.path.exists(raw_file):
            print(f"❌ Missing raw file for {province} at {raw_file}, skip.")
            continue

        with open(raw_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Generate continuous date range based on JSON metadata
        dates = pd.date_range(start=data['start_date'], end=data['end_date'])
        index_list = data['index_list']

        # Safety Check 1: Ensure date range length matches the search index list length
        if len(dates) != len(index_list):
            print(f"⚠️ Warning for {province}: Date length ({len(dates)}) "
                  f"does not match index_list length ({len(index_list)}). Skipping this file.")
            continue

        df_prov = pd.DataFrame({
            'province': province,                      # English name for econometric modeling
            'province_zh': data.get('province', ''),   # Chinese name from JSON for dashboard visualization
            'date': dates,
            'search_index': index_list,
            'policy_date': pd.to_datetime(p_date)
        })

        all_dfs.append(df_prov)

    if not all_dfs:
        print("❌ No valid raw data processed. Please check your output_*.json files.")
        return

    # Combine all individual dataframes into a single panel dataset
    df_panel = pd.concat(all_dfs, ignore_index=True)
    df_panel['date'] = pd.to_datetime(df_panel['date'])

    # Safety Check 2: Coerce search index to numeric to avoid data type anomalies during regression
    df_panel['search_index'] = pd.to_numeric(df_panel['search_index'], errors='coerce')

    # Feature Engineering: Construct Staggered DiD dummy and Event-Time axis
    # 'did' = 1 represents the treatment group during the post-treatment period
    df_panel['did'] = (df_panel['date'] >= df_panel['policy_date']).astype(int)
    # 'days_to_policy' is critical for parallel trends testing and event study plotting
    df_panel['days_to_policy'] = (df_panel['date'] - df_panel['policy_date']).dt.days

    # Export the finalized clean panel dataset
    output_path = os.path.join(processed_dir, "final_panel_data.csv")
    # Using utf-8-sig to prevent Excel from displaying Chinese characters as corrupted code
    df_panel.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ ETL process completed! Merged {len(df_panel)} observations.")
    print(f"💾 Cleaned panel dataset exported to: {output_path}\n")
    print("📊 Preview of the top 5 records:")
    print(df_panel.head())


if __name__ == "__main__":
    main()