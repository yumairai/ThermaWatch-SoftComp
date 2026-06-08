import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta

# Memasukkan direktori backend ke path pencarian modul Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from pipeline.run_daily_pipeline import run_daily_pipeline

def main():
    start_date = datetime(2026, 1, 9)
    end_date = datetime(2026, 5, 23)
    
    current_date = start_date
    dates = []
    while current_date <= end_date:
        dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
        
    print(f"Total days to backfill: {len(dates)}")
    
    for idx, d_str in enumerate(dates):
        print(f"[{idx+1}/{len(dates)}] Running daily pipeline for: {d_str}")
        try:
            run_daily_pipeline(d_str)
            time.sleep(1)
        except Exception as e:
            print(f"Error processing {d_str}: {e}")
            
    # Sync back to raw_data
    print("Syncing data to raw_data/ERA5_LST_Jabar_Daily_Clean.csv...")
    df = pd.read_csv('data/Dataset_Master_ERA5_Ready_LSTM.csv')
    era_cols = [
        'date', 'Kabupaten', 'ERA5_LST_Mean', 'ERA5_LST_Max', 
        'ERA5_LST_Percentile95', 'ERA5_Cloud_Cover_Percentage', 
        'ERA5_Max_Lon', 'ERA5_Max_Lat'
    ]
    df_raw = df[df['date'] <= '2026-05-31'][era_cols].copy()
    df_raw.to_csv('data/raw_data/ERA5_LST_Jabar_Daily_Clean.csv', index=False)
    print("Synchronization complete!")

if __name__ == '__main__':
    main()
