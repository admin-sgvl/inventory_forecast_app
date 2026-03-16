import pandas as pd
import numpy as np

def generate_sales():
    dates = pd.date_range(start="2024-01-01", end="2026-03-01")
    sales = []
    base = 100
    for d in dates:
        # Add weekly spike (Weekends)
        weekend_boost = 30 if d.weekday() >= 5 else 0
        # Add seasonal spike (December)
        holiday_boost = 50 if d.month == 12 else 0
        # Add noise
        daily_sale = base + weekend_boost + holiday_boost + np.random.normal(0, 10)
        sales.append(max(0, int(daily_sale)))
    
    df = pd.DataFrame({'Date': dates, 'Sales': sales})
    df.to_csv('historical_sales.csv', index=False)
    print("✅ Created historical_sales.csv")

if __name__ == "__main__":
    generate_sales()