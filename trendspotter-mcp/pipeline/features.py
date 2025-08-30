# src/features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def compute_timeseries_features(df, window=7):
    """
    df: DataFrame with columns ['hashtag', 'date', 'count'] where date is YYYY-MM-DD
    returns: DataFrame with one row per hashtag and features
    """
    df['date'] = pd.to_datetime(df['date'])
    feats = []
    for tag, grp in df.groupby('hashtag'):
        grp_sorted = grp.sort_values('date')
        counts = grp_sorted['count'].values.astype(float)
        if len(counts) < 5:
            # pad
            counts = np.pad(counts, (0, 5 - len(counts)), constant_values=0)
        # recent slope: linear regression on last window days
        y = counts[-window:]
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        volatility = np.std(counts)
        max_count = np.max(counts)
        mean_recent = np.mean(y)
        age_days = (grp_sorted['date'].max() - grp_sorted['date'].min()).days
        # simple decay metric: compare recent mean to historical mean
        overall_mean = np.mean(counts) + 1e-6
        recent_vs_overall = mean_recent / overall_mean
        feats.append({
            'hashtag': tag,
            'slope': slope,
            'volatility': volatility,
            'max_count': max_count,
            'mean_recent': mean_recent,
            'age_days': age_days,
            'recent_vs_overall': recent_vs_overall
        })
    feats_df = pd.DataFrame(feats)
    return feats_df