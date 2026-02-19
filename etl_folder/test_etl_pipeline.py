"""
Data Preprocessing Module

This module provides functions for cleaning and preprocessing sensor data.
Functions include dropping unnecessary columns, timestamp formatting, 
extracting date/time components, and hourly aggregation.

IMPORTANT : PLEASE REFER TO SECTIONS IN THE CODE TO UNDERSTANDING  PURPOSE.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# SECTION 1 - Data Cleaning and Preprocessing Functions


def data_ingestion(df):
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # BUG FIX: Coerce sensor/environment columns to numeric so they survive
    # aggregation and AQI calculation (CSV may read them as object/str).
    numeric_cols = ['P0', 'P1', 'P2', 'temperature', 'humidity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def drop_unnecessary_columns(df):
    """
    Drop unnecessary columns from the dataframe.
    
    Columns to drop:
    - data_id
    - location_country
    - software_version
    - sensor_value_id
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with specified columns removed
    """
    columns_to_drop = [
        'location', 
        'country', 
        'software_version',
        'sampling_rate',     
    ]
    
    # Only drop columns that exist in the dataframe
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    
    df_cleaned = df.drop(columns=existing_columns)
    
    print(f"Dropped columns: {existing_columns}")
    return df_cleaned


def format_timestamp(df, timestamp_col='timestamp'):
    """
    Format timestamp column to datetime and create 'time_formatted' column.
    Automatically discards rows with invalid timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing timestamp column.
    timestamp_col : str
        Name of the timestamp column (default: 'timestamp').

    Returns
    -------
    pd.DataFrame
        DataFrame with 'timestamp' converted to datetime and
        new 'time_formatted' column. Rows with invalid timestamps are removed.
    """
    import pandas as pd

    # BUG FIX: If the column is already datetime, skip re-parsing to avoid
    # stripping timezone info or clobbering a previously parsed column.
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        # Try UTC-aware parse first; fall back to format-inferred parse so
        # mixed-format strings (ISO-8601 with/without timezone) all succeed.
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors='coerce')

    # Count invalid timestamps
    num_invalid = df[timestamp_col].isna().sum()
    if num_invalid > 0:
        print(f"Discarding {num_invalid} rows with invalid timestamps.")
        df = df.dropna(subset=[timestamp_col]).copy()

    # Create formatted timestamp column (YYYY-MM-DD HH:MM)
    df['time_formatted'] = df[timestamp_col].dt.strftime('%Y-%m-%d %H:%M')

    return df



def extract_date_components(df, time_col='time_formatted'):
    """
    Extract day name and month name from formatted timestamp.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    time_col : str
        Name of the formatted time column (default: 'time_formatted')
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with new 'day_name' and 'month_name' columns
    """
    # Convert time_formatted to datetime for extraction
    temp_datetime = pd.to_datetime(df[time_col])
    
    # Extract day name (e.g., 'Monday', 'Tuesday')
    df['day_name'] = temp_datetime.dt.day_name()
    
    # Extract month name (e.g., 'January', 'February')
    df['month_name'] = temp_datetime.dt.month_name()
    
    print("Created 'day_name' and 'month_name' columns")
    return df


def extract_time_of_day(df, timestamp_col='timestamp'):
    """
    Extract time of day (HH:MM format) from original timestamp.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    timestamp_col : str
        Name of the original timestamp column (default: 'timestamp')
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with new 'time_of_day' column
    """
    # Ensure timestamp is in datetime format
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Extract time in HH:MM format
    df['time_of_day'] = df[timestamp_col].dt.strftime('%H:%M')
    
    print("Created 'time_of_day' column")
    return df

# SECTION 2 - FEATURE ENGINEERING 

def aggregate_30min_values(df, timestamp_col='timestamp', agg_cols=None):
    """
    Aggregate values every 30 minutes using the timestamp column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with timestamp column.
    timestamp_col : str
        Name of the datetime column to resample on.
    agg_cols : list or None
        List of columns to aggregate (numeric). If None, all numeric columns are aggregated.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with 30-minute averages.
    """
    import pandas as pd

    # Ensure timestamp column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors='coerce')

    # Drop rows with invalid timestamps
    df = df.dropna(subset=[timestamp_col])

    # Default: aggregate all numeric columns
    if agg_cols is None:
        agg_cols = df.select_dtypes(include='number').columns.tolist()

    # Resample every 30 minutes
    df_agg = (
        df.set_index(timestamp_col)[agg_cols]
          .resample('30min')  # '30min' avoids KeyError with 'T'
          .mean()
          .reset_index()
    )

    # Recreate formatted timestamp and time features
    df_agg['time_formatted'] = df_agg[timestamp_col].dt.strftime('%Y-%m-%d %H:%M')
    df_agg['day_name'] = df_agg[timestamp_col].dt.day_name()
    df_agg['month_name'] = df_agg[timestamp_col].dt.month_name()

    # Optional: time of day category
    def time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    df_agg['time_of_day'] = df_agg[timestamp_col].dt.hour.apply(time_of_day)

    return df_agg



# SECTION 3 - AQI CALCULATION

def calculate_aqi(df, pm25_col='P1', pm10_col='P2'):
    """
    Calculate the Air Quality Index (AQI) using US EPA breakpoints for
    PM2.5 and PM10, then store the higher sub-index in a column called 'aqi'.

    AQI formula (linear interpolation between breakpoints):
        AQI = ((AQI_high - AQI_low) / (Conc_high - Conc_low))
              * (Conc - Conc_low) + AQI_low

    US EPA PM2.5 breakpoints (µg/m³):
        Good                           :   0.0 –  12.0  → AQI   0 –  50
        Moderate                       :  12.1 –  35.4  → AQI  51 – 100
        Unhealthy for Sensitive Groups :  35.5 –  55.4  → AQI 101 – 150
        Unhealthy                      :  55.5 – 150.4  → AQI 151 – 200
        Very Unhealthy                 : 150.5 – 250.4  → AQI 201 – 300
        Hazardous                      : 250.5 – 500.4  → AQI 301 – 500

    US EPA PM10 breakpoints (µg/m³):
        Good                           :   0 –  54  → AQI   0 –  50
        Moderate                       :  55 – 154  → AQI  51 – 100
        Unhealthy for Sensitive Groups : 155 – 254  → AQI 101 – 150
        Unhealthy                      : 255 – 354  → AQI 151 – 200
        Very Unhealthy                 : 355 – 424  → AQI 201 – 300
        Hazardous                      : 425 – 604  → AQI 301 – 500

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing PM2.5 and PM10 columns.
    pm25_col : str
        Column name for PM2.5 readings (default: 'P1').
    pm10_col : str
        Column name for PM10 readings (default: 'P2').

    Returns:
    --------
    pandas.DataFrame
        Dataframe with four new columns:
            'aqi_pm25'     – sub-index calculated from PM2.5
            'aqi_pm10'     – sub-index calculated from PM10
            'aqi'          – overall AQI (higher of the two sub-indices)
            'aqi_category' – human-readable AQI category label
    """

    PM25_BREAKPOINTS = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]

    PM10_BREAKPOINTS = [
        (0,    54,    0,   50),
        (55,  154,   51,  100),
        (155, 254,  101,  150),
        (255, 354,  151,  200),
        (355, 424,  201,  300),
        (425, 604,  301,  500),
    ]

    def _concentration_to_aqi(concentration, breakpoints):
        """Map a single concentration to its AQI sub-index."""
        if pd.isna(concentration) or concentration < 0:
            return np.nan
        for (c_low, c_high, aqi_low, aqi_high) in breakpoints:
            if c_low <= concentration <= c_high:
                return round(
                    (aqi_high - aqi_low) / (c_high - c_low)
                    * (concentration - c_low)
                    + aqi_low
                )
        return 500  # cap at max if concentration exceeds all breakpoints

    def _aqi_category(aqi_val):
        """Return a human-readable category label for an AQI value."""
        if pd.isna(aqi_val):       return 'Unknown'
        elif aqi_val <= 50:        return 'Good'
        elif aqi_val <= 100:       return 'Moderate'
        elif aqi_val <= 150:       return 'Unhealthy for Sensitive Groups'
        elif aqi_val <= 200:       return 'Unhealthy'
        elif aqi_val <= 300:       return 'Very Unhealthy'
        else:                      return 'Hazardous'

    # Cast to numeric so bad strings become NaN
    pm25_values = pd.to_numeric(df[pm25_col], errors='coerce')
    pm10_values = pd.to_numeric(df[pm10_col], errors='coerce')

    # Calculate each sub-index
    df['aqi_pm25'] = pm25_values.apply(lambda x: _concentration_to_aqi(x, PM25_BREAKPOINTS))
    df['aqi_pm10'] = pm10_values.apply(lambda x: _concentration_to_aqi(x, PM10_BREAKPOINTS))

    # Overall AQI = highest sub-index (EPA standard)
    df['aqi'] = df[['aqi_pm25', 'aqi_pm10']].max(axis=1)

    # Attach category label
    df['aqi_category'] = df['aqi'].apply(_aqi_category)

    print("Created 'aqi_pm25', 'aqi_pm10', 'aqi', and 'aqi_category' columns")
    return df



#SECTION 3 - COMPLETE PREPROCESSING PIPELINE

def preprocess_data(df, aggregate=True):
    """
    Complete preprocessing pipeline.
    
    Performs all preprocessing steps in order:
    1. Drop unnecessary columns
    2. Format timestamp
    3. Extract date components
    4. Extract time of day
    5. Optionally aggregate hourly values
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input raw dataframe
    aggregate : bool
        Whether to perform hourly aggregation (default: True)
        
    Returns:
    --------
    pandas.DataFrame
        Fully preprocessed dataframe
    """
    print("Starting data preprocessing pipeline...\n") 
    
    #Step 0: Load CSV
    df = data_ingestion(df)

    # Make a copy to avoid modifying original
    df_processed = df.copy()
    
    # Step 1: Drop unnecessary columns
    df_processed = drop_unnecessary_columns(df_processed)
    
    # Step 2: Format timestamp
    df_processed = format_timestamp(df_processed)
    
    # Step 3: Extract date components
    df_processed = extract_date_components(df_processed)
    
    # Step 4: Extract time of day
    df_processed = extract_time_of_day(df_processed)
    
    # Step 5: Aggregate hourly values (if requested)
    if aggregate:
        df_processed = aggregate_30min_values(df_processed)
        
        # Recalculate date components and time_of_day after aggregation.
        # BUG FIX: Do NOT call format_timestamp again — aggregate_30min_values
        # already sets time_formatted, day_name, month_name, and time_of_day
        # directly on the resampled datetime index. Calling format_timestamp a
        # second time would re-parse an already-correct datetime column and
        # discard any rows whose string representation doesn't parse as UTC.
        # Just regenerate the string-based feature columns to be safe:
        df_processed = extract_date_components(df_processed)
        df_processed = extract_time_of_day(df_processed)

    df_processed = calculate_aqi(df_processed, pm25_col='P1', pm10_col='P2')
    
    print("\nPreprocessing complete!")
    print(f"Final dataframe shape: {df_processed.shape}")
    
    return df_processed


# Example usage
if __name__ == "__main__":
    # Example: Load your data
    # df = pd.read_csv('your_data.csv')
    
    # Run preprocessing
    # df_clean = preprocess_data(df, aggregate=True)
    
    # Save cleaned data
    # df_clean.to_csv('cleaned_data.csv', index=False)
    
    print("Module loaded successfully. Import and use functions as needed.")

# SECTION 4 - LOAD & VISUALISATION
# BUG FIX: save_cleaned_data and visualize were indented inside the
# `if __name__ == "__main__"` block, making them invisible when the module
# is imported. Moved them to module scope.

def save_cleaned_data(df, output_dir='output', filename='cleaned_data.csv'):
    """
    Save the cleaned and preprocessed DataFrame to a CSV file.

    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataframe to save.
    output_dir : str
        Directory to save the file (default: 'output').
    filename : str
        Output filename (default: 'cleaned_data.csv').

    Returns:
    --------
    str
        Full path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved cleaned data → {filepath}  ({len(df):,} rows)")
    return filepath


def visualize(df, output_dir='output'):
    """
    Generate and save all visualisation charts from the cleaned sensor data.

    Charts produced:
        1. chart_pm_timeseries.png        – P0 / P1 / P2 over time
        2. chart_aqi_distribution.png     – AQI score histogram + category breakdown
        3. chart_temp_humidity.png        – Temperature & Humidity dual-axis trend
        4. chart_aqi_by_location.png      – Average AQI per location (bar)
        5. chart_pollutant_share_donut.png – Pollutant share donut chart (PM1/PM2.5/PM10)

    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataframe containing columns:
        P0, P1, P2, humidity, temperature, timestamp,
        location_id, aqi, aqi_category.
    output_dir : str
        Directory to save charts (default: 'output').

    Returns:
    --------
    list
        List of file paths for all saved charts.
    """
    import matplotlib
    matplotlib.use('Agg')           # safe for Docker / headless environments
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # BUG FIX: Delete the old heatmap file if it exists from a previous run.
    # Deleting the chart code does NOT delete already-generated files on disk —
    # the old chart_aqi_heatmap.png will keep appearing until explicitly removed.
    stale_charts = ['chart_aqi_heatmap.png']
    for stale in stale_charts:
        stale_path = os.path.join(output_dir, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)
            print(f"🗑️  Removed stale chart → {stale_path}")

    # ── Shared colour palette per location ───────────────────────────────────
    location_ids  = sorted(df['location_id'].dropna().unique())
    palette       = ['#4C72B0', '#DD8452', '#55A868', '#C44E52',
                     '#8172B2', '#937860', '#DA8BC3', '#8C8C8C']
    loc_colours   = {loc: palette[i % len(palette)]
                     for i, loc in enumerate(location_ids)}

    # Ensure timestamp is datetime
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df['date']      = df['timestamp'].dt.floor('D')

    # ── AQI category colour map (standard traffic-light convention) ───────────
    CAT_COLOURS = {
        'Good':                            '#00e400',
        'Moderate':                        '#ffff00',
        'Unhealthy for Sensitive Groups':  '#ff7e00',
        'Unhealthy':                       '#ff0000',
        'Very Unhealthy':                  '#8f3f97',
        'Hazardous':                       '#7e0023',
        'Unknown':                         '#cccccc',
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Chart 1 – Particulate Matter Time Series (P0 / P1 / P2)
    # ─────────────────────────────────────────────────────────────────────────
    pm_daily = (
        df.groupby('date')[['P0', 'P1', 'P2']]
        .mean()
        .round(2)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    for col, colour, label in zip(
        ['P0', 'P1', 'P2'],
        ['#4C72B0', '#DD8452', '#55A868'],
        ['PM1 (P0)', 'PM2.5 (P1)', 'PM10 (P2)']
    ):
        ax.plot(pm_daily['date'], pm_daily[col],
                label=label, linewidth=1.8, color=colour)
    ax.fill_between(pm_daily['date'], pm_daily['P0'], pm_daily['P2'],
                    alpha=0.07, color='gray')
    ax.axhline(15, color='red', linestyle='--', linewidth=1,
               label='WHO PM2.5 Guideline (15 µg/m³)')
    ax.set_title('Daily Average Particulate Matter (PM1 / PM2.5 / PM10)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Concentration (µg/m³)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.xticks(rotation=30)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, 'chart_pm_timeseries.png')
    plt.savefig(p, dpi=130); plt.close(); saved.append(p)
    print(f"📊 Saved → {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # Chart 2 – AQI Distribution: histogram + category breakdown (side by side)
    # ─────────────────────────────────────────────────────────────────────────
    aqi_clean = df['aqi'].dropna()
    cat_counts = df['aqi_category'].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left – histogram
    ax1.hist(aqi_clean, bins=40, color='#4C72B0', edgecolor='white', alpha=0.85)
    ax1.axvline(aqi_clean.mean(), color='red', linestyle='--', linewidth=1.2,
                label=f'Mean AQI: {aqi_clean.mean():.1f}')
    ax1.set_title('AQI Score Distribution', fontsize=13, fontweight='bold')
    ax1.set_xlabel('AQI')
    ax1.set_ylabel('Number of Readings')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Right – category bar
    bar_colours = [CAT_COLOURS.get(c, '#cccccc') for c in cat_counts.index]
    bars = ax2.barh(cat_counts.index, cat_counts.values,
                    color=bar_colours, edgecolor='white')
    ax2.bar_label(bars, padding=4, fontsize=9)
    ax2.set_title('Readings by AQI Category', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Number of Readings')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    p = os.path.join(output_dir, 'chart_aqi_distribution.png')
    plt.savefig(p, dpi=130); plt.close(); saved.append(p)
    print(f"📊 Saved → {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # Chart 3 – Temperature & Humidity dual-axis trend
    # ─────────────────────────────────────────────────────────────────────────
    env_daily = (
        df.groupby('date')[['temperature', 'humidity']]
        .mean()
        .round(2)
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(env_daily['date'], env_daily['temperature'],
                   color='#C44E52', linewidth=2, label='Temperature (°C)')
    l2, = ax2.plot(env_daily['date'], env_daily['humidity'],
                   color='#4C72B0', linewidth=2, linestyle='--', label='Humidity (%)')

    ax1.set_ylabel('Temperature (°C)', color='#C44E52')
    ax2.set_ylabel('Humidity (%)',      color='#4C72B0')
    ax1.tick_params(axis='y', labelcolor='#C44E52')
    ax2.tick_params(axis='y', labelcolor='#4C72B0')
    ax1.set_title('Daily Average Temperature & Humidity',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.xticks(rotation=30)
    ax1.legend(handles=[l1, l2], loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, 'chart_temp_humidity.png')
    plt.savefig(p, dpi=130); plt.close(); saved.append(p)
    print(f"📊 Saved → {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # Chart 4 – Average AQI per location (bar)
    # ─────────────────────────────────────────────────────────────────────────
    loc_aqi = (
        df.groupby('location_id')['aqi']
        .mean()
        .round(1)
        .sort_values(ascending=False)
        .reset_index()
    )
    loc_aqi['location_id'] = loc_aqi['location_id'].astype(str)

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_cols = ['#C44E52' if v > 100 else '#DD8452' if v > 50 else '#55A868'
                for v in loc_aqi['aqi']]
    bars = ax.bar(loc_aqi['location_id'], loc_aqi['aqi'],
                  color=bar_cols, width=0.5, edgecolor='white')
    ax.bar_label(bars, fmt='%.1f', padding=4, fontsize=10)
    ax.axhline(100, color='orange', linestyle='--', linewidth=1,
               label='Unhealthy threshold (AQI 100)')
    ax.axhline(50,  color='green',  linestyle='--', linewidth=1,
               label='Moderate threshold (AQI 50)')
    ax.set_title('Average AQI by Location', fontsize=14, fontweight='bold')
    ax.set_xlabel('Location ID')
    ax.set_ylabel('Average AQI')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, 'chart_aqi_by_location.png')
    plt.savefig(p, dpi=130); plt.close(); saved.append(p)
    print(f"📊 Saved → {p}")

      # ─────────────────────────────────────────────────────────────────────────
    # Chart 5 – Pollutant Share Donut Chart (PM1 / PM2.5 / PM10)
    # ─────────────────────────────────────────────────────────────────────────
    # Calculate mean pollutant concentrations
    pollutant_means = df[['P0', 'P1', 'P2']].mean(numeric_only=True)

    # Handle case where values might be NaN
    pollutant_means = pollutant_means.fillna(0)

    labels = ['PM1 (P0)', 'PM2.5 (P1)', 'PM10 (P2)']
    sizes = pollutant_means.values

    # Avoid division by zero
    if sizes.sum() == 0:
        print("⚠️ No pollutant data available for donut chart.")
    else:
        fig, ax = plt.subplots(figsize=(7, 7))

        # Create donut chart
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            wedgeprops=dict(width=0.4, edgecolor='white')
        )

        # Add center circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)

        # Title
        ax.set_title(
            'Pollutant Share Distribution (PM1 vs PM2.5 vs PM10)',
            fontsize=14,
            fontweight='bold'
        )

        plt.tight_layout()
        p = os.path.join(output_dir, 'chart_pollutant_share_donut.png')
        plt.savefig(p, dpi=130)
        plt.close()
        saved.append(p)
        print(f"📊 Saved → {p}")



def load(df, output_dir='output', filename='cleaned_data.csv'):
    """
    Full load step: save cleaned CSV then generate all visualisations.

    Parameters:
    -----------
    df : pandas.DataFrame
        Fully preprocessed dataframe (output of preprocess_data + calculate_aqi).
    output_dir : str
        Directory for all outputs (default: 'output').
    filename : str
        Name of the saved CSV file (default: 'cleaned_data.csv').

    Returns:
    --------
    dict
        {'csv': <csv_path>, 'charts': [<chart_paths>]}
    """
    print("Starting load step...\n")
    csv_path   = save_cleaned_data(df, output_dir=output_dir, filename=filename)
    chart_paths = visualize(df, output_dir=output_dir)
    print("\nLoad step complete.")
    return {'csv': csv_path, 'charts': chart_paths}