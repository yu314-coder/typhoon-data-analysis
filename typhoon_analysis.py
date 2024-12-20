import dash
import plotly.graph_objects as go
import plotly.express as px
import pickle
import tropycal.tracks as tracks
import pandas as pd
import numpy as np
import cachetools
import functools
import hashlib
import os
import argparse
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from datetime import date, datetime
from scipy import stats
from scipy.optimize import minimize, curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import schedule
import time
import threading
import requests
from io import StringIO   
import tempfile
import csv  
from collections import defaultdict
import shutil
import filecmp

# Add command-line argument parsing
parser = argparse.ArgumentParser(description='Typhoon Analysis Dashboard')
parser.add_argument('--data_path', type=str, default=os.getcwd(), help='Path to the data directory')
args = parser.parse_args()

# Use the command-line argument for data path
DATA_PATH = args.data_path

ONI_DATA_PATH = os.path.join(DATA_PATH, 'oni_data.csv')
TYPHOON_DATA_PATH = os.path.join(DATA_PATH, 'processed_typhoon_data.csv')
LOCAL_iBtrace_PATH =  os.path.join(DATA_PATH, 'ibtracs.WP.list.v04r01.csv')
iBtrace_uri = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv'

CACHE_FILE = 'ibtracs_cache.pkl'
CACHE_EXPIRY_DAYS = 1
last_oni_update = None


def should_update_oni():
    today = datetime.now()
    # Beginning of the month: 1st day
    if today.day == 1:
        return True
    # Middle of the month: 15th day
    if today.day == 15:
        return True
    # End of the month: last day
    if today.day == (today.replace(day=1, month=today.month%12+1) - timedelta(days=1)).day:
        return True
    return False

color_map = {
    'C5 Super Typhoon': 'rgb(255, 0, 0)',      # Red
    'C4 Very Strong Typhoon': 'rgb(255, 63, 0)', # Red-Orange
    'C3 Strong Typhoon': 'rgb(255, 127, 0)',    # Orange
    'C2 Typhoon': 'rgb(255, 191, 0)',          # Orange-Yellow
    'C1 Typhoon': 'rgb(255, 255, 0)',          # Yellow
    'Tropical Storm': 'rgb(0, 255, 255)',       # Cyan
    'Tropical Depression': 'rgb(173, 216, 230)' # Light Blue
}

def convert_typhoondata(input_file, output_file):
    with open(input_file, 'r') as infile:
        # Skip the title and the unit line.
        next(infile)
        next(infile)
        
        reader = csv.reader(infile)
        
        # Used for storing data for each SID
        sid_data = defaultdict(list)
        
        for row in reader:
            if not row:  # Skip the blank lines
                continue
            
            sid = row[0]
            iso_time = row[6]
            sid_data[sid].append((row, iso_time))

    with open(output_file, 'w', newline='') as outfile:
        fieldnames = ['SID', 'ISO_TIME', 'LAT', 'LON', 'SEASON', 'NAME', 'WMO_WIND', 'WMO_PRES', 'USA_WIND', 'USA_PRES', 'START_DATE', 'END_DATE']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for sid, data in sid_data.items():
            start_date = min(data, key=lambda x: x[1])[1]
            end_date = max(data, key=lambda x: x[1])[1]
            
            for row, iso_time in data:
                writer.writerow({
                    'SID': row[0],
                    'ISO_TIME': iso_time,
                    'LAT': row[8],
                    'LON': row[9],
                    'SEASON': row[1],
                    'NAME': row[5],
                    'WMO_WIND': row[10].strip() or ' ',  
                    'WMO_PRES': row[11].strip() or ' ',
                    'USA_WIND': row[23].strip() or ' ',
                    'USA_PRES': row[24].strip() or ' ',
                    'START_DATE': start_date,
                    'END_DATE': end_date
                })


def download_oni_file(url, filename):
    print(f"Downloading file from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an exception for non-200 status codes
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"File successfully downloaded and saved as {filename}")
        return True
    except requests.RequestException as e:
        print(f"Download failed. Error: {e}")
        return False


def convert_oni_ascii_to_csv(input_file, output_file):
    data = defaultdict(lambda: [''] * 12)
    season_to_month = {
        'DJF': 12, 'JFM': 1, 'FMA': 2, 'MAM': 3, 'AMJ': 4, 'MJJ': 5,
        'JJA': 6, 'JAS': 7, 'ASO': 8, 'SON': 9, 'OND': 10, 'NDJ': 11
    }
    
    print(f"Attempting to read file: {input_file}")
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
            print(f"Successfully read {len(lines)} lines")
            
            if len(lines) <= 1:
                print("Error: File is empty or contains only header")
                return
            
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 4:
                    season, year = parts[0], parts[1]
                    anom = parts[-1]
                    
                    if season in season_to_month:
                        month = season_to_month[season]
                        
                        if season == 'DJF':
                            year = str(int(year) - 1)
                        
                        data[year][month-1] = anom
                    else:
                        print(f"Warning: Unknown season: {season}")
                else:
                    print(f"Warning: Skipping invalid line: {line.strip()}")
            
            print(f"Processed data for {len(data)} years")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Attempting to write file: {output_file}")
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            
            for year in sorted(data.keys()):
                row = [year] + data[year]
                writer.writerow(row)
            
            print(f"Successfully wrote {len(data)} rows of data")
    except Exception as e:
        print(f"Error writing file: {e}")
        return

    print(f"Conversion complete. Data saved to {output_file}")

def update_oni_data():
    global last_oni_update
    current_date = date.today()
    
    # Check if already updated today
    if last_oni_update == current_date:
        print("ONI data already checked today. Skipping update.")
        return
    
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    temp_file = os.path.join(DATA_PATH, "temp_oni.ascii.txt")
    input_file = os.path.join(DATA_PATH, "oni.ascii.txt")
    output_file = ONI_DATA_PATH
    
    if download_oni_file(url, temp_file):
        if not os.path.exists(input_file) or not filecmp.cmp(temp_file, input_file, shallow=False):
            # File doesn't exist or has been updated
            os.replace(temp_file, input_file)
            print("New ONI data detected. Converting to CSV.")
            convert_oni_ascii_to_csv(input_file, output_file)
            print("ONI data updated successfully.")
        else:
            print("ONI data is up to date. No conversion needed.")
            os.remove(temp_file)  # Remove temporary file
        
        last_oni_update = current_date
    else:
        print("Failed to download ONI data.")
        if os.path.exists(temp_file):
            os.remove(temp_file)  # Ensure cleanup of temporary file

def load_ibtracs_data():
    if os.path.exists(CACHE_FILE):
        cache_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if datetime.now() - cache_time < timedelta(days=CACHE_EXPIRY_DAYS):
            print("Loading data from cache...")
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
    
    if os.path.exists(LOCAL_iBtrace_PATH):
        print("Using local IBTrACS file...")
        ibtracs = tracks.TrackDataset(basin='west_pacific', source='ibtracs', ibtracs_url=LOCAL_iBtrace_PATH)
    else:
        print("Local IBTrACS file not found. Fetching data from remote server...")
        try:
            response = requests.get(iBtrace_uri)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
                temp_file.write(response.text)
                temp_file_path = temp_file.name
            
            # Save the downloaded data as the local file
            shutil.move(temp_file_path, LOCAL_iBtrace_PATH)
            print(f"Downloaded data saved to {LOCAL_iBtrace_PATH}")
            
            ibtracs = tracks.TrackDataset(basin='west_pacific', source='ibtracs', ibtracs_url=LOCAL_iBtrace_PATH)
        except requests.RequestException as e:
            print(f"Error downloading data: {e}")
            print("No local file available and download failed. Unable to load IBTrACS data.")
            return None
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(ibtracs, f)
    
    return ibtracs
    
def update_ibtracs_data():
    global ibtracs
    print("Checking for IBTrACS data updates...")

    try:
        # Get the last-modified time of the remote file
        response = requests.head(iBtrace_uri)
        remote_last_modified = datetime.strptime(response.headers['Last-Modified'], '%a, %d %b %Y %H:%M:%S GMT')

        # Get the last-modified time of the local file
        if os.path.exists(LOCAL_iBtrace_PATH):
            local_last_modified = datetime.fromtimestamp(os.path.getmtime(LOCAL_iBtrace_PATH))
        else:
            local_last_modified = datetime.min

        # Compare the modification times
        if remote_last_modified <= local_last_modified:
            print("Local IBTrACS data is up to date. No update needed.")
            if os.path.exists(CACHE_FILE):
                # Update the cache file's timestamp to extend its validity
                os.utime(CACHE_FILE, None)
                print("Cache file timestamp updated.")
            return

        print("Remote data is newer. Updating IBTrACS data...")
        
        # Download the new data
        response = requests.get(iBtrace_uri)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            temp_file.write(response.text)
            temp_file_path = temp_file.name
        
        # Save the downloaded data as the local file
        shutil.move(temp_file_path, LOCAL_iBtrace_PATH)
        print(f"Downloaded data saved to {LOCAL_iBtrace_PATH}")
        
        # Update the last modified time of the local file to match the remote file
        os.utime(LOCAL_iBtrace_PATH, (remote_last_modified.timestamp(), remote_last_modified.timestamp()))
        
        ibtracs = tracks.TrackDataset(basin='west_pacific', source='ibtracs', ibtracs_url=LOCAL_iBtrace_PATH)
        
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(ibtracs, f)
        print("IBTrACS data updated and cache refreshed.")

    except requests.RequestException as e:
        print(f"Error checking or downloading data: {e}")
        if os.path.exists(LOCAL_iBtrace_PATH):
            print("Using existing local file.")
            ibtracs = tracks.TrackDataset(basin='west_pacific', source='ibtracs', ibtracs_url=LOCAL_iBtrace_PATH)
            if os.path.exists(CACHE_FILE):
                # Update the cache file's timestamp even when using existing local file
                os.utime(CACHE_FILE, None)
                print("Cache file timestamp updated.")
        else:
            print("No local file available. Update failed.")

def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

def analyze_typhoon_generation(merged_data, start_date, end_date):
    filtered_data = merged_data[
        (merged_data['ISO_TIME'] >= start_date) & 
        (merged_data['ISO_TIME'] <= end_date)
    ]
    
    filtered_data['ENSO_Phase'] = filtered_data['ONI'].apply(classify_enso_phases)
    
    typhoon_counts = filtered_data['ENSO_Phase'].value_counts().to_dict()
    
    month_counts = filtered_data.groupby(['ENSO_Phase', filtered_data['ISO_TIME'].dt.month]).size().unstack(fill_value=0)
    concentrated_months = month_counts.idxmax(axis=1).to_dict()
    
    return typhoon_counts, concentrated_months

def cache_key_generator(*args, **kwargs):
    key = hashlib.md5()
    for arg in args:
        key.update(str(arg).encode())
    for k, v in sorted(kwargs.items()):
        key.update(str(k).encode())
        key.update(str(v).encode())
    return key.hexdigest()

def categorize_typhoon(wind_speed):
    wind_speed_kt = wind_speed / 2  # Convert kt to m/s
    
    # Add category classification
    if wind_speed_kt >= 137/2.35:
        return 'C5 Super Typhoon'
    elif wind_speed_kt >= 113/2.35:
        return 'C4 Very Strong Typhoon' 
    elif wind_speed_kt >= 96/2.35:
        return 'C3 Strong Typhoon'
    elif wind_speed_kt >= 83/2.35:
        return 'C2 Typhoon'
    elif wind_speed_kt >= 64/2.35:
        return 'C1 Typhoon'
    elif wind_speed_kt >= 34/2.35:
        return 'Tropical Storm'
    else:
        return 'Tropical Depression'

@functools.lru_cache(maxsize=None)
def process_oni_data_cached(oni_data_hash):
    return process_oni_data(oni_data)

def process_oni_data(oni_data):
    oni_long = oni_data.melt(id_vars=['Year'], var_name='Month', value_name='ONI')
    oni_long['Month'] = oni_long['Month'].map({
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    })
    oni_long['Date'] = pd.to_datetime(oni_long['Year'].astype(str) + '-' + oni_long['Month'] + '-01')
    oni_long['ONI'] = pd.to_numeric(oni_long['ONI'], errors='coerce')
    return oni_long

def process_oni_data_with_cache(oni_data):
    oni_data_hash = cache_key_generator(oni_data.to_json())
    return process_oni_data_cached(oni_data_hash)

@functools.lru_cache(maxsize=None)
def process_typhoon_data_cached(typhoon_data_hash):
    return process_typhoon_data(typhoon_data)

def process_typhoon_data(typhoon_data):
    typhoon_data['ISO_TIME'] = pd.to_datetime(typhoon_data['ISO_TIME'], errors='coerce')
    typhoon_data['USA_WIND'] = pd.to_numeric(typhoon_data['USA_WIND'], errors='coerce')
    typhoon_data['USA_PRES'] = pd.to_numeric(typhoon_data['USA_PRES'], errors='coerce')
    typhoon_data['LON'] = pd.to_numeric(typhoon_data['LON'], errors='coerce')
    
    typhoon_max = typhoon_data.groupby('SID').agg({
        'USA_WIND': 'max',
        'USA_PRES': 'min',
        'ISO_TIME': 'first',
        'SEASON': 'first',
        'NAME': 'first',
        'LAT': 'first',
        'LON': 'first'
    }).reset_index()
    
    typhoon_max['Month'] = typhoon_max['ISO_TIME'].dt.strftime('%m')
    typhoon_max['Year'] = typhoon_max['ISO_TIME'].dt.year
    typhoon_max['Category'] = typhoon_max['USA_WIND'].apply(categorize_typhoon)
    return typhoon_max

def process_typhoon_data_with_cache(typhoon_data):
    typhoon_data_hash = cache_key_generator(typhoon_data.to_json())
    return process_typhoon_data_cached(typhoon_data_hash)

def merge_data(oni_long, typhoon_max):
    return pd.merge(typhoon_max, oni_long, on=['Year', 'Month'])

def calculate_logistic_regression(merged_data):
    data = merged_data.dropna(subset=['USA_WIND', 'ONI'])
    
    # Create binary outcome for severe typhoons
    data['severe_typhoon'] = (data['USA_WIND'] >= 51).astype(int)
    
    # Create binary predictor for El Niño
    data['el_nino'] = (data['ONI'] >= 0.5).astype(int)
    
    X = data['el_nino']
    X = sm.add_constant(X)  # Add constant term
    y = data['severe_typhoon']
    
    model = sm.Logit(y, X).fit()
    
    beta_1 = model.params['el_nino']
    exp_beta_1 = np.exp(beta_1)
    p_value = model.pvalues['el_nino']
    
    return beta_1, exp_beta_1, p_value

@cachetools.cached(cache={})
def fetch_oni_data_from_csv(file_path):
    df = pd.read_csv(file_path, sep=',', header=0, na_values='-99.90')
    df.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df = df.melt(id_vars=['Year'], var_name='Month', value_name='ONI')
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + df['Month'], format='%Y%b')
    df = df.set_index('Date')
    return df

def classify_enso_phases(oni_value):
    if isinstance(oni_value, pd.Series):
        oni_value = oni_value.iloc[0]
    if oni_value >= 0.5:
        return 'El Nino'
    elif oni_value <= -0.5:
        return 'La Nina'
    else:
        return 'Neutral'

def load_data(oni_data_path, typhoon_data_path):
    oni_data = pd.read_csv(oni_data_path)
    typhoon_data = pd.read_csv(typhoon_data_path, low_memory=False)
    
    typhoon_data['ISO_TIME'] = pd.to_datetime(typhoon_data['ISO_TIME'], errors='coerce')
    
    typhoon_data = typhoon_data.dropna(subset=['ISO_TIME'])
    
    print(f"Typhoon data shape after cleaning: {typhoon_data.shape}")
    print(f"Year range: {typhoon_data['ISO_TIME'].dt.year.min()} - {typhoon_data['ISO_TIME'].dt.year.max()}")
    
    return oni_data, typhoon_data

def preprocess_data(oni_data, typhoon_data):
    typhoon_data['USA_WIND'] = pd.to_numeric(typhoon_data['USA_WIND'], errors='coerce')
    typhoon_data['WMO_PRES'] = pd.to_numeric(typhoon_data['WMO_PRES'], errors='coerce')
    typhoon_data['ISO_TIME'] = pd.to_datetime(typhoon_data['ISO_TIME'], errors='coerce')
    typhoon_data['Year'] = typhoon_data['ISO_TIME'].dt.year
    typhoon_data['Month'] = typhoon_data['ISO_TIME'].dt.month
    
    monthly_max_wind_speed = typhoon_data.groupby(['Year', 'Month'])['USA_WIND'].max().reset_index()
    
    oni_data_long = pd.melt(oni_data, id_vars=['Year'], var_name='Month', value_name='ONI')
    oni_data_long['Month'] = oni_data_long['Month'].apply(lambda x: pd.to_datetime(x, format='%b').month)
    
    merged_data = pd.merge(monthly_max_wind_speed, oni_data_long, on=['Year', 'Month'])
    
    return merged_data

def calculate_max_wind_min_pressure(typhoon_data):
    max_wind_speed = typhoon_data['USA_WIND'].max()
    min_pressure = typhoon_data['WMO_PRES'].min()
    return max_wind_speed, min_pressure

@functools.lru_cache(maxsize=None)
def get_storm_data(storm_id):
    return ibtracs.get_storm(storm_id)

def filter_west_pacific_coordinates(lons, lats):
    mask = (100 <= lons) & (lons <= 180) & (0 <= lats) & (lats <= 40)
    return lons[mask], lats[mask]

def polynomial_exp(x, a, b, c, d):
    return a * x**2 + b * x + c + d * np.exp(x)

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def generate_cluster_equations(cluster_center):
    X = cluster_center[:, 0]  # Longitudes
    y = cluster_center[:, 1]  # Latitudes
    
    x_min = X.min()
    x_max = X.max()
    
    equations = []

    # Fourier Series (up to 4th order)
    def fourier_series(x, a0, a1, b1, a2, b2, a3, b3, a4, b4):
        return (a0 + a1*np.cos(x) + b1*np.sin(x) + 
                a2*np.cos(2*x) + b2*np.sin(2*x) + 
                a3*np.cos(3*x) + b3*np.sin(3*x) + 
                a4*np.cos(4*x) + b4*np.sin(4*x))

    # Normalize X to the range [0, 2π]
    X_normalized = 2 * np.pi * (X - x_min) / (x_max - x_min)

    params, _ = curve_fit(fourier_series, X_normalized, y)
    a0, a1, b1, a2, b2, a3, b3, a4, b4 = params
    
    # Create the equation string
    fourier_eq = (f"y = {a0:.4f} + {a1:.4f}*cos(x) + {b1:.4f}*sin(x) + "
                  f"{a2:.4f}*cos(2x) + {b2:.4f}*sin(2x) + "
                  f"{a3:.4f}*cos(3x) + {b3:.4f}*sin(3x) + "
                  f"{a4:.4f}*cos(4x) + {b4:.4f}*sin(4x)")
    
    equations.append(("Fourier Series", fourier_eq))
    equations.append(("X Range", f"x goes from 0 to {2*np.pi:.4f}"))
    equations.append(("Longitude Range", f"Longitude goes from {x_min:.4f}°E to {x_max:.4f}°E"))

    return equations, (x_min, x_max)
    
#oni_df = fetch_oni_data_from_csv(ONI_DATA_PATH)
#ibtracs = load_ibtracs_data()
#oni_data, typhoon_data = load_data(ONI_DATA_PATH, TYPHOON_DATA_PATH)
#oni_long = process_oni_data_with_cache(oni_data)
#typhoon_max = process_typhoon_data_with_cache(typhoon_data)
#merged_data = merge_data(oni_long, typhoon_max)
#data = preprocess_data(oni_data, typhoon_data)
#max_wind_speed, min_pressure = calculate_max_wind_min_pressure(typhoon_data)
#
## Schedule the update to run daily at 1:00 AM
#schedule.every().day.at("01:00").do(update_ibtracs_data)
#
## Run the scheduler in a separate thread
#scheduler_thread = threading.Thread(target=run_schedule)
#scheduler_thread.start()


app = dash.Dash(__name__)

# First, add the classification standards
atlantic_standard = {
    'C5 Super Typhoon': {'wind_speed': 137, 'color': 'rgb(255, 0, 0)'},      
    'C4 Very Strong Typhoon': {'wind_speed': 113, 'color': 'rgb(255, 63, 0)'}, 
    'C3 Strong Typhoon': {'wind_speed': 96, 'color': 'rgb(255, 127, 0)'},    
    'C2 Typhoon': {'wind_speed': 83, 'color': 'rgb(255, 191, 0)'},          
    'C1 Typhoon': {'wind_speed': 64, 'color': 'rgb(255, 255, 0)'},          
    'Tropical Storm': {'wind_speed': 34, 'color': 'rgb(0, 255, 255)'},       
    'Tropical Depression': {'wind_speed': 0, 'color': 'rgb(173, 216, 230)'}  
}

taiwan_standard = {
    'Strong Typhoon': {'wind_speed': 51.0, 'color': 'rgb(255, 0, 0)'},       # >= 51.0 m/s
    'Medium Typhoon': {'wind_speed': 33.7, 'color': 'rgb(255, 127, 0)'},     # 33.7-50.9 m/s
    'Mild Typhoon': {'wind_speed': 17.2, 'color': 'rgb(255, 255, 0)'},       # 17.2-33.6 m/s
    'Tropical Depression': {'wind_speed': 0, 'color': 'rgb(173, 216, 230)'}  # < 17.2 m/s
}

app.layout = html.Div([
    html.H1("Typhoon Analysis Dashboard"),
    
    html.Div([
        dcc.Input(id='start-year', type='number', placeholder='Start Year', value=2000, min=1900, max=2024, step=1),
        dcc.Input(id='start-month', type='number', placeholder='Start Month', value=1, min=1, max=12, step=1),
        dcc.Input(id='end-year', type='number', placeholder='End Year', value=2024, min=1900, max=2024, step=1),
        dcc.Input(id='end-month', type='number', placeholder='End Month', value=6, min=1, max=12, step=1),
        dcc.Dropdown(
            id='enso-dropdown',
            options=[
                {'label': 'All Years', 'value': 'all'},
                {'label': 'El Niño Years', 'value': 'el_nino'},
                {'label': 'La Niña Years', 'value': 'la_nina'},
                {'label': 'Neutral Years', 'value': 'neutral'}
            ],
            value='all'
        ),
        html.Button('Analyze', id='analyze-button', n_clicks=0),
    ]),
    
    html.Div([
        dcc.Input(id='typhoon-search', type='text', placeholder='Search Typhoon Name'),
        html.Button('Find Typhoon', id='find-typhoon-button', n_clicks=0),
    ]),
    
    html.Div([
        html.Div(id='correlation-coefficient'),
        html.Div(id='max-wind-speed'),
        html.Div(id='min-pressure'),
    ]),
    
    dcc.Graph(id='typhoon-tracks-graph'),
    html.Div([
        html.P("Number of Clusters"),
        dcc.Input(id='n-clusters', type='number', placeholder='Number of Clusters', value=5, min=1, max=20, step=1),
        html.Button('Show Clusters', id='show-clusters-button', n_clicks=0),
        html.Button('Show Typhoon Routes', id='show-routes-button', n_clicks=0),
    ]),

    dcc.Graph(id='typhoon-routes-graph'),
    
    html.Div([
    html.Button('Fourier Series', id='fourier-series-button', n_clicks=0),
    ]),
    html.Div(id='cluster-equation-results'),
    
    html.Div([
        html.Button('Wind Speed Logistic Regression', id='wind-regression-button', n_clicks=0),
        html.Button('Pressure Logistic Regression', id='pressure-regression-button', n_clicks=0),
        html.Button('Longitude Logistic Regression', id='longitude-regression-button', n_clicks=0),
    ]),
    html.Div(id='logistic-regression-results'),

    html.H2("Typhoon Path Analysis"),
    html.Div([
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(year), 'value': year} for year in range(1950, 2025)],
            value=2024,
            style={'width': '200px'}
        ),
        dcc.Dropdown(
            id='typhoon-dropdown',
            style={'width': '300px'}
        ),
        dcc.Dropdown(
            id='classification-standard',
            options=[
                {'label': 'Atlantic Standard', 'value': 'atlantic'},
                {'label': 'Taiwan Standard', 'value': 'taiwan'}
            ],
            value='atlantic',
            style={'width': '200px'}
        )
    ], style={'display': 'flex', 'gap': '10px'}),
    
    dcc.Graph(id='typhoon-path-animation'),
    dcc.Graph(id='all-years-regression-graph'),
    dcc.Graph(id='wind-oni-scatter-plot'),
    dcc.Graph(id='pressure-oni-scatter'),
       
    html.Div(id='regression-graphs'),
    html.Div(id='slopes'),
    html.Div([
        html.H3("Correlation Analysis"),
        html.Div(id='wind-oni-correlation'),
        html.Div(id='pressure-oni-correlation'),
    ]),
    html.Div([
        html.H3("Typhoon Generation Analysis"),
        html.Div(id='typhoon-count-analysis'),
        html.Div(id='concentrated-months-analysis'),
    ]),
    html.Div(id='cluster-info'),
    
    html.Div([
        dcc.Dropdown(
            id='classification-standard',
            options=[
                {'label': 'Atlantic Standard', 'value': 'atlantic'},
                {'label': 'Taiwan Standard', 'value': 'taiwan'}
            ],
            value='atlantic',
            style={'width': '200px'}
        )
    ], style={'margin': '10px'}),
    
], style={'font-family': 'Arial, sans-serif'})

@app.callback(
    Output('year-dropdown', 'options'),
    Input('typhoon-tracks-graph', 'figure')
)
def initialize_year_dropdown(_):
    try:
        years = typhoon_data['ISO_TIME'].dt.year.unique()
        years = years[~np.isnan(years)]
        years = sorted(years)
        
        options = [{'label': str(int(year)), 'value': int(year)} for year in years]
        print(f"Generated options: {options[:5]}...")
        return options
    except Exception as e:
        print(f"Error in initialize_year_dropdown: {str(e)}")
        return [{'label': 'Error', 'value': 'error'}]

@app.callback(
    [Output('typhoon-dropdown', 'options'),
     Output('typhoon-dropdown', 'value')],
    [Input('year-dropdown', 'value')]
)
def update_typhoon_dropdown(selected_year):
    if not selected_year:
        raise PreventUpdate
    
    selected_year = int(selected_year)
    
    season = ibtracs.get_season(selected_year)
    storm_summary = season.summary()
    
    typhoon_options = []
    for i in range(storm_summary['season_storms']):
        storm_id = storm_summary['id'][i]
        storm_name = storm_summary['name'][i]
        typhoon_options.append({'label': f"{storm_name} ({storm_id})", 'value': storm_id})
    
    selected_typhoon = typhoon_options[0]['value'] if typhoon_options else None
    return typhoon_options, selected_typhoon

@app.callback(
    Output('typhoon-path-animation', 'figure'),
    [Input('year-dropdown', 'value'),
     Input('typhoon-dropdown', 'value'),
     Input('classification-standard', 'value')]
)
def update_typhoon_path(selected_year, selected_sid, standard):
    if not selected_year or not selected_sid:
        raise PreventUpdate

    storm = ibtracs.get_storm(selected_sid)
    return create_typhoon_path_figure(storm, selected_year, standard)

def create_typhoon_path_figure(storm, selected_year, standard='atlantic'):
    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lon=storm.lon,
            lat=storm.lat,
            mode='lines',
            line=dict(width=2, color='gray'),
            name='Path',
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lon=[storm.lon[0]],
            lat=[storm.lat[0]],
            mode='markers',
            marker=dict(size=10, color='green', symbol='star'),
            name='Starting Point',
            text=storm.time[0].strftime('%Y-%m-%d %H:%M'),
            hoverinfo='text+name',
        )
    )

    frames = []
    for i in range(len(storm.time)):
        category, color = categorize_typhoon_by_standard(storm.vmax[i], standard)
        
        r34_ne = storm.dict['USA_R34_NE'][i] if 'USA_R34_NE' in storm.dict else None
        r34_se = storm.dict['USA_R34_SE'][i] if 'USA_R34_SE' in storm.dict else None
        r34_sw = storm.dict['USA_R34_SW'][i] if 'USA_R34_SW' in storm.dict else None
        r34_nw = storm.dict['USA_R34_NW'][i] if 'USA_R34_NW' in storm.dict else None
        rmw = storm.dict['USA_RMW'][i] if 'USA_RMW' in storm.dict else None
        eye_diameter = storm.dict['USA_EYE'][i] if 'USA_EYE' in storm.dict else None

        radius_info = f"R34: NE={r34_ne}, SE={r34_se}, SW={r34_sw}, NW={r34_nw}<br>"
        radius_info += f"RMW: {rmw}<br>"
        radius_info += f"Eye Diameter: {eye_diameter}"
        
        frame_data = [
            go.Scattergeo(
                lon=storm.lon[:i+1],
                lat=storm.lat[:i+1],
                mode='lines',
                line=dict(width=2, color='blue'),
                name='Path Traveled',
                showlegend=False,
            ),
            go.Scattergeo(
                lon=[storm.lon[i]],
                lat=[storm.lat[i]],
                mode='markers+text',
                marker=dict(size=10, color=color, symbol='star'),
                text=category,
                textposition="top center",
                textfont=dict(size=12, color=color),
                name='Current Location',
                hovertext=f"{storm.time[i].strftime('%Y-%m-%d %H:%M')}<br>"
                          f"Category: {category}<br>"
                          f"Wind Speed: {storm.vmax[i]:.1f} m/s<br>"
                          f"{radius_info}",
                hoverinfo='text',
            ),
        ]
        frames.append(go.Frame(data=frame_data, name=f"frame{i}"))

    fig.frames = frames

    fig.update_layout(
        title=f"{selected_year} Year {storm.name} Typhoon Path",
        showlegend=False,
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            coastlinecolor='rgb(100, 100, 100)',
            showocean=True,
            oceancolor='rgb(230, 250, 255)',
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Time: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 100, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f"frame{k}"],
                             {"frame": {"duration": 100, "redraw": True},
                              "mode": "immediate",
                              "transition": {"duration": 0}}
                            ],
                    "label": storm.time[k].strftime('%Y-%m-%d %H:%M'),
                    "method": "animate"
                }
                for k in range(len(storm.time))
            ]
        }]
    )

    return fig
    
@app.callback(
    [Output('typhoon-routes-graph', 'figure'),
     Output('cluster-equation-results', 'children')],
    [Input('analyze-button', 'n_clicks'),
     Input('show-clusters-button', 'n_clicks'),
     Input('show-routes-button', 'n_clicks'),
     Input('fourier-series-button', 'n_clicks')],
    [State('start-year', 'value'),
     State('start-month', 'value'),
     State('end-year', 'value'),
     State('end-month', 'value'),
     State('n-clusters', 'value'),
     State('enso-dropdown', 'value')]
)

def update_route_clusters(analyze_clicks, show_clusters_clicks, show_routes_clicks,
                          fourier_clicks, start_year, start_month, end_year, end_month,
                          n_clusters, enso_value):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    
    filtered_oni_df = oni_df[(oni_df.index >= start_date) & (oni_df.index <= end_date)]

    fig_routes = go.Figure()

    clusters = np.array([])  # Initialize as empty NumPy array
    cluster_equations = []
    
    # Clustering analysis
    west_pacific_storms = []
    for year in range(start_year, end_year + 1):
        season = ibtracs.get_season(year)
        for storm_id in season.summary()['id']:
            storm = get_storm_data(storm_id)
            storm_date = storm.time[0]
            storm_oni = oni_df.loc[storm_date.strftime('%Y-%b')]['ONI']
            if isinstance(storm_oni, pd.Series):
                storm_oni = storm_oni.iloc[0]
            storm_phase = classify_enso_phases(storm_oni)
            
            if enso_value == 'all' or \
               (enso_value == 'el_nino' and storm_phase == 'El Nino') or \
               (enso_value == 'la_nina' and storm_phase == 'La Nina') or \
               (enso_value == 'neutral' and storm_phase == 'Neutral'):
                lons, lats = filter_west_pacific_coordinates(np.array(storm.lon), np.array(storm.lat))
                if len(lons) > 1:  # Ensure the storm has a valid path in West Pacific
                    west_pacific_storms.append((lons, lats))

    max_length = max(len(storm[0]) for storm in west_pacific_storms)
    standardized_routes = []
    
    for lons, lats in west_pacific_storms:
        if len(lons) < 2:  # Skip if not enough points
            continue
        t = np.linspace(0, 1, len(lons))
        t_new = np.linspace(0, 1, max_length)
        lon_interp = interp1d(t, lons, kind='linear')(t_new)
        lat_interp = interp1d(t, lats, kind='linear')(t_new)
        route_vector = np.column_stack((lon_interp, lat_interp)).flatten()
        standardized_routes.append(route_vector)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(standardized_routes)

    # Count the number of typhoons in each cluster
    cluster_counts = np.bincount(clusters)

    for lons, lats in west_pacific_storms:
        fig_routes.add_trace(go.Scattergeo(
            lon=lons, lat=lats,
            mode='lines',
            line=dict(width=1, color='lightgray'),
            showlegend=False,
            hoverinfo='none',
            visible=(button_id == 'show-routes-button')
        ))

    equations_output = []
    for i in range(n_clusters):
        cluster_center = kmeans.cluster_centers_[i].reshape(-1, 2)
        cluster_equations, (lon_min, lon_max) = generate_cluster_equations(cluster_center)
        
        #equations_output.append(html.H4(f"Cluster {i+1} (Typhoons: {cluster_counts[i]})"))
        equations_output.append(html.H4([
            f"Cluster {i+1} (Typhoons: ",
                html.Span(f"{cluster_counts[i]}", style={'color': 'blue'}),
                    ")"
                    ]))
        for name, eq in cluster_equations:
            equations_output.append(html.P(f"{name}: {eq}"))
        
        equations_output.append(html.P("To use in GeoGebra:"))
        equations_output.append(html.P(f"1. Set x-axis from 0 to {2*np.pi:.4f}"))
        equations_output.append(html.P(f"2. Use the equation as is"))
        equations_output.append(html.P(f"3. To convert x back to longitude: lon = {lon_min:.4f} + x * {(lon_max - lon_min) / (2*np.pi):.4f}"))
        equations_output.append(html.Hr())
        
        fig_routes.add_trace(go.Scattergeo(
            lon=cluster_center[:, 0],
            lat=cluster_center[:, 1],
            mode='lines',
            name=f'Cluster {i+1} (n={cluster_counts[i]})',
            line=dict(width=3),
            visible=(button_id == 'show-clusters-button')
        ))

    enso_phase_text = {
        'all': 'All Years',
        'el_nino': 'El Niño Years',
        'la_nina': 'La Niña Years',
        'neutral': 'Neutral Years'
    }
    fig_routes.update_layout(
        title=f'Typhoon Routes Clustering in West Pacific ({start_year}-{end_year}) - {enso_phase_text[enso_value]}',
        geo=dict(
            projection_type='mercator',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            coastlinecolor='rgb(100, 100, 100)',
            showocean=True,
            oceancolor='rgb(230, 250, 255)',
            lataxis={'range': [0, 40]},
            lonaxis={'range': [100, 180]},
            center={'lat': 20, 'lon': 140},
        ),
        legend_title='Clusters'
    )
    
    return fig_routes, html.Div(equations_output)

@app.callback(
    [Output('typhoon-tracks-graph', 'figure'),
     Output('all-years-regression-graph', 'figure'),
     Output('regression-graphs', 'children'),
     Output('slopes', 'children'),
     Output('wind-oni-scatter-plot', 'figure'),
     Output('pressure-oni-scatter', 'figure'),
     Output('correlation-coefficient', 'children'),
     Output('max-wind-speed', 'children'),
     Output('min-pressure', 'children'),
     Output('wind-oni-correlation', 'children'),
     Output('pressure-oni-correlation', 'children'),
     Output('typhoon-count-analysis', 'children'),
     Output('concentrated-months-analysis', 'children')],
    [Input('analyze-button', 'n_clicks'),
     Input('find-typhoon-button', 'n_clicks')],
    [State('start-year', 'value'),
     State('start-month', 'value'),
     State('end-year', 'value'),
     State('end-month', 'value'),
     State('enso-dropdown', 'value'),
     State('typhoon-search', 'value')]
)

def update_graphs(analyze_clicks, find_typhoon_clicks,
                  start_year, start_month, end_year, end_month,
                  enso_value, typhoon_search):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
  
    filtered_oni_df = oni_df[(oni_df.index >= start_date) & (oni_df.index <= end_date)]


    regression_data = {'El Nino': {'longitudes': [], 'oni_values': [], 'names': []},
                       'La Nina': {'longitudes': [], 'oni_values': [], 'names': []},
                       'Neutral': {'longitudes': [], 'oni_values': [], 'names': []},
                       'All': {'longitudes': [], 'oni_values': [], 'names': []}}

    fig_tracks = go.Figure()

    def process_storm(year, storm_id):
        storm = get_storm_data(storm_id)
        storm_dates = storm.time
        if any(start_date <= date <= end_date for date in storm_dates):
            storm_oni = filtered_oni_df.loc[storm_dates[0].strftime('%Y-%b')]['ONI']
            if isinstance(storm_oni, pd.Series):
                storm_oni = storm_oni.iloc[0]
            phase = classify_enso_phases(storm_oni)
            
            regression_data[phase]['longitudes'].append(storm.lon[0])
            regression_data[phase]['oni_values'].append(storm_oni)
            regression_data[phase]['names'].append(f'{storm.name} ({year})')
            regression_data['All']['longitudes'].append(storm.lon[0])
            regression_data['All']['oni_values'].append(storm_oni)
            regression_data['All']['names'].append(f'{storm.name} ({year})')
            
            if (enso_value == 'all' or 
                (enso_value == 'el_nino' and phase == 'El Nino') or
                (enso_value == 'la_nina' and phase == 'La Nina') or
                (enso_value == 'neutral' and phase == 'Neutral')):
                color = {'El Nino': 'red', 'La Nina': 'blue', 'Neutral': 'green'}[phase]
                return go.Scattergeo(
                    lon=storm.lon,
                    lat=storm.lat,
                    mode='lines',
                    name=storm.name,
                    text=f'{storm.name} ({year})',
                    hoverinfo='text',
                    line=dict(width=2, color=color)
                )
        return None

    with ThreadPoolExecutor() as executor:
        futures = []
        for year in range(start_year, end_year + 1):
            season = ibtracs.get_season(year)
            for storm_id in season.summary()['id']:
                futures.append(executor.submit(process_storm, year, storm_id))
        
        for future in futures:
            result = future.result()
            if result:
                fig_tracks.add_trace(result)

    fig_tracks.update_layout(
        title=f'Typhoon Tracks from {start_year}-{start_month} to {end_year}-{end_month}',
        geo=dict(
            projection_type='natural earth',
            showland=True,
        )
    )

    regression_figs = []
    slopes = []
    all_years_fig = go.Figure()  # Initialize with an empty figure

    for phase in ['El Nino', 'La Nina', 'Neutral', 'All']:
        df = pd.DataFrame({
            'Longitude': regression_data[phase]['longitudes'],
            'ONI': regression_data[phase]['oni_values'],
            'Name': regression_data[phase]['names']
        })
        
        if not df.empty and len(df) > 1:  # Ensure there's enough data for regression
            try:
                fig = px.scatter(df, x='Longitude', y='ONI', hover_data=['Name'],
                                 labels={'Longitude': 'Longitude of Typhoon Generation', 'ONI': 'ONI Value'},
                                 title=f'Typhoon Generation Location vs. ONI ({phase})')
                
                X = np.array(df['Longitude']).reshape(-1, 1)
                y = df['ONI']
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                slope = model.coef_[0]
                intercept = model.intercept_
                fraction_slope = Fraction(slope).limit_denominator()
                equation = f'ONI = {fraction_slope} * Longitude + {Fraction(intercept).limit_denominator()}'
                
                fig.add_trace(go.Scatter(x=df['Longitude'], y=y_pred, mode='lines', name='Regression Line'))
                fig.add_annotation(x=df['Longitude'].mean(), y=y_pred.mean(),
                                   text=equation, showarrow=False, yshift=10)
                
                if phase == 'All':
                    all_years_fig = fig
                else:
                    regression_figs.append(dcc.Graph(figure=fig))
                
                correlation_coef = np.corrcoef(df['Longitude'], df['ONI'])[0, 1]
                slopes.append(html.P(f'{phase} Regression Slope: {slope:.4f}, Correlation Coefficient: {correlation_coef:.4f}'))
            except Exception as e:
                print(f"Error in regression analysis for {phase}: {str(e)}")
                if phase != 'All':
                    regression_figs.append(html.Div(f"Error in analysis for {phase}"))
                slopes.append(html.P(f'{phase} Regression: Error in analysis'))
        else:
            if phase != 'All':
                regression_figs.append(html.Div(f"Insufficient data for {phase}"))
            slopes.append(html.P(f'{phase} Regression: Insufficient data'))

    if all_years_fig.data == ():
        all_years_fig = go.Figure()
        all_years_fig.add_annotation(text="No data available for regression analysis",
                                     xref="paper", yref="paper",
                                     x=0.5, y=0.5, showarrow=False)

    if button_id == 'find-typhoon-button' and typhoon_search:
        for trace in fig_tracks.data:
            if typhoon_search.lower() in trace.name.lower():
                trace.line.width = 5
                trace.line.color = 'yellow'

    filtered_data = merged_data[
          (merged_data['Year'] >= start_year) & 
          (merged_data['Year'] <= end_year) & 
          (merged_data['Month'].astype(int) >= start_month) & 
          (merged_data['Month'].astype(int) <= end_month)
    ]
    
    wind_oni_scatter = px.scatter(filtered_data, x='ONI', y='USA_WIND', color='Category',
                                  hover_data=['NAME', 'Year','Category'],
                                  title='Wind Speed vs ONI',
                                  labels={'ONI': 'ONI Value', 'USA_WIND': 'Maximum Wind Speed (knots)'},
                                  color_discrete_map=color_map)
    wind_oni_scatter.update_traces(hovertemplate='<b>%{customdata[0]} (%{customdata[1]})</b><br>Category: %{customdata[2]}<br>ONI: %{x}<br>Wind Speed: %{y} knots')
    
    pressure_oni_scatter = px.scatter(filtered_data, x='ONI', y='USA_PRES',color='Category',
                                      hover_data=['NAME', 'Year','Category'], 
                                      title='Pressure vs ONI',
                                      labels={'ONI': 'ONI Value', 'USA_PRES': 'Minimum Pressure (hPa)'},
                                      color_discrete_map=color_map)
    pressure_oni_scatter.update_traces(hovertemplate='<b>%{customdata[0]} (%{customdata[1]})</b><br>Category: %{customdata[2]}<br>ONI: %{x}<br>Pressure: %{y} hPa')
    
    if typhoon_search:
        for fig in [wind_oni_scatter, pressure_oni_scatter]:
            mask = filtered_data['NAME'].str.contains(typhoon_search, case=False, na=False)
            fig.add_trace(go.Scatter(
                x=filtered_data.loc[mask, 'ONI'],
                y=filtered_data.loc[mask, 'USA_WIND' if 'Wind' in fig.layout.title.text else 'USA_PRES'],
                mode='markers',
                marker=dict(size=10, color='red', symbol='star'),
                name=f'Matched: {typhoon_search}',
                hovertemplate='<b>%{text}</b><br>Category: %{customdata}<br>ONI: %{x}<br>Value: %{y}',
                text=filtered_data.loc[mask, 'NAME'] + ' (' + filtered_data.loc[mask, 'Year'].astype(str) + ')',
                customdata=filtered_data.loc[mask, 'Category']
            ))
    
            
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    typhoon_counts, concentrated_months = analyze_typhoon_generation(merged_data, start_date, end_date)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    count_analysis = [html.P(f"{phase}: {count} typhoons") for phase, count in typhoon_counts.items()]
    month_analysis = [html.P(f"{phase}: Most concentrated in {month_names[month-1]}") for phase, month in concentrated_months.items()]

    max_wind_speed = filtered_data['USA_WIND'].max()
    min_pressure = typhoon_data[(typhoon_data['ISO_TIME'].dt.year >= start_year) & 
                                (typhoon_data['ISO_TIME'].dt.year <= end_year)]['WMO_PRES'].min()

    correlation_text = f"Logistic Regression Results: see below"
    max_wind_speed_text = f"Maximum Wind Speed: {max_wind_speed:.2f} knots"
    min_pressure_text = f"Minimum Pressure: {min_pressure:.2f} hPa"


    return (fig_tracks, all_years_fig, regression_figs, slopes, 
            wind_oni_scatter, pressure_oni_scatter,
            correlation_text, max_wind_speed_text, min_pressure_text,
            "Wind-ONI correlation: See logistic regression results", 
            "Pressure-ONI correlation: See logistic regression results",
            count_analysis, month_analysis)

@app.callback(
    Output('logistic-regression-results', 'children'),
    [Input('wind-regression-button', 'n_clicks'),
     Input('pressure-regression-button', 'n_clicks'),
     Input('longitude-regression-button', 'n_clicks')],
    [State('start-year', 'value'),
     State('start-month', 'value'),
     State('end-year', 'value'),
     State('end-month', 'value')]
)
def update_logistic_regression(wind_clicks, pressure_clicks, longitude_clicks, 
                               start_year, start_month, end_year, end_month):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Click a button to see logistic regression results."
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    
    filtered_data = merged_data[
        (merged_data['ISO_TIME'] >= start_date) & 
        (merged_data['ISO_TIME'] <= end_date)
    ]
    
    if button_id == 'wind-regression-button':
        return calculate_wind_logistic_regression(filtered_data)
    elif button_id == 'pressure-regression-button':
        return calculate_pressure_logistic_regression(filtered_data)
    elif button_id == 'longitude-regression-button':
        return calculate_longitude_logistic_regression(filtered_data)

def calculate_wind_logistic_regression(data):
    data['severe_typhoon'] = (data['USA_WIND'] >= 64).astype(int)  # 64 knots threshold for severe typhoons
    X = sm.add_constant(data['ONI'])
    y = data['severe_typhoon']
    model = sm.Logit(y, X).fit()
    
    beta_1 = model.params['ONI']
    exp_beta_1 = np.exp(beta_1)
    p_value = model.pvalues['ONI']
    
    el_nino_data = data[data['ONI'] >= 0.5]
    la_nina_data = data[data['ONI'] <= -0.5]
    neutral_data = data[(data['ONI'] > -0.5) & (data['ONI'] < 0.5)]
    
    el_nino_severe = el_nino_data['severe_typhoon'].mean()
    la_nina_severe = la_nina_data['severe_typhoon'].mean()
    neutral_severe = neutral_data['severe_typhoon'].mean()
    
    return html.Div([
        html.H3("Wind Speed Logistic Regression Results"),
        html.P(f"β1 (ONI coefficient): {beta_1:.4f}"),
        html.P(f"exp(β1) (Odds Ratio): {exp_beta_1:.4f}"),
        html.P(f"P-value: {p_value:.4f}"),
        html.P("Interpretation:"),
        html.Ul([
            html.Li(f"For each unit increase in ONI, the odds of a severe typhoon are "
                    f"{'increased' if exp_beta_1 > 1 else 'decreased'} by a factor of {exp_beta_1:.2f}."),
            html.Li(f"This effect is {'statistically significant' if p_value < 0.05 else 'not statistically significant'} "
                    f"at the 0.05 level.")
        ]),
        html.P("Proportion of severe typhoons:"),
        html.Ul([
            html.Li(f"El Niño conditions: {el_nino_severe:.2%}"),
            html.Li(f"La Niña conditions: {la_nina_severe:.2%}"),
            html.Li(f"Neutral conditions: {neutral_severe:.2%}")
        ])
    ])

def calculate_pressure_logistic_regression(data):
    data['intense_typhoon'] = (data['USA_PRES'] <= 950).astype(int)  # 950 hPa threshold for intense typhoons
    X = sm.add_constant(data['ONI'])
    y = data['intense_typhoon']
    model = sm.Logit(y, X).fit()
    
    beta_1 = model.params['ONI']
    exp_beta_1 = np.exp(beta_1)
    p_value = model.pvalues['ONI']
    
    el_nino_data = data[data['ONI'] >= 0.5]
    la_nina_data = data[data['ONI'] <= -0.5]
    neutral_data = data[(data['ONI'] > -0.5) & (data['ONI'] < 0.5)]
    
    el_nino_intense = el_nino_data['intense_typhoon'].mean()
    la_nina_intense = la_nina_data['intense_typhoon'].mean()
    neutral_intense = neutral_data['intense_typhoon'].mean()
    
    return html.Div([
        html.H3("Pressure Logistic Regression Results"),
        html.P(f"β1 (ONI coefficient): {beta_1:.4f}"),
        html.P(f"exp(β1) (Odds Ratio): {exp_beta_1:.4f}"),
        html.P(f"P-value: {p_value:.4f}"),
        html.P("Interpretation:"),
        html.Ul([
            html.Li(f"For each unit increase in ONI, the odds of an intense typhoon (pressure <= 950 hPa) are "
                    f"{'increased' if exp_beta_1 > 1 else 'decreased'} by a factor of {exp_beta_1:.2f}."),
            html.Li(f"This effect is {'statistically significant' if p_value < 0.05 else 'not statistically significant'} "
                    f"at the 0.05 level.")
        ]),
        html.P("Proportion of intense typhoons:"),
        html.Ul([
            html.Li(f"El Niño conditions: {el_nino_intense:.2%}"),
            html.Li(f"La Niña conditions: {la_nina_intense:.2%}"),
            html.Li(f"Neutral conditions: {neutral_intense:.2%}")
        ])
    ])

def calculate_longitude_logistic_regression(data):
    # Use only the data points where longitude is available
    data = data.dropna(subset=['LON'])
    
    if len(data) == 0:
        return html.Div("Insufficient data for longitude analysis")
    
    data['western_typhoon'] = (data['LON'] <= 140).astype(int)  # 140°E as threshold for western typhoons
    X = sm.add_constant(data['ONI'])
    y = data['western_typhoon']
    model = sm.Logit(y, X).fit()
    
    beta_1 = model.params['ONI']
    exp_beta_1 = np.exp(beta_1)
    p_value = model.pvalues['ONI']
    
    el_nino_data = data[data['ONI'] >= 0.5]
    la_nina_data = data[data['ONI'] <= -0.5]
    neutral_data = data[(data['ONI'] > -0.5) & (data['ONI'] < 0.5)]
    
    el_nino_western = el_nino_data['western_typhoon'].mean()
    la_nina_western = la_nina_data['western_typhoon'].mean()
    neutral_western = neutral_data['western_typhoon'].mean()
    
    return html.Div([
        html.H3("Longitude Logistic Regression Results"),
        html.P(f"β1 (ONI coefficient): {beta_1:.4f}"),
        html.P(f"exp(β1) (Odds Ratio): {exp_beta_1:.4f}"),
        html.P(f"P-value: {p_value:.4f}"),
        html.P("Interpretation:"),
        html.Ul([
            html.Li(f"For each unit increase in ONI, the odds of a typhoon forming west of 140°E are "
                    f"{'increased' if exp_beta_1 > 1 else 'decreased'} by a factor of {exp_beta_1:.2f}."),
            html.Li(f"This effect is {'statistically significant' if p_value < 0.05 else 'not statistically significant'} "
                    f"at the 0.05 level.")
        ]),
        html.P("Proportion of typhoons forming west of 140°E:"),
        html.Ul([
            html.Li(f"El Niño conditions: {el_nino_western:.2%}"),
            html.Li(f"La Niña conditions: {la_nina_western:.2%}"),
            html.Li(f"Neutral conditions: {neutral_western:.2%}")
        ])
    ])

def categorize_typhoon_by_standard(wind_speed, standard='atlantic'):
    """
    Categorize typhoon based on wind speed and chosen standard
    wind_speed is in knots
    """
    if standard == 'taiwan':
        # Convert knots to m/s for Taiwan standard
        wind_speed_ms = wind_speed * 0.514444
        
        if wind_speed_ms >= 51.0:
            return 'Strong Typhoon', taiwan_standard['Strong Typhoon']['color']
        elif wind_speed_ms >= 33.7:
            return 'Medium Typhoon', taiwan_standard['Medium Typhoon']['color']
        elif wind_speed_ms >= 17.2:
            return 'Mild Typhoon', taiwan_standard['Mild Typhoon']['color']
        else:
            return 'Tropical Depression', taiwan_standard['Tropical Depression']['color']
    else:
        # Atlantic standard uses knots
        if wind_speed >= 137:
            return 'C5 Super Typhoon', atlantic_standard['C5 Super Typhoon']['color']
        elif wind_speed >= 113:
            return 'C4 Very Strong Typhoon', atlantic_standard['C4 Very Strong Typhoon']['color']
        elif wind_speed >= 96:
            return 'C3 Strong Typhoon', atlantic_standard['C3 Strong Typhoon']['color']
        elif wind_speed >= 83:
            return 'C2 Typhoon', atlantic_standard['C2 Typhoon']['color']
        elif wind_speed >= 64:
            return 'C1 Typhoon', atlantic_standard['C1 Typhoon']['color']
        elif wind_speed >= 34:
            return 'Tropical Storm', atlantic_standard['Tropical Storm']['color']
        else:
            return 'Tropical Depression', atlantic_standard['Tropical Depression']['color']

if __name__ == "__main__":
    print(f"Using data path: {DATA_PATH}")
    # Update ONI data before starting the application
    update_oni_data()
    oni_df = fetch_oni_data_from_csv(ONI_DATA_PATH)
    ibtracs = load_ibtracs_data()
    convert_typhoondata(LOCAL_iBtrace_PATH, TYPHOON_DATA_PATH)
    oni_data, typhoon_data = load_data(ONI_DATA_PATH, TYPHOON_DATA_PATH)
    oni_long = process_oni_data_with_cache(oni_data)
    typhoon_max = process_typhoon_data_with_cache(typhoon_data)
    merged_data = merge_data(oni_long, typhoon_max)
    data = preprocess_data(oni_data, typhoon_data)
    max_wind_speed, min_pressure = calculate_max_wind_min_pressure(typhoon_data)
    
    
    # Schedule IBTrACS data update daily
    schedule.every().day.at("01:00").do(update_ibtracs_data)
    
    # Schedule ONI data check daily, but only update on specified dates
    schedule.every().day.at("00:00").do(lambda: update_oni_data() if should_update_oni() else None)
    
    # Run the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_schedule)
    scheduler_thread.start()
    
    
    app.run_server(debug=True, host='127.0.0.1', port=8050)
