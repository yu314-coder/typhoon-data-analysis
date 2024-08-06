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
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor

# Add command-line argument parsing
parser = argparse.ArgumentParser(description='Typhoon Analysis Dashboard')
parser.add_argument('--data_path', type=str, default=os.getcwd(), help='Path to the data directory')
args = parser.parse_args()

# Use the command-line argument for data path
DATA_PATH = args.data_path

ONI_DATA_PATH = os.path.join(DATA_PATH, 'oni_data.csv')
TYPHOON_DATA_PATH = os.path.join(DATA_PATH, 'processed_typhoon_data.csv')

CACHE_FILE = 'ibtracs_cache.pkl'
CACHE_EXPIRY_DAYS = 1

color_map = {
    '強烈颱風': 'rgb(255, 0, 0)',
    '中度颱風': 'rgb(255, 165, 0)',
    '輕度颱風': 'rgb(255, 255, 0)',
    '熱帶性低氣壓': 'rgb(0, 255, 255)'
}

def load_ibtracs_data():
    if os.path.exists(CACHE_FILE):
        cache_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if datetime.now() - cache_time < timedelta(days=CACHE_EXPIRY_DAYS):
            print("Loading data from cache...")
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
    
    print("Fetching new data from ibtracs...")
    ibtracs = tracks.TrackDataset(basin='west_pacific', source='ibtracs')
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(ibtracs, f)
    
    return ibtracs

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
    wind_speed_kt = wind_speed / 1.94384  # Convert m/s to kt
    if wind_speed_kt >= 51:
        return '強烈颱風'
    elif wind_speed_kt >= 32.7:
        return '中度颱風'
    elif wind_speed_kt >= 17.2:
        return '輕度颱風'
    else:
        return '熱帶性低氣壓'

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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_cost(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
    return cost

def logistic_regression_gradient(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (1/m) * X.T @ (h - y)
    return grad

def calculate_correlations(merged_data):
    data = merged_data.dropna(subset=['USA_WIND', 'USA_PRES', 'ONI'])
    
    def logistic_regression_score(X, y):
        X = np.column_stack((np.ones(X.shape[0]), X))
        y_binary = (y > np.median(y)).astype(int)
        
        initial_theta = np.zeros(X.shape[1])
        res = minimize(logistic_regression_cost, initial_theta, args=(X, y_binary),
                       method='BFGS', jac=logistic_regression_gradient)
        
        theta = res.x
        predictions = sigmoid(X @ theta) >= 0.5
        accuracy = np.mean(predictions == y_binary)
        
        return accuracy, theta[1]
    
    wind_score, wind_coef = logistic_regression_score(data['USA_WIND'].values.reshape(-1, 1), data['ONI'].values)
    pressure_score, pressure_coef = logistic_regression_score(data['USA_PRES'].values.reshape(-1, 1), data['ONI'].values)
    
    return wind_score, wind_coef, pressure_score, pressure_coef

@cachetools.cached(cache={})
def fetch_oni_data_from_csv(file_path):
    df = pd.read_csv(file_path, sep=',', header=0, na_values='-99.90')
    df.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df = df.melt(id_vars=['Year'], var_name='Month', value_name='ONI')
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + df['Month'], format='%Y%b')
    df = df.set_index('Date')
    return df

def classify_enso_phases(oni_value):
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

ibtracs = load_ibtracs_data()
oni_data, typhoon_data = load_data(ONI_DATA_PATH, TYPHOON_DATA_PATH)
oni_long = process_oni_data_with_cache(oni_data)
typhoon_max = process_typhoon_data_with_cache(typhoon_data)
merged_data = merge_data(oni_long, typhoon_max)
data = preprocess_data(oni_data, typhoon_data)
max_wind_speed, min_pressure = calculate_max_wind_min_pressure(typhoon_data)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Typhoon Analysis Dashboard"),
    
    html.Div([
        dcc.Input(id='start-year', type='number', placeholder='Start Year', value=2000, min=1900, max=2024, step=1),
        dcc.Input(id='start-month', type='number', placeholder='Start Month', value=1, min=1, max=12, step=1),
        dcc.Input(id='end-year', type='number', placeholder='End Year', value=2024, min=1900, max=2024, step=1),
        dcc.Input(id='end-month', type='number', placeholder='End Month', value=12, min=1, max=12, step=1),
        dcc.Input(id='n-clusters', type='number', placeholder='Number of Clusters', value=5, min=1, max=20, step=1),
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
    dcc.Graph(id='typhoon-routes-graph'),
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
    
    html.H2("Typhoon Path Analysis"),
    html.Div([
        dcc.Dropdown(id='year-dropdown', style={'width': '200px'}),
        dcc.Dropdown(id='typhoon-dropdown', style={'width': '300px'})
    ],style={'display': 'flex', 'gap': '10px'}),
    
    dcc.Graph(id='typhoon-path-animation'),
    
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
     Input('typhoon-dropdown', 'value')]
)
def update_typhoon_path(selected_year, selected_sid):
    if not selected_year or not selected_sid:
        raise PreventUpdate

    storm = ibtracs.get_storm(selected_sid)
    return create_typhoon_path_figure(storm, selected_year)

def create_typhoon_path_figure(storm, selected_year):
    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lon=storm.lon,
            lat=storm.lat,
            mode='lines',
            line=dict(width=2, color='gray'),
            name='路徑',
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lon=[storm.lon[0]],
            lat=[storm.lat[0]],
            mode='markers',
            marker=dict(size=10, color='green', symbol='star'),
            name='起始點',
            text=storm.time[0].strftime('%Y-%m-%d %H:%M'),
            hoverinfo='text+name',
        )
    )

    frames = []
    for i in range(len(storm.time)):
        category = categorize_typhoon(storm.vmax[i])
        color = color_map.get(category, 'gray')
        
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
                name='已經過路徑',
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
                name='當前位置',
                hovertext=f"{storm.time[i].strftime('%Y-%m-%d %H:%M')}<br>"
                          f"分類: {category}<br>"
                          f"風速: {storm.vmax[i]:.1f} m/s<br>"
                          f"{radius_info}",
                hoverinfo='text',
            ),
        ]
        frames.append(go.Frame(data=frame_data, name=f"frame{i}"))

    fig.frames = frames

    fig.update_layout(
        title=f"{selected_year} 年 {storm.name} 颱風路徑",
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
                    "label": "播放",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "暫停",
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
                "prefix": "時間: ",
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
    [Output('typhoon-tracks-graph', 'figure'),
     Output('typhoon-routes-graph', 'figure'),
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
     Output('concentrated-months-analysis', 'children'),
     Output('cluster-info', 'children')],
    [Input('analyze-button', 'n_clicks'),
     Input('find-typhoon-button', 'n_clicks'),
     Input('year-dropdown', 'value'),
     Input('typhoon-dropdown', 'value')],
    [State('start-year', 'value'),
     State('start-month', 'value'),
     State('end-year', 'value'),
     State('end-month', 'value'),
     State('n-clusters', 'value'),
     State('enso-dropdown', 'value'),
     State('typhoon-search', 'value')]
)
def update_graphs(analyze_clicks, find_typhoon_clicks, selected_year, selected_typhoon,
                  start_year, start_month, end_year, end_month, n_clusters, enso_value, 
                  typhoon_search):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
  
    oni_df = fetch_oni_data_from_csv(ONI_DATA_PATH)
    oni_df = oni_df[(oni_df.index >= start_date) & (oni_df.index <= end_date)]

    regression_data = {'El Nino': {'longitudes': [], 'oni_values': [], 'names': []},
                       'La Nina': {'longitudes': [], 'oni_values': [], 'names': []},
                       'Neutral': {'longitudes': [], 'oni_values': [], 'names': []},
                       'All': {'longitudes': [], 'oni_values': [], 'names': []}}

    fig_tracks = go.Figure()

    def process_storm(year, storm_id):
        storm = get_storm_data(storm_id)
        storm_dates = storm.time
        if any(start_date <= date <= end_date for date in storm_dates):
            storm_oni = oni_df.loc[storm_dates[0].strftime('%Y-%b')]['ONI']
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
    
    wind_score, wind_coef, pressure_score, pressure_coef = calculate_correlations(filtered_data)
    
    wind_oni_relationship = f"Wind Speed - ONI Logistic Regression Score: {wind_score:.4f}, Coefficient: {wind_coef:.4f}"
    pressure_oni_relationship = f"Pressure - ONI Logistic Regression Score: {pressure_score:.4f}, Coefficient: {pressure_coef:.4f}"
    
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
    
    # Clustering analysis
    west_pacific_storms = []
    for year in range(start_year, end_year + 1):
        season = ibtracs.get_season(year)
        for storm_id in season.summary()['id']:
            storm = get_storm_data(storm_id)
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

    fig_routes = go.Figure()

    for lons, lats in west_pacific_storms:
        fig_routes.add_trace(go.Scattergeo(
            lon=lons, lat=lats,
            mode='lines',
            line=dict(width=1, color='lightgray'),
            showlegend=False,
            hoverinfo='none',
        ))

    for i in range(n_clusters):
        cluster_center = kmeans.cluster_centers_[i].reshape(-1, 2)
        fig_routes.add_trace(go.Scattergeo(
            lon=cluster_center[:, 0],
            lat=cluster_center[:, 1],
            mode='lines',
            name=f'Cluster {i+1}',
            line=dict(width=3),
        ))

    fig_routes.update_layout(
        title=f'Typhoon Routes Clustering in West Pacific ({start_year}-{end_year})',
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
            
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    typhoon_counts, concentrated_months = analyze_typhoon_generation(merged_data, start_date, end_date)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    count_analysis = [html.P(f"{phase}: {count} typhoons") for phase, count in typhoon_counts.items()]
    month_analysis = [html.P(f"{phase}: Most concentrated in {month_names[month-1]}") for phase, month in concentrated_months.items()]

    max_wind_speed = filtered_data['USA_WIND'].max()
    min_pressure = typhoon_data[(typhoon_data['ISO_TIME'].dt.year >= start_year) & 
                                (typhoon_data['ISO_TIME'].dt.year <= end_year)]['WMO_PRES'].min()

    correlation_text = f"Logistic Regression Results: {wind_oni_relationship}, {pressure_oni_relationship}"
    max_wind_speed_text = f"Maximum Wind Speed: {max_wind_speed:.2f} knots"
    min_pressure_text = f"Minimum Pressure: {min_pressure:.2f} hPa"

    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    cluster_info = [html.P(f"Cluster {i+1}: {count} typhoons") for i, count in enumerate(cluster_counts)]

    return (fig_tracks, fig_routes, all_years_fig, regression_figs, slopes, 
            wind_oni_scatter, pressure_oni_scatter,
            correlation_text, max_wind_speed_text, min_pressure_text,
            wind_oni_relationship, pressure_oni_relationship,
            count_analysis, month_analysis, cluster_info)

if __name__ == "__main__":
    print(f"Using data path: {DATA_PATH}")
    app.run_server(debug=True)
