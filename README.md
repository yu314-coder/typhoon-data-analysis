# Typhoon Analysis Dashboard

This project is a Dash-based web application for analyzing typhoon data in the West Pacific region. It provides various visualizations and analyses of typhoon tracks, wind speeds, pressures, and their relationships with the Oceanic Niño Index (ONI).

## Features

- Interactive typhoon track visualization
- Typhoon route clustering
- Wind speed and pressure analysis in relation to ONI
- ENSO phase impact analysis on typhoon generation
- Individual typhoon path animation

## Installation

1. Clone this repository:
git clone https://github.com/yu314-coder/typhoon-analysis-dashboard.git
Copy
2. Navigate to the project directory:
cd typhoon-analysis-dashboard
Copy
3. Install the required packages:
pip install -r requirements.txt
Copy
## Usage

1. Ensure you have the necessary data files in the correct directory (oni_data.csv and processed_typhoon_data.csv).

2. Run the Dash app:
python typhoon_analysis.py
Copy
3. Open a web browser and go to `http://127.0.0.1:8050/` to view the dashboard.

## Data Sources

- IBTrACS (International Best Track Archive for Climate Stewardship)
- Oceanic Niño Index (ONI) data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
