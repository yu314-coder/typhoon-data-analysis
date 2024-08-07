Certainly! Here's the full README.md with all the changes incorporated:

```markdown
# Typhoon Analysis Dashboard

This project is a Dash-based web application for analyzing typhoon data in the West Pacific region. It provides various visualizations and analyses of typhoon tracks, wind speeds, pressures, and their relationships with the Oceanic Niño Index (ONI).

## Features

- Interactive typhoon track visualization
- Typhoon route clustering
- Wind speed and pressure analysis in relation to ONI
- ENSO phase impact analysis on typhoon generation
- Individual typhoon path animation

## Prerequisites

- Python 3.10 or higher
- Git

## Installation and Usage

Follow these steps to set up and run the Typhoon Analysis Dashboard:
```

1. Clone the repository:
   ```bash
   git clone https://github.com/yu314-coder/typhoon-data-analysis.git
   ```

2. Navigate to the project directory:
   ```bash
   cd typhoon-analysis-dashboard
   ```
3.install gitpython (if you don't have previous)
   ```bash
   pip install gitpython
   ```
4. Run the management script:
   
   For Linux:
   ```bash
   python3 manage.py
   ```
   
   For Windows:
   ```bash
   python manage.py
   ```

5. In the management script menu, select option 2 to download and install the required libraries.

6. After the installation is complete, select option 3 to run the script.

7. The script will provide a Dash link. Open this link in your web browser to access the Typhoon Analysis Dashboard.

## Updating the Project

To update the project with the latest changes:

1. Run the management script as shown in step 3 of Installation and Usage.

2. Select option 1 to update all scripts and requirements from GitHub.

3. After updating, select option 2 again to ensure all required libraries are up to date.

## Data Sources

- IBTrACS (International Best Track Archive for Climate Stewardship)
- Oceanic Niño Index (ONI) data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Troubleshooting

If you encounter any issues:

1. Ensure your Python version is 3.10 or higher:
   ```bash
   python3 --version
   ```

2. If you're having issues with specific libraries, try updating them manually:
   ```bash
   pip install --upgrade [library-name]
   ```

3. If problems persist, please open an issue on the GitHub repository with details of the error and your system configuration.
