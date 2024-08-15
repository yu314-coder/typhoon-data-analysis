import sys
import subprocess
import importlib.util
import os
import venv
import glob
import time
import socket
from importlib import reload

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install gitpython
if importlib.util.find_spec("git") is None:
    print("Installing gitpython...")
    install_package("gitpython")
else:
    print("gitpython is already installed.")

# Check and install requests
if importlib.util.find_spec("requests") is None:
    print("Installing requests...")
    install_package("requests")
else:
    print("requests is already installed.")

# Reload sys module to ensure newly installed packages are recognized
reload(sys)

# Now import the installed packages
import git
import requests
from requests.exceptions import RequestException

# Rest of the code remains the same
VENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')

def create_venv():
    if not os.path.exists(VENV_PATH):
        print("Creating virtual environment...")
        venv.create(VENV_PATH, with_pip=True)
        print("Virtual environment created.")
    else:
        print("Virtual environment already exists.")

def get_venv_python():
    if sys.platform == "win32":
        return os.path.join(VENV_PATH, 'Scripts', 'python.exe')
    return os.path.join(VENV_PATH, 'bin', 'python')

def run_in_venv(command):
    venv_python = get_venv_python()
    return subprocess.Popen([venv_python] + command)

# ... (rest of the functions remain the same)

def main_menu():
    create_venv()
    while True:
        print("\n--- Typhoon Analysis Dashboard Manager ---")
        print("1. Update all scripts and requirements.txt from GitHub")
        print("2. Update installed packages")
        print("3. Run Typhoon Analysis Dashboard")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            if update_from_github():
                print("All scripts and requirements.txt updated. Please restart the manager to use the latest version.")
                sys.exit(0)
        elif choice == '2':
            update_requirements()
        elif choice == '3':
            run_script()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
