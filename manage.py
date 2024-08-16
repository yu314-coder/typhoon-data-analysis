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

# Path for the virtual environment
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

def update_from_github():
    try:
        repo = git.Repo(os.path.dirname(os.path.abspath(__file__)))
        origin = repo.remotes.origin

        # Check for local changes
        if repo.is_dirty(untracked_files=True):
            print("Local changes detected. Please choose an option:")
            print("1. Stash local changes and pull")
            print("2. Discard local changes and pull")
            print("3. Cancel update")
            
            choice = input("Enter your choice (1-3): ")
            
            if choice == '1':
                print("Stashing local changes...")
                repo.git.stash()
            elif choice == '2':
                print("Discarding local changes...")
                repo.git.reset('--hard')
                repo.git.clean('-fd')
            elif choice == '3':
                print("Update cancelled.")
                return False
            else:
                print("Invalid choice. Update cancelled.")
                return False

        # Pull changes from remote
        origin.pull()
        print("Successfully updated from GitHub.")
        
        # Update all Python files
        for py_file in glob.glob("*.py"):
            print(f"Updated {py_file}")
        
        # Update requirements.txt
        if os.path.exists('requirements.txt'):
            print("Updated requirements.txt")
        
        # Apply stashed changes if they were stashed
        if 'choice' in locals() and choice == '1':
            try:
                repo.git.stash('pop')
                print("Reapplied local changes.")
            except git.GitCommandError:
                print("Failed to reapply local changes. They remain in the stash.")
        
        return True
    except Exception as e:
        print(f"Failed to update from GitHub: {str(e)}")
        return False

def update_requirements():
    try:
        subprocess.run([get_venv_python(), "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Successfully updated requirements in virtual environment.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to update requirements: {str(e)}")
        return False

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def run_script():
    try:
        process = run_in_venv(["typhoon_analysis.py"])
        print("Typhoon Analysis Dashboard is starting...")
        
        url = 'http://127.0.0.1:8050/'
        
        # Wait for the server to be ready
        max_attempts = 200
        for attempt in range(max_attempts):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"\nDashboard is now ready at: {url}")
                    break
            except RequestException:
                if attempt < max_attempts - 1:
                    time.sleep(1)
                else:
                    print(f"\nWarning: Dashboard might not be ready. You can still try accessing it at: {url}")
        
        print("\nPress Ctrl+C to stop the server when you're done.")
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping the server...")
    except Exception as e:
        print(f"Error running script: {str(e)}")
    finally:
        if 'process' in locals():
            process.terminate()
        print("Server stopped.")

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
