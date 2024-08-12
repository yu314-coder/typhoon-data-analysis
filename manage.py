import os
import sys
import git
import subprocess
import venv
import glob

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
    return subprocess.run([venv_python] + command, check=True)

def update_from_github():
    try:
        repo = git.Repo(os.path.dirname(os.path.abspath(__file__)))
        origin = repo.remotes.origin
        origin.pull()
        print("Successfully updated from GitHub.")
        
        # Update all Python files
        for py_file in glob.glob("*.py"):
            print(f"Updated {py_file}")
        
        # Update requirements.txt
        if os.path.exists('requirements.txt'):
            print("Updated requirements.txt")
        
        return True
    except Exception as e:
        print(f"Failed to update from GitHub: {str(e)}")
        return False

def update_requirements():
    try:
        run_in_venv(["-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully updated requirements in virtual environment.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to update requirements: {str(e)}")
        return False

def run_script():
    try:
        run_in_venv(["typhoon_analysis.py"])
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {str(e)}")

def main_menu():
    menu = """
--- Typhoon Analysis Dashboard Manager ---
1. Update all scripts and requirements.txt from GitHub
2. Update installed packages
3. Run Typhoon Analysis Dashboard
4. Exit
    
Enter your choice (1-4): """
    return menu

def open_new_console():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "manage.py")
    
    if sys.platform == "win32":
        # For Windows
        subprocess.Popen(f'start cmd /k python "{script_path}" run_menu', shell=True)
    else:
        # For Linux and MacOS
        terminal_command = "x-terminal-emulator -e" if sys.platform.startswith("linux") else "open -a Terminal"
        subprocess.Popen(f'{terminal_command} python3 "{script_path}" run_menu', shell=True)

def run_menu():
    create_venv()
    while True:
        choice = input(main_menu())
        
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
    if len(sys.argv) > 1 and sys.argv[1] == "run_menu":
        run_menu()
    else:
        open_new_console()
