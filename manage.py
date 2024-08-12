import os
import sys
import git
import subprocess
import requests
import venv
import glob
import cmd

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

class TyphoonAnalysisShell(cmd.Cmd):
    intro = "Welcome to the Typhoon Analysis Dashboard Manager. Type help or ? to list commands.\n"
    prompt = "(typhoon) "

    def do_update(self, arg):
        """Update all scripts and requirements.txt from GitHub"""
        if update_from_github():
            print("All scripts and requirements.txt updated. Please restart the manager to use the latest version.")

    def do_install(self, arg):
        """Update installed packages"""
        update_requirements()

    def do_run(self, arg):
        """Run Typhoon Analysis Dashboard"""
        run_script()

    def do_exit(self, arg):
        """Exit the Typhoon Analysis Dashboard Manager"""
        print("Exiting...")
        return True

    def do_EOF(self, arg):
        """Exit on EOF"""
        print("Exiting...")
        return True

def main():
    create_venv()
    TyphoonAnalysisShell().cmdloop()

if __name__ == "__main__":
    main()
