import os
import sys
import git
import subprocess
import requests
import venv
import glob
import tkinter as tk
from tkinter import scrolledtext, messagebox

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
    return subprocess.run([venv_python] + command, check=True, capture_output=True, text=True)

def update_from_github():
    try:
        repo = git.Repo(os.path.dirname(os.path.abspath(__file__)))
        origin = repo.remotes.origin
        origin.pull()
        output = "Successfully updated from GitHub.\n\n"
        
        # Update all Python files
        for py_file in glob.glob("*.py"):
            output += f"Updated {py_file}\n"
        
        # Update requirements.txt
        if os.path.exists('requirements.txt'):
            output += "Updated requirements.txt\n"
        
        return True, output
    except Exception as e:
        return False, f"Failed to update from GitHub: {str(e)}"

def update_requirements():
    try:
        result = run_in_venv(["-m", "pip", "install", "-r", "requirements.txt"])
        return True, "Successfully updated requirements in virtual environment.\n" + result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Failed to update requirements: {str(e)}\n" + e.stdout

def run_script():
    try:
        result = run_in_venv(["typhoon_analysis.py"])
        return True, "Script executed successfully.\n" + result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error running script: {str(e)}\n" + e.stdout

class TyphoonAnalysisGUI:
    def __init__(self, master):
        self.master = master
        master.title("Typhoon Analysis Dashboard Manager")
        master.geometry("600x400")

        self.output_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=70, height=20)
        self.output_area.pack(padx=10, pady=10)

        self.input_area = tk.Entry(master, width=50)
        self.input_area.pack(pady=5)
        self.input_area.bind("<Return>", self.process_command)

        self.submit_button = tk.Button(master, text="Submit", command=self.process_command)
        self.submit_button.pack(pady=5)

        self.output_area.insert(tk.END, "Welcome to the Typhoon Analysis Dashboard Manager.\n")
        self.output_area.insert(tk.END, "Available commands:\n")
        self.output_area.insert(tk.END, "  update  - Update all scripts and requirements.txt from GitHub\n")
        self.output_area.insert(tk.END, "  install - Update installed packages\n")
        self.output_area.insert(tk.END, "  run     - Run Typhoon Analysis Dashboard\n")
        self.output_area.insert(tk.END, "  exit    - Exit the manager\n\n")

    def process_command(self, event=None):
        command = self.input_area.get().strip().lower()
        self.input_area.delete(0, tk.END)

        if command == "update":
            success, output = update_from_github()
            self.output_area.insert(tk.END, output + "\n")
            if success:
                messagebox.showinfo("Update Successful", "Please restart the manager to use the latest version.")
                self.master.quit()
        elif command == "install":
            success, output = update_requirements()
            self.output_area.insert(tk.END, output + "\n")
        elif command == "run":
            success, output = run_script()
            self.output_area.insert(tk.END, output + "\n")
        elif command == "exit":
            self.master.quit()
        else:
            self.output_area.insert(tk.END, f"Unknown command: {command}\n")

        self.output_area.see(tk.END)

def main():
    create_venv()
    root = tk.Tk()
    gui = TyphoonAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
