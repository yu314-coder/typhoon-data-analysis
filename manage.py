import os
import sys
import git
import subprocess
import requests

def update_from_github():
    try:
        repo = git.Repo(os.path.dirname(os.path.abspath(__file__)))
        origin = repo.remotes.origin
        origin.pull()
        print("Successfully updated from GitHub.")
        return True
    except Exception as e:
        print(f"Failed to update from GitHub: {str(e)}")
        return False

def download_requirements():
    try:
        # Replace 'your_username' and 'your_repo' with your GitHub username and repository name
        url = 'https://raw.githubusercontent.com/yu314-coder/typhoon-data-analysis-/main/requirements.txt'
        response = requests.get(url)
        if response.status_code == 200:
            with open('requirements.txt', 'wb') as f:
                f.write(response.content)
            print("Successfully downloaded latest requirements.txt")
            return True
        else:
            print(f"Failed to download requirements.txt. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading requirements.txt: {str(e)}")
        return False

def update_requirements():
    if download_requirements():
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("Successfully updated requirements.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to update requirements: {str(e)}")
            return False
    return False

def run_script():
    try:
        subprocess.check_call([sys.executable, "typhoon_analysis.py"])
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {str(e)}")

def main_menu():
    while True:
        print("\n--- Typhoon Analysis Dashboard Manager ---")
        print("1. Update script from GitHub")
        print("2. Update requirements")
        print("3. Run Typhoon Analysis Dashboard")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            update_from_github()
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
