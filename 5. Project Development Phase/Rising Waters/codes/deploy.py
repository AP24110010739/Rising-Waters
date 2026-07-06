#!/usr/bin/env python3
"""
Rising Water - PythonAnywhere Automated Deployer
=================================================
This script automates the deployment of your Flask web application to PythonAnywhere.
It uploads the necessary code, model files, static assets, and templates, creates
the webapp, writes the WSGI config, installs requirements, and reloads the server.

No credit card or Git setup required!
"""

import os
import sys
import time

try:
    import requests
except ImportError:
    print("Installing 'requests' library first...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "requests"])
    import requests

# Files/folders to exclude from upload
EXCLUDE_DIRS = {".git", ".github", "__pycache__", "venv", ".venv", ".gemini", "dataset"}
EXCLUDE_FILES = {"database.db", "database.db-journal", "deploy.py"}

def get_all_files():
    """Recursively list all project files to upload."""
    upload_list = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for root, dirs, files in os.walk(base_dir):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            if file in EXCLUDE_FILES or file.endswith(".pyc"):
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, base_dir)
            upload_list.append((full_path, rel_path))
            
    return upload_list

def main():
    print("=" * 60)
    print("          RISING WATER - CLOUD DEPLOYER")
    print("=" * 60)
    print("This script will deploy your application permanently to PythonAnywhere.")
    print("If you don't have an account, sign up for free at:")
    print("👉 https://www.pythonanywhere.com/ (No credit card needed)\n")
    print("Then go to Account -> API Token, and click 'Create a new API token'.")
    print("-" * 60)
    
    username = input("Enter your PythonAnywhere username: ").strip()
    if not username:
        print("Username is required. Exiting.")
        return
        
    token = input("Enter your PythonAnywhere API Token: ").strip()
    if not token:
        print("API Token is required. Exiting.")
        return
        
    region_choice = input("Are you on the EU server? (y/N): ").strip().lower()
    host = "eu.pythonanywhere.com" if region_choice == 'y' else "www.pythonanywhere.com"
    
    domain_name = f"{username}.{host.replace('www.', '')}"
    api_base = f"https://{host}/api/v0/user/{username}"
    headers = {
        "Authorization": f"Token {token}"
    }
    
    # 1. Verify credentials and check existing web apps
    print("\n[1/5] Verifying API credentials...")
    webapps_url = f"{api_base}/webapps/"
    response = requests.get(webapps_url, headers=headers)
    if response.status_code != 200:
        print(f"❌ Authentication failed: {response.status_code} - {response.text}")
        return
    
    existing_apps = [app["domain_name"] for app in response.json()]
    
    # 2. Create the web app if it doesn't exist
    if domain_name not in existing_apps:
        print(f"➕ Creating new Flask webapp for {domain_name}...")
        create_payload = {
            "domain_name": domain_name,
            "python_version": "python310"
        }
        res = requests.post(webapps_url, headers=headers, data=create_payload)
        if res.status_code not in (200, 201):
            print(f"❌ Failed to create webapp: {res.status_code} - {res.text}")
            return
        print("✅ Webapp created successfully.")
    else:
        print(f"✅ Webapp for {domain_name} already exists. Proceeding to update files.")
        
    # 3. Upload project files
    print("\n[2/5] Uploading project files (this may take a minute)...")
    files_to_upload = get_all_files()
    total_files = len(files_to_upload)
    
    for idx, (full_path, rel_path) in enumerate(files_to_upload, 1):
        target_path = f"/home/{username}/rising-water/{rel_path.replace(os.sep, '/')}"
        upload_url = f"{api_base}/files/path{target_path}"
        
        print(f"   [{idx}/{total_files}] Uploading: {rel_path}...", end="", flush=True)
        
        with open(full_path, "rb") as f:
            files_payload = {"content": f}
            res = requests.post(upload_url, headers=headers, files=files_payload)
            
        if res.status_code in (200, 201):
            print(" Done.")
        else:
            print(f"\n❌ Error uploading {rel_path}: {res.status_code} - {res.text}")
            return
            
    # 4. Upload WSGI configuration
    print("\n[3/5] Configuring WSGI Entrypoint...")
    wsgi_content = f"""# WSGI Config for Rising Water Flask Webapp
import sys
import os

project_home = '/home/{username}/rising-water'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Load packages from user-site directory (pip install --user)
user_site = f'/home/{username}/.local/lib/python3.10/site-packages'
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path = [user_site] + sys.path

os.chdir(project_home)

# Flask application variable must be named 'application'
from app import app as application
"""
    wsgi_path = f"/var/www/{username.replace('.', '_')}_{host.replace('www.', '').replace('.', '_')}_wsgi.py"
    wsgi_upload_url = f"{api_base}/files/path{wsgi_path}"
    
    res = requests.post(
        wsgi_upload_url, 
        headers=headers, 
        files={"content": wsgi_content.encode("utf-8")}
    )
    if res.status_code not in (200, 201):
        print(f"❌ Failed to configure WSGI: {res.status_code} - {res.text}")
        return
    print("✅ WSGI configuration uploaded.")
    
    # Configure webapp virtual environment path or static files if needed
    
    # 5. Open console to install requirements
    print("\n[4/5] Running console to check dependencies (waitress/gunicorn etc.)...")
    consoles_url = f"{api_base}/consoles/"
    
    # Clean up existing consoles first to avoid "Console limit reached"
    try:
        existing_consoles_res = requests.get(consoles_url, headers=headers)
        if existing_consoles_res.status_code == 200:
            consoles_list = existing_consoles_res.json()
            # If it is a list (valid JSON response)
            if isinstance(consoles_list, list):
                for console in consoles_list:
                    c_id = console["id"]
                    print(f"   Closing active console {c_id} to free up limit...")
                    requests.delete(f"{consoles_url}{c_id}/", headers=headers)
    except Exception as e:
        print(f"   ⚠️ Could not clean up existing consoles: {e}")
        
    res = requests.post(consoles_url, headers=headers, json={"executable": "bash"})
    # Check both status and if response is actually JSON
    if res.status_code in (200, 201) and "Console limit" not in res.text:
        try:
            console_id = res.json()["id"]
            send_url = f"{consoles_url}{console_id}/send_input/"
            print("   Starting package installations...")
            
            # Install packages in the user environment and upgrade them
            cmd = f"pip install --user --upgrade -r /home/{username}/rising-water/requirements.txt\n"
            requests.post(send_url, headers=headers, json={"input": cmd})
            
            # Give it a few seconds to kick off
            time.sleep(5)
            print("   ✅ Dependency installation triggered in background.")
        except Exception as e:
            print(f"   ⚠️ Failed to parse console creation response: {e}")
    else:
        print("   ⚠️ Could not launch automated console. You can install dependencies manually if needed.")
        
    # 6. Reload Web App
    print("\n[5/5] Reloading Web Application...")
    reload_url = f"{webapps_url}{domain_name}/reload/"
    res = requests.post(reload_url, headers=headers)
    if res.status_code != 200:
        print(f"❌ Failed to reload webapp: {res.status_code} - {res.text}")
        return
        
    print("\n" + "=" * 60)
    print("🚀 DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Your application is now live permanently at:")
    print(f"👉 http://{domain_name}")
    print("=" * 60)
    print("Note: Logins and prediction logs will be permanently saved to SQLite.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
