import os
import sys
import subprocess
import platform

# --- CONFIGURATION ---
ENV_NAME = "sam3_tracker"
PYTHON_VERSION = "3.10"
REQUIREMENTS_FILE = "requirements.txt"

def print_step(msg):
    print("\n" + "="*50)
    print(f"[*] {msg}")
    print("="*50 + "\n")

def run_command(command, error_msg="Command failed"):
    """
    Runs a shell command. 
    Handles Windows/Linux differences via shell=True.
    """
    try:
        # On Windows, shell=True is often required for conda commands to be found
        is_windows = platform.system().lower() == "windows"
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError:
        print(f"\n[ERROR] {error_msg}")
        print("Please check the error message above.")
        sys.exit(1)

def check_conda():
    """Checks if Conda is installed and accessible."""
    print_step("Checking for Conda...")
    try:
        subprocess.check_call(["conda", "--version"], shell=(platform.system().lower() == "windows"))
        print("Conda found.")
    except Exception:
        print("[ERROR] Conda is not installed or not in your PATH.")
        print("Please install Anaconda or Miniconda first.")
        sys.exit(1)

def create_environment():
    """Creates the Conda environment."""
    print_step(f"Creating Conda Environment: '{ENV_NAME}' with Python {PYTHON_VERSION}")
    
    # Check if env exists
    # We use 'conda create' with -y (yes) flag. 
    # If it exists, it might fail or ask to overwrite. 
    # We'll try to create it.
    
    cmd = f"conda create -n {ENV_NAME} python={PYTHON_VERSION} -y"
    run_command(cmd, "Failed to create environment.")

def install_requirements():
    """Installs dependencies from requirements.txt using pip inside the conda env."""
    print_step("Installing Dependencies...")

    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"[ERROR] {REQUIREMENTS_FILE} not found in current directory.")
        sys.exit(1)

    # We use 'conda run -n <env_name>' to execute pip inside the specific environment
    # This ensures we don't install to the base environment.
    
    # 1. Install PyTorch specifically (Optional recommendation)
    # It is usually safer to let pip handle it via requirements, 
    # but for GPU support, conda install pytorch is often better.
    # For simplicity in this script, we use the pip requirements file.
    
    cmd = f"conda run -n {ENV_NAME} pip install -r {REQUIREMENTS_FILE}"
    run_command(cmd, "Failed to install requirements.")

def install_sam3_dependencies():
    """
    SAM 3 often requires installing from Git or specific setup.
    This step adds the Meta SAM 2/3 specific requirements.
    """
    print_step("Installing SAM 3 specific dependencies (hydra, etc.)")
    
    # These are common dependencies for SAM 2/3 that might not be in standard pips
    pkgs = "hydra-core iopath"
    cmd = f"conda run -n {ENV_NAME} pip install {pkgs}"
    run_command(cmd, "Failed to install SAM3 extras.")

    print(f"Note: To use SAM 3, ensure you have the 'sam3' folder in your project")
    print(f"or install it via: conda run -n {ENV_NAME} pip install git+https://github.com/facebookresearch/sam2.git")

def main():
    print(f"Detected OS: {platform.system()}")
    
    # 1. Check Conda
    check_conda()
    
    # 2. Create Env
    create_environment()
    
    # 3. Install Requirements
    install_requirements()
    
    # 4. Install SAM extras
    install_sam3_dependencies()
    
    print_step("INSTALLATION COMPLETE!")
    print(f"To run your application:")
    print(f"1. Activate the environment:  conda activate {ENV_NAME}")
    print(f"2. Run the app:               python main.py")
    print("\nNote: If you have an NVIDIA GPU, you may need to reinstall PyTorch")
    print("with CUDA support specifically for your system.")

if __name__ == "__main__":
    main()