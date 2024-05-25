# /main.py installs the abraxus architecture and asks to instantiate testing and further hypothesis of diverse (read: fuzzy) input/output data.
"""
make install
make lint
make format
make test
make bench
make pre-commit-install
pdm run python main.py
"""
import subprocess
import sys
import argparse
import os
import shutil
import platform

def run_command(command, check=True, shell=False):
    """Utility to run a shell command and handle exceptions"""
    try:
        result = subprocess.run(command, shell=shell, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with error:\n{e.stderr.decode()}")
        if check:
            exit(e.returncode)

def ensure_pipx():
    """Ensure pipx is installed"""
    try:
        subprocess.run("pipx --version", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("pipx not found, installing pipx...")
        run_command("pip install pipx")
        run_command("pipx ensurepath")

def ensure_pdm():
    """Ensure pdm is installed via pipx"""
    try:
        output = subprocess.run("pipx list", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if b'pdm' not in output.stdout:
            raise KeyError('pdm not found in pipx list')
        print("pdm is already installed.")
    except (subprocess.CalledProcessError, KeyError):
        print("pdm not found, installing pdm...")
        run_command("pipx install pdm")

def create_virtualenv():
    """Create a virtual environment and activate it using pdm"""
    # Ensure environment variables for pdm are set
    os.environ["PDM_VENV_IN_PROJECT"] = "1"
    
    venv_path = ".venv"
    if os.path.exists(venv_path):
        choice = input("Virtual environment already exists. Overwrite? (y/n): ").lower()
        if choice == 'y':
            shutil.rmtree(venv_path)
            run_command("pdm venv create")
        else:
            print("Reusing the existing virtual environment.")
    else:
        run_command("pdm venv create")

    # Activate virtual environment
    activate_script = os.path.join(venv_path, "Scripts", "activate")
    if platform.system() == "Windows":
        # On Windows, use PowerShell script to activate the virtual environment
        ps_activate_script = activate_script + '.ps1'
        if os.path.exists(ps_activate_script):
            run_command(f"& {ps_activate_script}", shell=True)
        else:
            run_command(f".\\{activate_script}.bat", shell=True)
    else:
        # POSIX shells (bash, zsh)
        run_command(f"source {activate_script}", shell=True)

    # Install dependencies
    run_command("pdm install")

def prompt_for_mode():
    """Prompt the user to choose between development and non-development setup"""
    while True:
        choice = input("Choose setup mode: [d]evelopment or [n]on-development? ").lower()
        if choice in ['d', 'n']:
            return choice
        print("Invalid choice, please enter 'd' or 'n'.")

def install():
    """Run installation"""
    run_command("pdm install")

def lint():
    """Run linting tools"""
    run_command("pdm run flake8 .")
    run_command("pdm run black --check .")
    run_command("pdm run mypy .")

def format_code():
    """Format the code"""
    run_command("pdm run black .")
    run_command("pdm run isort .")

def test():
    """Run tests"""
    run_command("pdm run pytest")

def bench():
    """Run benchmarks"""
    run_command("pdm run python src/bench/bench.py")

def pre_commit_install():
    """Install pre-commit hooks"""
    run_command("pdm run pre-commit install")

def main():
    # Ensure necessary tools are installed
    ensure_pipx()
    ensure_pdm()

    # Create virtual environment and install dependencies
    create_virtualenv()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Setup and run Abraxus project")
    parser.add_argument('-m', '--mode', choices=['dev', 'non-dev'], help="Setup mode: 'dev' or 'non-dev'")
    args = parser.parse_args()

    # Determine the mode if not specified in the arguments
    mode = args.mode
    if not mode:
        choice = prompt_for_mode()
        mode = 'dev' if choice == 'd' else 'non-dev'

    # Run appropriate commands based on the mode
    if mode == 'dev':
        install()
        lint()
        format_code()
        test()
        bench()
        pre_commit_install()
        run_command("pdm run python src/bench/bench.py")
    else:
        install()
        run_command("pdm run python main.py")  # Adjust command if needed
        
if __name__ == "__main__":
    main()