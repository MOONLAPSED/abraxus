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

def run_command(command, check=True, shell=False, verbose=False):
    """Utility to run a shell command and handle exceptions"""
    if verbose:
        command += " -v"
    try:
        result = subprocess.run(command, check=check, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        run_command("pip install pipx", shell=True)
        run_command("pipx ensurepath", shell=True)

def ensure_pdm():
    """Ensure pdm is installed via pipx"""
    try:
        output = subprocess.run("pipx list", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if b'pdm' not in output.stdout:
            raise KeyError('pdm not found in pipx list')
        print("pdm is already installed.")
    except (subprocess.CalledProcessError, KeyError):
        print("pdm not found, installing pdm...")
        run_command("pipx install pdm", shell=True)

def create_virtualenv():
    """Create a virtual environment and activate it using pdm"""
    # Ensure environment variables for pdm are set
    os.environ["PDM_VENV_IN_PROJECT"] = "1"
    
    venv_path = ".venv"
    if os.path.exists(venv_path):
        choice = input("Virtual environment already exists. Overwrite? (y/n): ").lower()
        if choice == 'y':
            shutil.rmtree(venv_path)
            run_command("pdm venv create", shell=True)
        else:
            print("Reusing the existing virtual environment.")
    else:
        run_command("pdm venv create", shell=True)

    # Activate virtual environment
    if platform.system() == "Windows":
        ps_activate_script = os.path.join(venv_path, "Scripts", "Activate.ps1")
        cmd_activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
        if os.path.exists(ps_activate_script):
            run_command(f'powershell.exe -ExecutionPolicy Bypass -File {ps_activate_script}', shell=True)
        else:
            run_command(cmd_activate_script, shell=True)
    else:
        # POSIX shells (bash, zsh)
        activate_script = os.path.join(venv_path, "bin", "activate")
        run_command(f"source {activate_script}", shell=True)

    # Ensure lockfile exists
    run_command("pdm lock", shell=True)
    
    # Install dependencies with verbose logging
    run_command("pdm install", shell=True, verbose=True)

def prompt_for_mode():
    """Prompt the user to choose between development and non-development setup"""
    while True:
        choice = input("Choose setup mode: [d]evelopment or [n]on-development? ").lower()
        if choice in ['d', 'n']:
            return choice
        print("Invalid choice, please enter 'd' or 'n'.")

def install():
    """Run installation"""
    run_command("pdm install", shell=True, verbose=True)

def lint():
    """Run linting tools"""
    run_command("pdm run flake8 .", shell=True)
    run_command("pdm run black --check .", shell=True)
    run_command("pdm run mypy .", shell=True)

def format_code():
    """Format the code"""
    run_command("pdm run black .", shell=True)
    run_command("pdm run isort .", shell=True)

def test():
    """Run tests"""
    run_command("pdm run pytest", shell=True)

def bench():
    """Run benchmarks"""
    run_command("pdm run python src/bench/bench.py", shell=True)

def pre_commit_install():
    """Install pre-commit hooks"""
    run_command("pdm run pre-commit install", shell=True)

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
        run_command("pdm run python src/bench/bench.py", shell=True)
    else:
        install()
        run_command("pdm run python main.py", shell=True)  # Adjust command if needed
        
if __name__ == "__main__":
    main()
    #usermain()

def usermain(arg=None):
    import src.app
    if arg:
        print(f"Main called with argument: {arg}")
    else:
        print("Main called with no arguments")

    # Demonstrations
    top = AtomicData(value=True)
    bottom = AtomicData(value=False)
    
    # FormalTheory demonstration
    formal_theory = FormalTheory[int](top_atom=top, bottom_atom=bottom)
    encoded_ft = formal_theory.encode()
    print("Encoded FormalTheory:", encoded_ft)
    new_formal_theory = FormalTheory[int](top_atom=top, bottom_atom=bottom)
    new_formal_theory.decode(encoded_ft)
    print("Decoded FormalTheory:", new_formal_theory)

    # Execution example - not fully functional but placeholder for showing usage
    try:
        result = formal_theory.execute(lambda x, y: x + y, 1, 2)
        print("Execution Result:", result)
    except NotImplementedError:
        print("Execution logic not implemented for FormalTheory.")

    # AtomicData demonstration
    atomic_data = AtomicData(value="Hello World")
    encoded_data = atomic_data.encode()
    print("Encoded AtomicData:", encoded_data)
    new_atomic_data = AtomicData(value=None)
    new_atomic_data.decode(encoded_data)
    print("Decoded AtomicData:", new_atomic_data)

    # Thread-safe context example
    print("Using ThreadSafeContextManager")
    with ThreadSafeContextManager():
        # Any thread-safe operations here
        pass

    # Using ScopeLifetimeGarden
    print("Using ScopeLifetimeGarden")
    garden = ScopeLifetimeGarden()
    garden.set(AtomicData(value="Initial Data"))
    print("Garden Data:", garden.get())
    """
    with garden.scope():
        garden.set(AtomicData(value="New Data"))
        print("Garden Data:", garden.get())
    print("Garden Data:", garden.get())
    """



if __name__ == "__main__":
    main()