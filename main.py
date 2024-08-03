# main.py
# This script is part of "cognosis - cognitive coherence coroutines" project,
# which is a pythonic implementation of a model cognitive system,
# utilizing concepts from signal processing, cognitive theories,
# and machine learning to create adaptive systems.

import subprocess
import sys
import argparse
import os
import shutil
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, TypeVar, Generic, Optional, List
import logging
import json
import struct

T = TypeVar('T')  # Type Variable to allow type-checking, linting,.. of Generic "T" and "V"

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
            sys.exit(e.returncode)

def ensure_delete(path):
    """Ensure that a file or directory can be deleted"""
    try:
        os.chmod(path, 0o777)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        print(f"Failed to delete {path}. Reason: {e}")

def ensure_path():
    """Ensure that the PATH is set correctly"""
    path = os.getenv('PATH')
    if 'desired_path_entry' not in path:
        os.environ['PATH'] = f'/desired_path_entry:{path}'
        print('Updated PATH environment variable.')


state = {
    "pipx_installed": False,
    "pdm_installed": False,
    "virtualenv_created": False,
    "dependencies_installed": False,
    "lint_passed": False,
    "code_formatted": False,
    "tests_passed": False,
    "benchmarks_run": False,
    "pre_commit_installed": False,
}

def update_path():
    home = os.path.expanduser("~")
    local_bin = os.path.join(home, ".local", "bin")
    if local_bin not in os.environ["PATH"]:
        os.environ["PATH"] = f"{local_bin}:{os.environ['PATH']}"
        print(f"Added {local_bin} to PATH")
    # Call this after installations

def ensure_pipx():
    """Ensure pipx is installed"""
    global state
    try:
        subprocess.run("pipx --version", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        state['pipx_installed'] = True
        print("pipx is already installed.")
    except subprocess.CalledProcessError:
        print("pipx not found, installing pipx...")
        run_command("pip install pipx", shell=True)
        run_command("pipx ensurepath", shell=True)
        state['pipx_installed'] = True

def ensure_pdm():
    """Ensure pdm is installed via pipx"""
    global state
    try:
        output = subprocess.run("pipx list", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if b'pdm' not in output.stdout:
            raise KeyError('pdm not found in pipx list')
        print("pdm is already installed.")
        state['pdm_installed'] = True
    except (subprocess.CalledProcessError, KeyError):
        print("pdm not found, installing pdm...")
        run_command("pipx install pdm", shell=True)
        state['pdm_installed'] = True


def prompt_for_mode():
    """Prompt the user to choose between development and non-development setup"""
    while True:
        choice = input("Choose setup mode: [d]evelopment or [n]on-development? ").lower()
        if choice in ['d', 'n']:
            return choice
        print("Invalid choice, please enter 'd' or 'n'.")


def install(mode):
    """Run installation"""
    if mode == 'dev':
        run_command("pdm install", shell=True, verbose=True)
    else:
        run_command("pipx install . --force", shell=True, verbose=True)


def lint():
    """Run linting tools"""
    global state
    run_command("pdm run flake8 .", shell=True)
    run_command("pdm run black --check .", shell=True)
    run_command("pdm run mypy .", shell=True)
    state['lint_passed'] = True

def format_code():
    """Format the code"""
    global state
    run_command("pdm run black .", shell=True)
    run_command("pdm run isort .", shell=True)
    state['code_formatted'] = True

def test():
    """Run tests"""
    global state
    run_command("pdm run pytest", shell=True)
    state['tests_passed'] = True

def bench():
    """Run benchmarks"""
    global state
    run_command("pdm run python src/bench/bench.py", shell=True)
    state['benchmarks_run'] = True

def pre_commit_install():
    """Install pre-commit hooks"""
    global state
    run_command("pdm run pre-commit install", shell=True)
    state['pre_commit_installed'] = True


def introspect():
    """Introspect the current state and print results"""
    print("Introspection results:")
    for key, value in state.items():
        print(f"{key}: {'✅' if value else '❌'}")


def update_shell_environment():
    if platform.system() != "Windows":
        home = os.path.expanduser("~")
        bashrc_path = os.path.join(home, ".bashrc")
        if os.path.exists(bashrc_path):
            subprocess.run(f"source {bashrc_path}", shell=True, executable="/bin/bash")
            print("Updated shell environment from .bashrc")
        else:
            print(".bashrc not found, shell environment might not be up to date")
    else:
        print("On Windows, manual PATH update might be necessary")
    # Call this function after ensure_pipx() and ensure_pdm()


def main():
    ensure_pipx()
    ensure_pdm()
    update_shell_environment()
    update_path()

    parser = argparse.ArgumentParser(description="Setup and run Abraxus project")
    parser.add_argument('-m', '--mode', choices=['dev', 'non-dev'], help="Setup mode: 'dev' or 'non-dev'")
    parser.add_argument('--run-user-main', action='store_true', help="Run the user-defined main function")
    args = parser.parse_args()
    mode = args.mode
    if not mode:
        choice = prompt_for_mode()
        mode = 'dev' if choice == 'd' else 'non-dev'
    if mode == 'dev':
        install(mode)
        lint()
        format_code()
        test()
        bench()
        pre_commit_install()
    else:
        install(mode)
        # Use pipx run instead of pdm run for non-dev mode
        run_command("pipx run main", shell=True)

    if args.run_user_main:
        from src.main import usermain
        usermain()  # Call the user-defined main function
    else:
        print("No additional arguments provided. Skipping user-defined main function.")

    introspect()

if __name__ == "__main__":
    main()
    # use ./src/utils/cleanup.py to clean up the environment (optional: clean up pipx and pdm and conda)