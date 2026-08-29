#!/usr/bin/env python3
import subprocess
import os

os.chdir(r"c:\Users\priya\Desktop\THERMO WATCH")

commands = [
    ["git", "status"],
    ["git", "add", ".gitignore"],
    ["git", "add", "*.py"],
    ["git", "add", "scripts/"],
    ["git", "add", "processed/"],
    ["git", "status"],
    ["git", "commit", "-m", "Add all project source files and data"],
    ["git", "push", "-u", "origin", "master"],
]

for cmd in commands:
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error: Command failed with code {result.returncode}")
