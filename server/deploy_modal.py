#!/usr/bin/env python3
"""
Script to deploy your model to Modal and set up the GPU service.

Usage:
1. Install Modal: pip install modal
2. Set up Modal token: modal token new
3. Upload your model files: python deploy_modal.py upload
4. Deploy the service: python deploy_modal.py deploy
"""

import subprocess
import sys
from pathlib import Path
import shutil


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def upload_model_files():
    """Upload model files to Modal volume"""
    print("🔄 Uploading model files to Modal...")

    # Check if model files exist
    model_path = Path("app/model/model.pt")
    tokenizer_path = Path("app/model/tokenizer")

    if not model_path.exists():
        print(f"❌ Model file not found at {model_path}")
        return False

    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found at {tokenizer_path}")
        return False

    # Use Modal volume copy commands directly with --force to overwrite
    commands = [
        f"modal volume put --force story-model {model_path} model.pt",
        f"modal volume put --force story-model {tokenizer_path} tokenizer",
        f"modal volume put --force story-model app/model/transformer.py app/model/transformer.py",
        f"modal volume put --force story-model app/model/__init__.py app/model/__init__.py",
        f"modal volume put --force story-model app/__init__.py app/__init__.py",
        f"modal volume put --force story-model app/schemas app/schemas",
    ]

    # Create volume first
    if not run_command("modal volume create story-model", "Creating Modal volume"):
        print("⚠️  Volume might already exist, continuing...")

    # Upload files
    for cmd in commands:
        if not run_command(cmd, f"Uploading {cmd.split()[-1]}"):
            return False

    return True


def deploy_modal_app():
    """Deploy the Modal app"""
    return run_command("modal deploy modal_app.py", "Deploying Modal app")


def get_modal_url():
    """Get the Modal endpoint URL"""
    print("🔗 Getting Modal endpoint URL...")
    try:
        result = subprocess.run(
            "modal app list", shell=True, check=True, capture_output=True, text=True
        )

        # Parse output to find the URL
        lines = result.stdout.split("\n")
        for line in lines:
            if "story-generator" in line and "https://" in line:
                # Extract URL from the line
                parts = line.split()
                for part in parts:
                    if part.startswith("https://"):
                        print(f"✅ Modal endpoint URL: {part}")
                        print(f"\n📋 Add this to your environment variables:")
                        print(f"export MODAL_ENDPOINT_URL={part}")
                        print(f"export USE_MODAL=true")
                        return part

        print(
            "⚠️  Could not parse Modal URL from output. Check 'modal app list' manually"
        )
        return None

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to get Modal URL: {e.stderr}")
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python deploy_modal.py [upload|deploy|url]")
        print("  upload - Upload model files to Modal")
        print("  deploy - Deploy Modal app")
        print("  url    - Get Modal endpoint URL")
        sys.exit(1)

    command = sys.argv[1]

    if command == "upload":
        success = upload_model_files()
        if success:
            print("\n🎉 Model files uploaded successfully!")
            print("Next step: python deploy_modal.py deploy")
        else:
            print("\n❌ Upload failed. Please check the errors above.")
            sys.exit(1)

    elif command == "deploy":
        success = deploy_modal_app()
        if success:
            print("\n🎉 Modal app deployed successfully!")
            print("Next step: python deploy_modal.py url")
        else:
            print("\n❌ Deploy failed. Please check the errors above.")
            sys.exit(1)

    elif command == "url":
        url = get_modal_url()
        if url:
            print(
                "\n🎉 Ready to use Modal! Set the environment variables and restart your server."
            )
        else:
            print("\n❌ Could not get URL. Check 'modal app list' manually.")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
