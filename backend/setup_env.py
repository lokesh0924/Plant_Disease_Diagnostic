#!/usr/bin/env python3
"""
Setup script to help configure environment variables for the Farm Health application.
This script will create a .env file with the necessary API keys.
"""

import os
import sys

def create_env_file():
    """Create .env file with environment variables"""
    env_content = """# NVIDIA API Configuration
# Get your API key from: https://console.nvidia.com/
NVIDIA_API_KEY=your_nvidia_api_key_here

# Weather API Configuration  
# Get your API key from: https://openweathermap.org/api
WEATHER_API_KEY=your_weather_api_key_here
"""
    
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    
    if os.path.exists(env_path):
        print(f".env file already exists at {env_path}")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Keeping existing .env file")
            return
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        print(f"Created .env file at {env_path}")
        print("\nPlease edit the .env file and add your actual API keys:")
        print("1. NVIDIA_API_KEY: Get from https://console.nvidia.com/")
        print("2. WEATHER_API_KEY: Get from https://openweathermap.org/api")
    except Exception as e:
        print(f"Error creating .env file: {e}")

def check_environment():
    """Check if environment variables are set"""
    nvidia_key = os.getenv('NVIDIA_API_KEY')
    weather_key = os.getenv('WEATHER_API_KEY')
    
    print("Environment variable status:")
    print(f"NVIDIA_API_KEY: {'✓ Set' if nvidia_key else '✗ Not set'}")
    print(f"WEATHER_API_KEY: {'✓ Set' if weather_key else '✗ Not set'}")
    
    if not nvidia_key or not weather_key:
        print("\nSome environment variables are missing.")
        print("Please set them or create a .env file with the required keys.")

if __name__ == "__main__":
    print("Farm Health Application - Environment Setup")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_environment()
    else:
        create_env_file()
        print("\nAfter setting up your API keys, you can check the status with:")
        print("python setup_env.py check")
