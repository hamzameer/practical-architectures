"""check.py

This script checks if the libraries required for the project are installed
"""
import os

# List of libraries to check for import
libraries_to_check = [
    "torch",
    "transformers",
    "datasets",
    "mediapy",
    "mingpt",
    "torch_spatial_kmeans",
    "ffmpeg"
]

successful_imports = []

# Check and print the results for each library
for lib_name in libraries_to_check:
    try:
        __import__(lib_name)
        print(f"Success: {lib_name} library found")
        successful_imports.append(lib_name)
    except ImportError:
        print(f"WARNING: {lib_name} not found")

print()

# Check GPU availability in Torch

if "torch" in successful_imports:
    import torch
    if  torch.cuda.is_available():
        print("Success: GPU is available in Torch")
    else:
        print("WARNING: GPU is not available in Torch")

# Function to check if a video can be loaded successfully with mediapy
def check_video_load(video_path):
    import mediapy

    try:
        with mediapy.VideoReader(video_path) as reader:
            for frame in reader:
                break

        return True
    except Exception as e:
        print(e)
        return False

# Print the result of loading the video

if "mediapy" in successful_imports:
    VIDEO_PATH = os.getenv("EXAMPLE_FILE")

    if check_video_load(VIDEO_PATH):
        print(f"Video at '{VIDEO_PATH}' loaded successfully with mediapy")
    else:
        print(f"WARNING: Failed to load video at '{VIDEO_PATH}' with mediapy")
else:
    print("WARNING: mediapy not found so skipping video load check")
