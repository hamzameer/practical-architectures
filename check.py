"""check.py

This script checks if the libraries required for the project are installed
"""

# List of libraries to check for import
libraries_to_check = [
    "torch",
    "transformers",
    "datasets",
    "mediapy",
    "mingpt",
    "torch_spatial_kmeans",
]

successful_imports = []

# Check and print the results for each library
for lib_name in libraries_to_check:
    try:
        __import__(lib_name)
        if lib_name == "torch" and torch.cuda.is_available():
            print(f"{lib_name} (GPU available)")
        else:
            print(lib_name)
        successful_imports.append(lib_name)
    except ImportError:
        print(f"Warning: {lib_name} not found")

# Check GPU availability in Torch

if "torch" in successful_imports:
    if  torch.cuda.is_available():
        print("GPU is available in Torch")
    else:
        print("GPU is not available in Torch")

# Define the path to the video
VIDEO_PATH = "your_video_path_here"

# Function to check if a video can be loaded successfully with mediapy
def check_video_load(video_path):
    try:
        mediapy.read(video_path)
        return True
    except Exception as e:
        return False

# Print the result of loading the video

if "mediapy" in successful_imports:
    if check_video_load(VIDEO_PATH):
        print(f"Video at '{VIDEO_PATH}' loaded successfully with mediapy")
    else:
        print(f"Warning: Failed to load video at '{VIDEO_PATH}' with mediapy")
else:
    print("mediapy not found so skipping video load check")
