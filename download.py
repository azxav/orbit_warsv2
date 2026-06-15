import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaggle/orbit-wars-episodes-2026-06-05", output_dir="./replays5")

print("Path to dataset files:", path)