import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # For progress bar

# Define dataset paths
DATASET_PATH = os.path.dirname(os.path.abspath(__file__))  # Root folder (YOLO_Dataset)
TRAIN_LABEL_PATH = os.path.join(DATASET_PATH, "train", "labels")
VAL_LABEL_PATH = os.path.join(DATASET_PATH, "val", "labels")
CLASS_FILE = os.path.join(DATASET_PATH, "class.txt")

# Create a directory to save the plots
PLOTS_SAVE_PATH = os.path.join(DATASET_PATH, "plots")
os.makedirs(PLOTS_SAVE_PATH, exist_ok=True)

# Load class names
with open(CLASS_FILE, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

num_classes = len(class_names)

# Function to analyze a dataset (train, val, or full)
def analyze_dataset(label_paths, dataset_name):
    print(f"\nProcessing {dataset_name} dataset...\n")
    
    class_counts = np.zeros(num_classes, dtype=int)
    bbox_widths = []
    bbox_heights = []
    aspect_ratios = []

    label_files = []
    for label_path in label_paths:
        label_files.extend([os.path.join(label_path, file) for file in os.listdir(label_path) if file.endswith(".txt")])

    # Process label files with a progress bar
    for label_file in tqdm(label_files, desc=f"Analyzing {dataset_name}"):
        with open(label_file, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            values = line.strip().split()
            
            if len(values) < 5:  # Ensure the line has the expected number of values
                continue  # Skip lines that don't have the expected number of values (class_id, x_center, y_center, width, height)
            
            try:
                class_id = int(values[0])
                x_center, y_center, width, height = map(float, values[1:])
            except ValueError:
                continue  # Skip any lines where conversion fails (e.g., malformed data)
            
            # Count class instances
            if class_id < num_classes:  # Ensure the class_id is within the valid range
                class_counts[class_id] += 1

            # Collect bounding box dimensions
            bbox_widths.append(width)
            bbox_heights.append(height)
            aspect_ratios.append(width / height)

    print(f"\nFinished processing {dataset_name}. Generating plots...\n")

    # Plot 1: Class Distribution
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, class_counts, color=["green", "red"])
    plt.xlabel("Class")
    plt.ylabel("Number of Instances")
    plt.title(f"{dataset_name} - Class Distribution")
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_SAVE_PATH, f"{dataset_name}_class_distribution.png"))
    plt.close()

    # Plot 2: Bounding Box Width & Height Distribution
    plt.figure(figsize=(8, 6))
    plt.hist(bbox_widths, bins=20, alpha=0.7, label="Width", color="blue")
    plt.hist(bbox_heights, bins=20, alpha=0.7, label="Height", color="orange")
    plt.xlabel("Normalized Size")
    plt.ylabel("Frequency")
    plt.title(f"{dataset_name} - Bounding Box Size Distribution")
    plt.legend()
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_SAVE_PATH, f"{dataset_name}_bbox_size_distribution.png"))
    plt.close()

    # Plot 3: Aspect Ratio Distribution
    plt.figure(figsize=(8, 6))
    plt.hist(aspect_ratios, bins=20, color="purple", alpha=0.7)
    plt.xlabel("Aspect Ratio (Width/Height)")
    plt.ylabel("Frequency")
    plt.title(f"{dataset_name} - Bounding Box Aspect Ratio Distribution")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_SAVE_PATH, f"{dataset_name}_aspect_ratio_distribution.png"))
    plt.close()

    print(f"\nPlots saved for {dataset_name} ✅\n")

# Run analysis for train, val, and whole dataset
if __name__ == "__main__":
    analyze_dataset([TRAIN_LABEL_PATH], "Train Dataset")
    analyze_dataset([VAL_LABEL_PATH], "Validation Dataset")
    analyze_dataset([TRAIN_LABEL_PATH, VAL_LABEL_PATH], "Full Dataset")  # Combined train + val
