# YOLO Dataset Visualization

This repository is designed to help visualize and analyze the distribution of bounding boxes, class labels, and other key metrics in a YOLO-style dataset. The dataset is organized into `train` and `val` directories, with each directory containing `images` and `labels` subdirectories.

## Folder Structure

The expected folder structure is as follows:

```
Dataset/
│── train/
│   ├── images/
│   ├── labels/
│── val/
│   ├── images/
│   ├── labels/
│── class.txt
│── visualize.py
│── plots/    <-- Generated plots will be saved here
```

- `train/labels/`: Contains the label files for the training dataset
- `val/labels/`: Contains the label files for the validation dataset
- `class.txt`: Contains the class names used for the dataset (one per line)
- `visualize.py`: Python script to generate and save visualizations
- `plots/`: Directory where generated plots will be saved

## Requirements

This project requires the following Python libraries:

- `matplotlib`
- `numpy`
- `tqdm`

To install these dependencies, run the following command:

```bash
pip install matplotlib numpy tqdm
```

## Usage

To generate the visualizations, simply run the `visualize.py` script.

1. Navigate to the `YOLO_Dataset` directory
2. Run the Python script:

```bash
python visualize.py
```

The script will:

- Analyze the class distribution, bounding box sizes (width and height), and aspect ratios of the bounding boxes
- Generate plots for the `train`, `val`, and `full` datasets (train + val)
- Save the generated plots in the `plots/` directory

### Plots Generated

The following plots will be generated and saved in the `plots/` directory:

1. **Class Distribution**: A bar chart showing the number of instances per class
2. **Bounding Box Size Distribution**: Histograms showing the distribution of bounding box widths and heights
3. **Aspect Ratio Distribution**: A histogram showing the distribution of bounding box aspect ratios

Each plot is saved as a PNG file with a filename indicating the dataset it corresponds to. Example filenames:

- `Train Dataset_class_distribution.png`
- `Validation Dataset_class_distribution.png`
- `Full Dataset_class_distribution.png`

### Example Output

After running the script, the `plots/` directory will contain files like:

```
YOLO_Dataset/
│── plots/
│   ├── Train Dataset_class_distribution.png
│   ├── Train Dataset_bbox_size_distribution.png
│   ├── Train Dataset_aspect_ratio_distribution.png
│   ├── Validation Dataset_class_distribution.png
│   ├── Validation Dataset_bbox_size_distribution.png
│   ├── Validation Dataset_aspect_ratio_distribution.png
│   ├── Full Dataset_class_distribution.png
│   ├── Full Dataset_bbox_size_distribution.png
│   ├── Full Dataset_aspect_ratio_distribution.png
```

## Notes

- Ensure your dataset follows the YOLO format for labels (with each label file containing one line per object in the image)
- The `class.txt` file should contain a list of class names, with one name per line (no additional formatting required)
- The bounding box coordinates in each label file should be in the format:

  ```
  class_id x_center y_center width height
  ```
  where `x_center`, `y_center`, `width`, and `height` are normalized by the image dimensions
