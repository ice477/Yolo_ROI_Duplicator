# Image Similarity Detector

A deep learning-based tool for detecting similar regions of interest (ROIs) across multiple images using YOLO object detection and ResNet50 feature extraction. The tool is particularly optimized for analyzing surveillance footage and identifying duplicate objects across frames.

*Modified from: [livejiaquan/image-similarity-detector](https://github.com/livejiaquan/image-similarity-detector): A deep learning-based image similarity detection tool using ResNet50 and Cosine Similarity. It finds duplicate or similar images, generates reports, and supports CCTV image analysis.*

## Features

- **Object Detection**: Uses YOLO model to detect objects in images
- **Feature Extraction**: Extracts deep features from detected objects using ResNet50
- **Similarity Analysis**: Computes cosine similarity between object features
- **Customizable Classes**: Can focus on specific object classes (e.g., PPE equipment)
- **Detailed Reporting**: Generates comprehensive JSON reports with similarity metrics
- **CCTV Optimization**: Specifically designed for analyzing surveillance footage

## Requirements

- Python 3.6+
- PyTorch
- OpenCV
- Pillow
- Ultralytics YOLO
- NumPy
- tqdm

## Installation

```bash
pip install torch torchvision opencv-python pillow ultralytics numpy tqdm
```

## Usage

1. Prepare your YOLO model weights file
2. Set the folder paths containing your images in the `folder_list` variable
3. Run the script:

```bash
python image_similarity_detector.py
```

## Configuration

You can modify the following parameters in the `main()` function:

- `folder_list`: List of directories containing images to analyze
- `yolo_weights`: Path to your trained YOLO model weights
- `threshold`: Similarity threshold (0.0-1.0) for detecting similar ROIs (default: 0.975)
- `classes`: List of specific classes to detect (set to None for all classes)

## How It Works

1. **Object Detection**: The script uses YOLO to detect objects in each image
2. **Feature Extraction**: For each detected object, a ResNet50 model extracts a feature vector
3. **Similarity Computation**: Cosine similarity is calculated between all pairs of feature vectors
4. **Analysis**: Objects with similarity scores above the threshold are identified as similar
5. **Reporting**: Results are saved as JSON files with detailed metrics

## Output

The script generates a timestamped results folder containing:
- `report.json`: Contains analysis metrics and details of similar ROIs

The analysis includes:
- Total images and ROIs processed
- Duplicate detection rate
- Most frequent duplicates
- Average similarity by class pairs

## Example

```python
# Example configuration
folder_list = ["/path/to/your/images"]
yolo_weights = "/path/to/your/model.pt"
threshold = 0.975
classes = None # Set to None to detect all classes
```

## Notes

- The script is optimized for PPE (Personal Protective Equipment) detection
- Performance may vary based on image quality and lighting conditions
- For large datasets, consider running on a GPU for faster processing
