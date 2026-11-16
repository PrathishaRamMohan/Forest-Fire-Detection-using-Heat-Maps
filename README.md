# Forest-Fire-Detection-using-Heat-Maps
 A deep learning model on temperature data to classify potential forest fire risk, and visualized the results as heat maps to highlight fire-prone zones for early detection.

 # 🔥 FireVision: Forest Fire Detection Using Heat Maps

A deep learning project that detects forest fires in images, explains its predictions with Grad-CAM heat maps, and visualises real-world fire activity on a geographic heat map built from NASA FIRMS satellite data.

> Mini project for **AI23531 Deep Learning**, Department of Artificial Intelligence and Machine Learning, Rajalakshmi Engineering College (2025–2026).

---

## Overview

Forest fires are often noticed only after they have spread, because traditional monitoring depends on manual observation and delayed satellite analysis. FireVision combines three ideas to support earlier, more trustworthy detection:

1. **Detect:** a ResNet18 transfer-learning model classifies an image as `Fire` or `Non_Fire`.
2. **Explain:** Grad-CAM heat maps highlight the image regions that influenced each prediction, so a human can check the model's reasoning.
3. **Map:** a geospatial heat map plots active-fire detections from NASA FIRMS for any chosen bounding box.

## Features

- Binary fire / non-fire image classification with a fine-tuned **ResNet18** (ImageNet pretrained)
- Training with augmentation, best-model checkpointing (by validation F1) and saved training history
- Evaluation with Accuracy, Precision, Recall, F1-score, confusion matrix and classification report
- **Grad-CAM** overlays for model explainability
- **Geographic heat map** (interactive HTML) from NASA FIRMS active-fire data
- Single-command CLI for training, evaluation, prediction and heat map generation
- Dataset preparation utilities (splitting, restructuring, corrupt-image removal)

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Deep learning | PyTorch, torchvision (ResNet18, transfer learning) |
| Explainability | Grad-CAM, OpenCV |
| Evaluation | scikit-learn |
| Geospatial heat map | Folium (Leaflet), NASA FIRMS active-fire CSV |
| Data and plotting | NumPy, Pandas, Pillow, Matplotlib |

## How It Works

```
Forest image ──► Preprocess (224×224, normalise) ──► ResNet18 (fine-tuned) ──► Fire / Non_Fire
                                                            │
                                                            └──► Grad-CAM overlay (why this prediction?)

NASA FIRMS active-fire CSV ──► Filter by bounding box ──► Folium heat map (HTML)
```

**Model and training**

- Backbone: ResNet18 with ImageNet weights; final layer replaced for 2 classes
- Input size 224×224, ImageNet normalisation
- Training augmentation: random resized crop and horizontal flip
- Loss and optimiser: Cross-Entropy, Adam (learning rate 1e-4)
- Defaults: 8 epochs, batch size 24; the checkpoint with the best validation F1 is saved
- Optional `--feature_extract` flag freezes the backbone and trains only the classifier head

## Project Structure

```
.
├── forest_fire_project.py        # Main CLI: train, eval, predict (Grad-CAM), heatmap
├── generate_report.py            # Evaluates a model and saves metrics + Grad-CAM overlays
├── kerala_heatmap.html           # Sample FIRMS heat map (open in a browser)
├── reports/
│   ├── report.json               # Accuracy, classification report, confusion matrix
│   └── gradcam/                  # Grad-CAM overlays for the test images
├── Test_Data/                    # Sample test images (Fire / Non_Fire)
├── split_dataset.py              # Merge Train/Test folders and create a validation split
├── merge_and_split_dataset.py    # Re-split per class into train_final / val_final
├── restructure_dataset.py        # Reorganise the original dataset into train / val
├── check_dataset.py              # List dataset contents
├── check_images.py               # Find corrupt images
├── remove_bad_images.py          # Delete corrupt images
├── DL_exp_1.ipynb … DL_exp_9.ipynb  # Deep learning course experiment notebooks
└── DL MINI PROJ REPORT.pdf       # Full project report
```

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/PrathishaRamMohan/Forest-Fire-Detection-using-Heat-Maps.git
cd Forest-Fire-Detection-using-Heat-Maps

pip install torch torchvision scikit-learn pandas numpy pillow matplotlib folium requests opencv-python
```

### 2. Prepare the dataset

The full training dataset is not included in this repository. Arrange your images in this layout:

```
data/forest_fire_images/
├── train/
│   ├── Fire/
│   └── Non_Fire/
└── val/
    ├── Fire/
    └── Non_Fire/
```

The helper scripts (`restructure_dataset.py`, `split_dataset.py`, `remove_bad_images.py`) can create this structure from a `Train_Data` / `Test_Data` download.

### 3. Run

**Train**
```bash
python forest_fire_project.py train --data_dir data/forest_fire_images --epochs 8 --batch 24 --lr 1e-4
```
The best model is saved to `models/best_resnet18_fire.pth`, and training history to `models/train_history.json`.

**Evaluate**
```bash
python forest_fire_project.py eval --model_path models/best_resnet18_fire.pth --data_dir data/forest_fire_images
```

**Predict one image with a Grad-CAM overlay**
```bash
python forest_fire_project.py predict --model_path models/best_resnet18_fire.pth --image Test_Data/Fire/F_0.jpg --gradcam_out gradcam_example.jpg
```

**Generate a fire heat map for a region** (example bounding box covering Kerala, India)
```bash
python forest_fire_project.py heatmap --source viirs --days 7 --lat_min 8.2 --lat_max 12.8 --lon_min 74.8 --lon_max 77.4 --out kerala_heatmap.html
```
Open the generated HTML file in your browser to explore the map.

**Generate the full evaluation report with Grad-CAM overlays**
```bash
python generate_report.py --model_path models/best_resnet18_fire.pth --data_dir Test_Data --out_dir reports --device cpu
```

## Results

Evaluation on the 50-image sample test set included in this repository (`Test_Data`, 25 Fire and 25 Non_Fire images):

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Fire | 0.828 | 0.960 | 0.889 | 25 |
| Non_Fire | 0.952 | 0.800 | 0.870 | 25 |
| **Overall accuracy** | | | **0.88** | 50 |

**Confusion matrix** (rows are the true class, columns are the predicted class)

| | Predicted Fire | Predicted Non_Fire |
|---|---|---|
| **Actual Fire** | 24 | 1 |
| **Actual Non_Fire** | 5 | 20 |

The model rarely misses a real fire (recall 0.96 on the Fire class). Its main errors are Non_Fire images predicted as Fire. For early warning, false alarms are usually preferable to missed fires. Grad-CAM overlays for these images are in `reports/gradcam/`.

## Limitations

- The sample test set is small (50 images), so these figures are indicative rather than definitive.
- The model is an image classifier. It does not locate fires within an image or estimate their size.
- FIRMS data download URLs can change over time. If the automatic download fails, download an active-fire CSV from the [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) website and pass it to `create_heatmap_from_csv`.

## Future Improvements

- Add smoke detection and fire/smoke segmentation
- Build a web interface (for example Gradio or Streamlit) for uploads and live maps
- Combine image predictions with satellite hotspots to raise alerts automatically
- Evaluate on a larger, more varied dataset and compare other backbones (EfficientNet, MobileNet)

## Authors

- **Prathisha R** (231501119)

Department of Artificial Intelligence and Machine Learning, Rajalakshmi Engineering College, Chennai.

## Acknowledgements

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) for active-fire data
- PyTorch and torchvision for the pretrained ResNet18 model

