# Facial Keypoint Detection — CNN Regression with PyTorch

Detecting 67 facial keypoints on face images using a 
custom CNN regression model built in PyTorch, trained 
with MSE loss and evaluated with train/validation 
loss curves.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Task](https://img.shields.io/badge/Task-Keypoint%20Detection-purple)
![Type](https://img.shields.io/badge/Type-Regression-green)

---

## Overview

Builds a CNN regression model to predict the (x, y) 
coordinates of 67 facial keypoints per image — 
landmarks such as eye corners, nose tip, mouth edges, 
and jawline points. Unlike classification (which predicts 
a class label), this is a **regression task** — the model 
outputs 134 continuous values (67 keypoints × 2 coordinates). 
The pipeline covers annotation loading, a custom PyTorch 
`Dataset` class, data augmentation, a 3-block CNN 
architecture, AdamW training with MSE loss, and a 
prediction visualisation function that overlays 
detected keypoints on face images.

---

## Dataset

- **Annotation file:** `key_points.csv`
  - Each row: image filename + 67 (x, y) keypoint 
    coordinate pairs
  - Column names lowercased on load
  - Rows with missing keypoints dropped via `dropna()`
- **Image folders:**
  - `dataset_1/` — primary face images
  - `dataset_2/` — secondary face images
- **Train / Val split:**
  - Training: 1,200 samples
  - Validation: remaining samples
- **Source:** [Download from Kaggle](https://www.kaggle.com/datasets/blessontomjoseph/dataset-1-and-dataframe) and [Download from Kaggle](https://www.kaggle.com/datasets/blessontomjoseph/dataset-2) 


---

## Approach

### Custom Dataset Class (FaceData)
- Loads face image from disk using `PIL.Image`
- Looks up 67 keypoint (x, y) pairs from 
  the annotation CSV by image filename
- Returns `{'x': image_tensor, 'y': keypoints_tensor}`
  where `y` has shape `(67, 2)`

### Data Augmentation (Training only)
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, 
                           contrast=0.2, 
                           saturation=0.2)
])
```

### CNN Architecture (KeypointModel)

| Layer | Type | Details |
|-------|------|---------|
| Conv1 | Conv2d + ReLU + MaxPool | 3→32 channels, 3×3 kernel |
| Conv2 | Conv2d + ReLU + MaxPool | 32→64 channels, 3×3 kernel |
| Conv3 | Conv2d + ReLU + MaxPool | 64→128 channels, 3×3 kernel |
| FC1 | Linear + ReLU | 128×64×64 → 512 |
| FC2 | Linear | 512 → 134 (67×2 keypoints) |

Output is reshaped to `(batch, 67, 2)` — 
one (x, y) coordinate pair per keypoint.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | MSELoss |
| Optimiser | AdamW |
| Learning Rate | 1e-5 |
| Epochs | 5 |
| Batch Size | 5 (train and val) |
| Device | CUDA if available, else CPU |

### Prediction Visualisation
`plot_preds()` function:
- Loads a face image from disk
- Runs inference with `torch.no_grad()`
- Draws red ellipses at each predicted 
  (x, y) keypoint using `PIL.ImageDraw`

---

## Results

| Metric | Value |
|--------|-------|
| Keypoints Detected | 67 per image |
| Output Shape | (batch, 67, 2) |
| Epochs | 5 |
| Final Train Loss | [add from notebook output] |
| Final Val Loss | [add from notebook output] |

> Run the notebook and copy the final epoch 
> `train_loss` and `val_loss` values printed 
> by the training loop into the table above.

---

## Visualisations

1. **Train / Val loss curve** — red (train) and 
   blue (val) loss lines plotted over 5 epochs
2. **Keypoint overlay** — `plot_preds()` draws 
   predicted keypoints as red ellipses 
   directly on the face image

---

## Technologies Used

Python, PyTorch (nn.Module, DataLoader, MSELoss, 
AdamW), torchvision (transforms), PIL (Image, 
ImageDraw), Pandas, NumPy, tqdm, Matplotlib

---

## How to Run

```bash
git clone https://github.com/OyelolaIbrahim/facial-keypoint-detection.git
cd facial-keypoint-detection
pip install -r requirements.txt
jupyter notebook facial_keypoint.ipynb
```




torch
torchvision
pandas
numpy
matplotlib
pillow
tqdm
jupyter
```
