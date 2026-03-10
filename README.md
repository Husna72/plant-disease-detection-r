# 🌿 Plant Disease Detection using Deep Learning (R + Keras)

A deep learning image classifier that identifies **30 plant diseases** from leaf photos,
built entirely in R using Keras and MobileNet transfer learning.

## 🏆 Results

| Model | Validation Accuracy |
|---|---|
| CNN from Scratch | 93.42% |
| MobileNet Transfer Learning (Phase 1) | 95.08% |
| **MobileNet Fine-tuned (Phase 2)** | **96.49%** |

## 📁 Project Structure
```
plant-disease-detection-r/
├── 01_load_data.R          # Data pipeline & augmentation
├── 02_simple_cnn.R         # CNN from scratch
├── 03_transfer_learning.R  # MobileNet transfer learning
├── 04_evaluation.R         # Confusion matrix & metrics
├── 05_predict.R            # Prediction function
└── class_names.rds         # Saved class labels
```

## 🌱 Disease Classes (30 total)
Covers: Apple, Blueberry, Cherry, Orange, Peach, Pepper,
Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

## ⚙️ How It Works
1. 54,000+ leaf images from the PlantVillage dataset
2. Data augmentation (rotation, zoom, flipping) to prevent overfitting
3. **Phase 1:** Train custom head on frozen MobileNet → 95.08%
4. **Phase 2:** Fine-tune top MobileNet layers at 10x lower learning rate → 96.49%

## 🔍 Quick Prediction
```r
source("05_predict.R")
predict_plant_disease("path/to/leaf/image.jpg")

# Output:
# 🌿 Plant Disease Prediction
# ─────────────────────────────────────────
#  Plant    Disease         Confidence
#  Tomato   Early_blight    96.3%
```

## 📦 Requirements
```r
install.packages(c("keras", "tensorflow", "tidyverse", "caret"))
library(keras)
install_keras()
```

## 📂 Dataset
[PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
— Download and place the `color/` folder in the project root.

## 🛠️ Tech Stack
R · Keras · TensorFlow · MobileNet · tidyverse · ggplot2 · caret

