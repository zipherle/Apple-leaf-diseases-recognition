# Apple Leaf Diseases Recognition

This project leverages ResNet, a deep learning architecture, to recognize and classify apple leaf diseases. It aims to provide a robust solution for researchers and agriculturalists to identify and diagnose diseases in apple leaves with high accuracy.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Applications](#running-the-applications)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [License](#license)

---

## Overview

Apple leaf diseases can significantly impact the yield and quality of apple crops. The purpose of this project is to create a deep learning-based classification system using ResNet to automate the detection of diseases in apple leaves. This system can help improve decision-making in agriculture by enabling early detection and treatment.

---

## Features

- **Deep Learning Architecture**: Utilizes ResNet for robust feature extraction and classification.
- **Preprocessing Pipeline**: Includes tools for image preprocessing and augmentation.
- **Interactive Applications**: Provides `app.py` and `app-vi.py` for running the model through a web-based interface using Streamlit.
- **Evaluation Metrics**: Measures model accuracy, precision, recall, and F1 score.

---

## Installation

Follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/zipherle/Apple-leaf-diseases-recognition.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Apple-leaf-diseases-recognition
   ```
3. Setup the enviroment:
   ```bash
   sudo apt install python3-venv -y
   python3 -m venv .venv
   source .venv/bin/activate
   ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

This project contains two main applications (`app.py` and `app-vi.py`) and scripts for training and evaluating the model.

### Running the Applications

1. To launch the English version of the application:
   ```bash
   streamlit run app.py
   ```
2. To launch the Vietnamese version of the application:
   ```bash
   streamlit run app-vi.py
   ```

Both applications will open in your default web browser, and you can upload images of apple leaves for classification.

---

## Dataset

The dataset used for this project consists of images of apple leaves labeled with the corresponding diseases. Ensure your dataset is structured as follows:

```plaintext
data/
├── APPLE ROT LEAVES/
├── HEALTHY LEAVES/
├── LEAF BLOTCH/
├── SCAB LEAVES/
└── archive.zip
```

Make sure to replace the disease categories (`healthy`, `scab`, `rust`, `black_rot`) with the actual labels in your dataset.

---

## Model Architecture

The project uses ResNet, a convolutional neural network architecture, which is pre-trained on ImageNet. Key features of the model:
- **Transfer Learning**: The ResNet model is fine-tuned on the apple leaf disease dataset.
- **Customizable Hyperparameters**: Hyperparameters such as learning rate, batch size, and epochs can be adjusted in `config.py`.

---

## Training

To train the model, follow these steps:

1. Prepare the dataset as described in the [Dataset](#dataset) section.
2. Open the terminal and run the training script:
   ```bash
   python train.py
   ```
3. Adjust hyperparameters as needed in `config.py` (e.g., learning rate, batch size, number of epochs).
4. During training, the model’s progress (e.g., loss and accuracy) will be logged.
5. The trained model will be saved in the `models/` directory upon completion.

---

## Results

Model performance is evaluated based on the following metrics:
- **Accuracy**: Percentage of correctly classified samples.
- **Precision**: Ratio of true positive predictions to total positive predictions.
- **Recall**: Ratio of true positive predictions to all actual positives.
- **F1 Score**: Harmonic mean of precision and recall.

You can visualize the results and metrics by running the evaluation script:
```bash
python evaluate.py
```

---

