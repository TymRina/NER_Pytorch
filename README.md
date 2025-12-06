# NER_Pytorch

A PyTorch-based Named Entity Recognition (NER) system for Chinese text processing.

## Overview

This project implements a comprehensive NER system specifically designed for Chinese text analysis. It provides end-to-end functionality from data preprocessing to model training, validation, and inference, with a user-friendly graphical interface for easy interaction.

## Features

- **Data Preprocessing**: Efficient handling of Chinese text datasets for NER tasks
- **Model Training & Validation**: Complete pipeline for training and evaluating NER models
- **Entity Recognition**: Accurate identification of named entities in Chinese text
- **Long Text Support**: Robust processing of lengthy documents with proper entity position tracking
- **Graphical Interface**: Intuitive PyQt5-based GUI for user interaction
- **Pretrained Model**: Includes trained model weights for immediate use

## Technology Stack

- **Framework**: PyTorch
- **Model Architecture**: LSTM-based neural network
- **GUI**: PyQt5
- **Language**: Python 3.7+

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.6 or higher
- PyQt5

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NER_Pytorch.git
   cd NER_Pytorch
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   *Note: If requirements.txt is not available, install dependencies manually:*
   ```bash
   pip install torch torchvision torchaudio
   pip install PyQt5
   ```

## Usage

### 1. Data Preprocessing

```bash
python 01_data_preprocess.py
```

This script processes raw data into the required format for model training.

### 2. Model Training

```bash
python 02_train.py
```

Or use the provided PowerShell script:
```powershell
./run_training.ps1
```

Trained models and parameters will be saved in the `runs` directory.

### 3. Model Validation

```bash
python 03_val.py
```

This script evaluates the trained model's performance on the validation set.

### 4. Inference

```bash
python 04_infer.py
```

This script provides a command-line interface for entity recognition.

### 5. Graphical Interface

```bash
python 05_pyqt_interface.py
```

Launch the user-friendly GUI for interactive entity recognition.

## Project Structure

```
NER_Pytorch/
├── 01_data_preprocess.py    # Data preprocessing script
├── 02_train.py              # Model training script
├── 03_val.py                # Model validation script
├── 04_infer.py              # Inference script
├── 05_pyqt_interface.py     # PyQt GUI interface
├── model.py                 # Neural network model definition
├── loss_related_code.py     # Loss function implementation
├── run_training.ps1         # Training PowerShell script
├── dataset/                 # Processed dataset files
├── NER命名实体识别数据集/     # Raw dataset files
└── runs/                    # Model weights and parameters
    ├── best_model.pth       # Best trained model
    ├── loss_curve.png       # Training loss curve
    ├── model_params.json    # Model parameters
    ├── tag_map.json         # Entity tag mapping
    └── vocab.json           # Vocabulary dictionary
```

## Model Details

The NER model is based on a bidirectional LSTM architecture with the following key components:

- Embedding layer for character representation
- Bidirectional LSTM layers for contextual feature extraction
- Linear layer with CRF (Conditional Random Field) for sequence labeling

## Performance

The model achieves high accuracy in recognizing various types of named entities in Chinese text, including:
- Person names
- Locations
- Organizations
- And other custom entity types

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is built using PyTorch, which provides excellent support for deep learning research and development.
- The CRF implementation is adapted from state-of-the-art NER research.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For any questions or inquiries, please open an issue on GitHub.
