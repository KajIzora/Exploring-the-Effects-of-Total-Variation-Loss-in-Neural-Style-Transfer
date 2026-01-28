# Fast Neural Style Transfer

A PyTorch implementation of fast neural style transfer that compares training with and without Total Variation (TV) loss regularization. This project implements a feed-forward neural network that can stylize images in real-time after training.

## Overview

This project implements the fast neural style transfer algorithm, which trains a feed-forward convolutional neural network (TransformerNet) to apply artistic styles to images. The model uses VGG16 as a feature extractor to compute content and style losses during training.

**Key Features:**
- Fast feed-forward style transfer network
- Comparison between models trained with and without Total Variation loss
- Comprehensive analysis and visualization tools
- Support for MPS (Metal Performance Shaders) on Apple Silicon

## Project Structure

```
.
├── models.py                      # Neural network architectures (VGG16, TransformerNet)
├── utils.py                       # Utility functions (transforms, gram matrix, etc.)
├── neural_style-transfer.ipynb    # Main training notebook
├── comp3-TVvsNOTV.ipynb          # Analysis and comparison notebook
├── data_processing.ipynb          # Data preparation notebook
├── data/
│   ├── train2014/                 # Original COCO training images
│   ├── train2014_40k/             # Subset of 40k images for training
│   ├── stylized/                  # Style reference images
│   └── comp3-TVvsNOTV/           # Training outputs and comparisons
│       ├── output_40k_5epochs_tv/     # Model trained with TV loss
│       ├── output_40k_5epochs_notv/   # Model trained without TV loss
│       └── comps/                     # Comparison visualizations
├── README.md
└── Exploring-the-Effects-of-Total-Variation-Loss-in-Neural-Style-Transfer.pdf # In depth discussion of results 
```

## Architecture

### TransformerNet
A feed-forward convolutional neural network that transforms input images into stylized versions. The architecture consists of:
- Three downsampling convolutional blocks
- Five residual blocks
- Two upsampling convolutional blocks
- Instance normalization and reflection padding

### VGG16 Feature Extractor
A pre-trained VGG16 network used to extract features at multiple layers (relu1_2, relu2_2, relu3_3, relu4_3) for computing content and style losses.

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+ (with MPS support for Apple Silicon)
- torchvision
- NumPy
- Pillow (PIL)
- Matplotlib
- Pandas
- PyAV (for video frame extraction)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd final_code
```

2. Install dependencies:
```bash
pip install torch torchvision numpy pillow matplotlib pandas av
```

## Usage

### Data Preparation

1. Place your training images in `data/train2014/` (or modify the path in the notebook)
2. Run `data_processing.ipynb` to create a subset of images for training (optional)

### Training

1. Open `neural_style-transfer.ipynb`
2. Configure training parameters:
   - `DATASET_PATH`: Path to training images
   - `STYLE_PATH`: Path to style reference image
   - `EPOCHS`: Number of training epochs
   - `BATCH_SIZE`: Batch size for training
   - `IMAGE_SIZE`: Input image size (default: 256)
   - `LAMBDA_CONTENT`: Weight for content loss (default: 1e5)
   - `LAMBDA_STYLE`: Weight for style loss (default: 1e10)
   - `TV_ACTIVE`: Enable/disable Total Variation loss (True/False)
   - `TV_WEIGHT`: Weight for TV loss (default: 1e-6)

3. Run the training cells. The notebook will:
   - Load and preprocess the style image
   - Train the TransformerNet
   - Save checkpoints and sample outputs periodically
   - Save the final model weights

### Analysis and Comparison

Run `comp3-TVvsNOTV.ipynb` to:
- Plot loss curves for different training runs
- Compare individual loss components (content, style, TV)
- Visualize checkpoint progression
- Compare final models on test images
- Generate comparison grids

## Training Configuration

Default training parameters:
- **Epochs**: 5
- **Batch Size**: 12
- **Image Size**: 256×256
- **Learning Rate**: 1e-3
- **Content Loss Weight**: 1e5
- **Style Loss Weight**: 1e10
- **TV Loss Weight**: 1e-6 (when enabled)

## Loss Functions

The total loss is composed of:

1. **Content Loss**: MSE between VGG features of content and stylized images
2. **Style Loss**: MSE between Gram matrices of style and stylized images
3. **Total Variation Loss** (optional): Encourages spatial smoothness in the output

```
Total Loss = λ_content × Content Loss + λ_style × Style Loss + λ_tv × TV Loss
```

## Device Support

The code automatically detects and uses:
- **MPS** (Metal Performance Shaders) on Apple Silicon Macs
- **CUDA** on NVIDIA GPUs (if available)
- **CPU** as fallback

## Acknowledgments

This implementation is adapted from [Fast Neural Style Transfer](https://github.com/eriklindernoren/Fast-Neural-Style-Transfer) by Erik Linder-Norén.

The architecture is based on the paper:
- **Perceptual Losses for Real-Time Style Transfer and Super-Resolution** by Johnson et al. (ECCV 2016)

## License

This project is for educational purposes. Please refer to the original repository for licensing information.

## Notes

- The style image used in the experiments is "Paysanne Couchant" (1882)
- Training on 40k images for 5 epochs takes approximately 1-2 hours on Apple Silicon
- Checkpoints are saved every 500 iterations
- The model can stylize images in real-time after training

