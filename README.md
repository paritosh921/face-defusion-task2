# Diffusion-Based Face Generation

This project implements a conditional diffusion model for generating human face images. The model uses a Vision Transformer (ViT) encoder (pretrained with DINOv2 weights) to condition a U-Net-based diffusion model that progressively denoises images. The code also logs training loss and saves a loss curve for performance tracking.


## Sample Image Note

The following sample image was generated after just 5 epochs of training:

![samples_epoch_5 (1)](https://github.com/user-attachments/assets/51f59a92-787c-4df0-ab87-9a491109fb95){: width="400px"}



**Note:** The image may still appear noisy or blurry because the diffusion model has not yet converged. **Increasing the number of training epochs** (e.g., 50, 100, or more) and/or tuning hyperparameters (like the learning rate or U-Net architecture) can significantly improve the quality of generated images. Diffusion models often require extensive training time to capture detailed structures in the data.


## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Installation Using uv Package Manager](#installation-using-uv-package-manager)
- [Training the Model](#training-the-model)
- [Model Outputs](#model-outputs)
- [Additional Information](#additional-information)

## Overview

- **Encoder:**  
  A pretrained ViT model (using DINOv2 weights) is used to extract high-level embeddings from face images. The classification head is removed, and a projection layer is added to obtain fixed-dimension embeddings.

- **Diffusion Model:**  
  A conditional U-Net takes as input the noisy images along with time and conditioning embeddings. A beta schedule is used to progressively add noise during training, and the model is trained to reverse this process.

- **Training Metrics:**  
  Training losses are logged per epoch. At the end of training, a loss curve is saved (e.g., `training_loss.png`) that can be used to monitor the model's performance.

## Environment Setup

This project requires Python 3.8+ and the following libraries:
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm
- Pillow

## Installation Using uv Package Manager

We recommend using the **uv package manager** to create an isolated environment and install dependencies. Follow these steps:

1. **Install uv (if not already installed):**

   Follow the instructions on the [uv package manager GitHub page](https://github.com/uvpm/uvpm). For example:
   ```bash
   pip install uvpm
   ```

2. **Create a new environment:**
   ```bash
   uv create face-diffusion-env python=3.8
   ```

3. **Activate the environment:**
   ```bash
   uv activate face-diffusion-env
   ```

4. **Install required packages:**
   ```bash
   uv install torch torchvision numpy matplotlib tqdm Pillow
   ```

   Alternatively, if a `requirements.txt` is provided, you can install all dependencies with:
   ```bash
   uv install -r requirements.txt
   ```

## Training the Model

1. **Mount Google Drive (if using Colab):**  
   Ensure your Google Drive is mounted so that the dataset and model checkpoints can be accessed:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Set the dataset path:**  
   Update the `data_dir` variable in the training script to point to your face image dataset (e.g., `/content/drive/MyDrive/img_align_celeba`).

3. **Run the Training Script:**  
   Execute the training script:
   ```bash
   python train_face_generation.py
   ```
   The script will:
   - Load and preprocess images.
   - Train the diffusion model for a set number of epochs.
   - Log the training loss per epoch.
   - Save intermediate sample images every few epochs.
   - Save the final model weights in `final_model.pth` within the output directory.

## Model Outputs

After training, you will find:
- **Model Weights:**  
  The encoder and diffusion model weights are saved as `final_model.pth` in the specified output directory. Upload these weight files to Google Drive and share the link as required.

- **Training Loss Curve:**  
  A plot (`training_loss.png`) showing the average training loss per epoch is saved in the output directory. This plot can be used to evaluate training progress.

- **Sample Images:**  
  Generated samples from the diffusion model are saved periodically (e.g., `![samples_epoch_5 (1)](https://github.com/user-attachments/assets/51f59a92-787c-4df0-ab87-9a491109fb95){: width="400px"}
`
  ) in the output directory.

## Additional Information

- **Report:**  
  A detailed report explaining the algorithm, its components (encoder, diffusion process, conditioning), and the training procedure is included in the repository as `REPORT.md`.

- **Reproducibility:**  
  For reproducibility, please ensure you have installed the required dependencies using the uv package manager as described above.

- **GPU Usage:**  
  The model training script uses CUDA if available. Ensure that your environment is configured to use a GPU for faster training.
