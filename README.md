# Face Generation from Embeddings

This project implements a conditional generative model that creates human face images from embeddings. The pipeline consists of two main components:

1. **Fine-tuning a DINOv2 encoder** on a face dataset to create high-quality face embeddings
2. **Training a conditional diffusion model** that generates face images from these embeddings

## Implementation Details

### Encoder Fine-tuning

We fine-tune a DINOv2 model (ViT-S/16) on our face dataset to create embeddings specifically optimized for face representation. Key implementation details:

- Use contrastive learning to improve the quality of face embeddings
- Freeze most of the model parameters and only train the last block to prevent overfitting and speed up training
- Apply data augmentation (horizontal flips, color jitter) to enhance robustness
- Implement a custom contrastive loss function to maximize embedding similarity for related faces

### Generative Model

We implement a conditional latent diffusion model for face generation. Key features:

- U-Net architecture with self-attention layers for better image quality
- Conditioning mechanism that effectively incorporates the DINOv2 embeddings
- Time-step and condition embeddings to control the denoising process
- Classifier-free guidance for improved generation quality

### Edge Cases and Design Decisions

1. **Training Time Management**: We implement automatic checkpointing and early stopping to ensure the training completes within the 6-hour limit. Resources are allocated with approximately 1.5 hours for encoder fine-tuning and 4.5 hours for diffusion model training.

2. **Batch Size Adaptation**: The code dynamically adjusts batch sizes based on available GPU memory to optimize training speed.

3. **Embedding Caching**: To avoid redundant computations, embeddings are pre-computed and cached after the encoder fine-tuning.

4. **Zero-shot Evaluation**: We implement a dedicated evaluation pipeline for testing the model's generalization capabilities on unseen face images.

5. **Handling Unusual Faces**: The model is trained with diverse face images to handle variations in pose, expression, and lighting. During inference, classifier-free guidance helps generate plausible faces even from embeddings of unusual face images.

## Usage

### Setup

```bash
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
python run_pipeline.py --data_dir /path/to/face/dataset --output_dir ./output --test_dir /path/to/test/images
```

### Fine-tuning the Encoder Only

```bash
python finetune_encoder.py --data_dir /path/to/face/dataset --output_dir ./encoder_output
```

### Training the Diffusion Model Only

```bash
python diffusion_generator.py --data_dir /path/to/face/dataset --embedding_model_path ./encoder_output/best_model.pth --output_dir ./diffusion_output
```

### Zero-shot Evaluation

```bash
python diffusion_generator.py --data_dir /path/to/face/dataset --embedding_model_path ./encoder_output/best_model.pth --output_dir ./diffusion_output --test --test_dir /path/to/test/images --test_output_dir ./test_results
```

## Results

The model achieves strong zero-shot generalization capabilities, accurately generating faces from unseen embeddings with high fidelity. Training converges within the 6-hour limit by optimizing both the encoder fine-tuning and diffusion model training phases.