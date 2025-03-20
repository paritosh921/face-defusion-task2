import os
import argparse
import torch
import time
from pathlib import Path
import subprocess

def run_pipeline(args):
    start_time = time.time()
    
    # Step 1: Fine-tune the encoder model
    print("="*50)
    print("STEP 1: Fine-tuning DINOv2 encoder")
    print("="*50)
    
    encoder_output_dir = os.path.join(args.output_dir, "encoder")
    os.makedirs(encoder_output_dir, exist_ok=True)
    
    encoder_cmd = [
        "python", "-m", "finetune_encoder",
        "--data_dir", args.data_dir,
        "--output_dir", encoder_output_dir,
        "--batch_size", str(args.encoder_batch_size),
        "--lr", str(args.encoder_lr),
        "--epochs", str(args.encoder_epochs)
    ]
    
    subprocess.run(encoder_cmd)
    
    # Check if fine-tuning was successful
    encoder_checkpoint = os.path.join(encoder_output_dir, "best_model.pth")
    if not os.path.exists(encoder_checkpoint):
        print("Error: Encoder fine-tuning failed to produce a checkpoint")
        return
    
    # Step 2: Train the diffusion model
    print("\n" + "="*50)
    print("STEP 2: Training conditional diffusion model")
    print("="*50)
    
    diffusion_output_dir = os.path.join(args.output_dir, "diffusion")
    os.makedirs(diffusion_output_dir, exist_ok=True)
    
    diffusion_cmd = [
        "python", "-m", "diffusion_generator",
        "--data_dir", args.data_dir,
        "--embedding_model_path", encoder_checkpoint,
        "--output_dir", diffusion_output_dir,
        "--batch_size", str(args.diffusion_batch_size),
        "--lr", str(args.diffusion_lr),
        "--epochs", str(args.diffusion_epochs)
    ]
    
    subprocess.run(diffusion_cmd)
    
    # Step 3: Evaluate zero-shot performance if test directory is provided
    if args.test_dir:
        print("\n" + "="*50)
        print("STEP 3: Evaluating zero-shot performance")
        print("="*50)
        
        test_output_dir = os.path.join(args.output_dir, "test_results")
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Load best model
        diffusion_checkpoint = os.path.join(diffusion_output_dir, "best_model.pth")
        
        evaluation_cmd = [
            "python", "-m", "diffusion_generator",
            "--data_dir", args.data_dir,
            "--embedding_model_path", encoder_checkpoint,
            "--output_dir", diffusion_output_dir,
            "--test",
            "--test_dir", args.test_dir,
            "--test_output_dir", test_output_dir
        ]
        
        subprocess.run(evaluation_cmd)
    
    # Calculate total runtime
    total_hours = (time.time() - start_time) / 3600
    print(f"\nTotal pipeline runtime: {total_hours:.2f} hours")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the face generation pipeline")
    
    # Main arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with face dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--test_dir", type=str, help="Directory with test images")
    
    # Encoder hyperparameters
    parser.add_argument("--encoder_batch_size", type=int, default=32, help="Batch size for encoder training")
    parser.add_argument("--encoder_lr", type=float, default=1e-4, help="Learning rate for encoder training")
    parser.add_argument("--encoder_epochs", type=int, default=10, help="Number of epochs for encoder training")
    
    # Diffusion model hyperparameters
    parser.add_argument("--diffusion_batch_size", type=int, default=16, help="Batch size for diffusion model training")
    parser.add_argument("--diffusion_lr", type=float, default=1e-4, help="Learning rate for diffusion model training")
    parser.add_argument("--diffusion_epochs", type=int, default=100, help="Number of epochs for diffusion model training")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the pipeline
    run_pipeline(args)