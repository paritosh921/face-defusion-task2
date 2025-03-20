import os
import torch
import sys

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import torchvision
        import tqdm
        import numpy
        import PIL
        import matplotlib
        print("All dependencies successfully imported!")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s).")
            print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. The training will run on CPU, which will be very slow.")
            print("Consider using a machine with a GPU for this project.")
        
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all dependencies using: pip install -r requirements.txt")
        return False

def setup_directories():
    """Create necessary directories for the project"""
    # Create output directory
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/encoder", exist_ok=True)
    os.makedirs("output/diffusion", exist_ok=True)
    os.makedirs("output/test_results", exist_ok=True)
    
    # Create data directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    
    print("Created all necessary directories:")
    print("- output/: for model outputs")
    print("- data/: for training, validation, and test data")
    print("  - Add face images to data/train/ and data/val/ for training")
    print("  - Add test images to data/test/ for evaluation")

if __name__ == "__main__":
    print("Checking environment and setting up directories for Face Generation project...")
    if check_dependencies():
        setup_directories()
        print("\nSetup complete!")
        print("\nTo run the full pipeline, use:")
        print("python run_pipeline.py --data_dir ./data --output_dir ./output --test_dir ./data/test")
    else:
        sys.exit(1) 