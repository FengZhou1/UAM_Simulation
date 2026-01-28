import torch
import sys

def test_gpu():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        try:
            # Test tensor creation on GPU
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"Successfully created tensor on GPU: {x}")
            print(f"Tensor device: {x.device}")
            
            # Test simple operation
            y = x + x
            print(f"GPU computation result: {y}")
            
        except Exception as e:
            print(f"Error using GPU: {e}")
    else:
        print("CUDA is NOT available. PyTorch will use CPU.")

if __name__ == "__main__":
    test_gpu()
