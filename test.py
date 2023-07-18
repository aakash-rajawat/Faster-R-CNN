import torch

if torch.cuda.is_available():
    print("I ts is available")

else:
    print("It is not available")
