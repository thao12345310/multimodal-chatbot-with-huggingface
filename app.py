import torch, mlx.core as mx

# Kiểm tra MPS (PyTorch)
print(torch.backends.mps.is_available())   # → True
print(torch.backends.mps.is_built())       # → True

# Kiểm tra MLX
print(mx.default_device())                 # → Device(gpu, 0)

# Kiểm tra RAM
import subprocess
result = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True)
ram_gb = int(result.stdout.split()[-1]) / 1e9
print(f"RAM: {ram_gb:.0f}GB")             # → RAM: 16GB