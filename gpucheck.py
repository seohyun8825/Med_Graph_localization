import torch

# GPU 사용 가능 여부 확인
print("CUDA available:", torch.cuda.is_available())

# 사용 가능한 GPU 수 확인
print("Number of GPUs:", torch.cuda.device_count())

# 현재 사용 중인 GPU 이름
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available.")
