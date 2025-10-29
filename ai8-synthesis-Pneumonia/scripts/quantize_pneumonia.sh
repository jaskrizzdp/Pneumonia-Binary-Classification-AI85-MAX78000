# #!/bin/sh

# LOG_DIRECTORY="D:/KRIZZ/COE187/ai8x-training/logs/2025.10.28-205603"

# python ../quantize.py "$LOG_DIRECTORY/best.pth.tar" ../trained/ai85-catsdogs-qat8-q.pth.tar --device MAX78000 -v "$@"


#!/bin/sh

# Path to your training log folder (from ai8x-training)
LOG_DIRECTORY="D:/KRIZZ/COE187/ai8x-training/logs/2025.10.29-033749"

# Run quantization
python quantize.py "$LOG_DIRECTORY/best.pth.tar" trained/ai85-pneumonia-qat8-q.pth.tar --device MAX78000 -v "$@"
