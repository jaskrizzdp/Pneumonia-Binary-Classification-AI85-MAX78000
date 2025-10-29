#!/bin/sh
# Device and target folder
DEVICE="MAX78000"
TARGET="D:/KRIZZ/MaximSDK/Examples/MAX78000/CNN"

# Common arguments for ai8xize.py
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

# Run synthesis
python ai8xize.py \
    --test-dir $TARGET \
    --prefix pneumonia \
    --checkpoint-file D:/KRIZZ/COE187/ai8x-synthesis/trained/ai85-pneumonia-qat8-q.pth.tar \
    --config-file networks/pneumonia-hwc.yaml \
    --fifo \
    --softmax \
    $COMMON_ARGS "$@"
