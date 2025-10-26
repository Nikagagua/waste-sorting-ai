#!/bin/bash

# Training Monitor Script
# Tracks progress of model training in real-time

LOG_FILE=$(ls -t training_improved_final_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No training log file found!"
    exit 1
fi

echo "======================================"
echo "WASTE SORTING AI - TRAINING MONITOR"
echo "======================================"
echo "Log file: $LOG_FILE"
echo "Started: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LOG_FILE")"
echo ""

# Check if training is still running
if pgrep -f "model_comparison.py" > /dev/null; then
    echo "Status: ✅ TRAINING IN PROGRESS"
else
    echo "Status: ⚠️ NOT RUNNING (completed or crashed)"
fi

echo ""
echo "======================================"
echo "LATEST METRICS"
echo "======================================"

# Extract latest accuracy metrics
echo "Recent Validation Accuracies:"
grep "val_accuracy:" "$LOG_FILE" | tail -5

echo ""
echo "Model Progress:"
grep -E "(Training ResNet|Training EfficientNet|Training DenseNet|Training MobileNet)" "$LOG_FILE" | tail -4

echo ""
echo "======================================"
echo "ERROR CHECK"
echo "======================================"
if grep -i "error\|exception\|failed" "$LOG_FILE" | tail -3; then
    echo "⚠️ Errors detected - check log file"
else
    echo "✅ No recent errors"
fi

echo ""
echo "======================================"
echo "TAIL OF LOG (last 20 lines)"
echo "======================================"
tail -20 "$LOG_FILE"

echo ""
echo "To follow live: tail -f $LOG_FILE"
