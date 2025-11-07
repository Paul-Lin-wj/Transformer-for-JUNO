# Transformer with DSA (Dynamic Sparse Attention)

This implementation integrates the DSA algorithm into the existing Transformer model with minimal code changes.

## Features

1. **DSA Integration**: Optional Dynamic Sparse Attention for improved efficiency
2. **Configurable Parameters**: All parameters configurable in scripts, no CLI args needed
3. **Checkpoint Resuming**: Resume training from last saved checkpoint
4. **Periodic Saving**: Save model every N epochs (configurable)
5. **Timestamped Outputs**: Models saved in timestamped folders
6. **Multi-GPU Support**: Automatic multi-GPU training with accelerate
7. **Structured Logging**: Logs saved in dedicated directories
8. **Original Compatibility**: Original model and training method still works

## File Structure

```
test/
├── python/
│   ├── dsa_algorithm.py          # DSA algorithm implementation
│   ├── trainer_dsa.py            # DSA-enhanced trainer
│   ├── ModelTrain.py             # Modified to support DSA
│   ├── RunModule.py              # Modified to support DSA arguments
│   ├── TensorPre/
│   │   ├── MLModel.py            # Added TransformerWithDSA class
│   │   └── TensorDataset.py      # Unchanged
│   └── [other original files]    # Copied from original
├── example/
│   ├── run_training.sh           # Shell script with all parameters
│   └── train_dsa.py              # Simplified Python training script
│   ├── model/                    # Model save directory (created automatically)
│   └── log/                      # Log directory (created automatically)
└── README.md                     # This file
```

## Usage

### Option 1: Using the Python Script (Recommended)

1. Edit the configuration in `example/train_dsa.py`
2. Run the script:

```bash
cd example
python train_dsa.py
```

### Option 2: Using the Shell Script

1. Edit the parameters at the top of `example/run_training.sh`
2. Make it executable (already done):
```bash
chmod +x example/run_training.sh
```
3. Run the script:
```bash
cd example
./run_training.sh
```

### Option 3: Using the Original Method

You can still use the original training method with command-line arguments:

```bash
cd python
python RunModule.py \
    --TrainModel \
    --mission_name "test" \
    --pre_method "TensorPre" \
    --train_model_name "Transformer" \
    --pklfile_train_path "/path/to/dataset" \
    --embed_dim 64 \
    --num_heads 8 \
    --num_layers 4 \
    --hidden_dim 128 \
    --input_dim 5 \
    --output_dim 6 \
    --num_epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --dsa_enabled \
    --sparsity_ratio 0.1 \
    --target_sparsity 0.05
```

## Key Parameters

### DSA Parameters
- `dsa_enabled`: Enable/disable DSA (true/false)
- `sparsity_ratio`: Initial sparsity ratio (0.1 = 10% connections kept)
- `target_sparsity`: Target sparsity ratio (0.05 = 5% connections kept)
- `sparsity_weight`: Sparsity regularization weight
- `entropy_weight`: Entropy regularization weight

### Training Parameters
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate
- `save_every`: Save model every N epochs
- `resume_training`: Resume from last checkpoint (true/false)

### Model Parameters
- `input_dim`: Input feature dimension (5 for muon track data)
- `embed_dim`: Transformer embedding dimension
- `num_heads`: Number of attention heads
- `num_layers`: Number of transformer layers
- `output_dim`: Output dimension (6 for track coordinates)

## Data

The model expects data in the same format as the original transformer:
- Dataset path: `/data/juno/lin/JUNO/transformer/muon_track_reco_transformer/sample/dataset`
- Format: `.pt` files created by the original data creation script

## Output

### Models
- Saved in `example/model/[timestamp]/`
- Files:
  - `model_best.pth`: Best model (lowest validation loss)
  - `checkpoint_latest.pth`: Latest checkpoint for resuming
  - `model_epoch_N.pth`: Checkpoint from epoch N
  - `config.json`: Training configuration
  - `training_history.json`: Training history

### Logs
- Saved in `example/log/`
- Format: `training_YYYYMMDD_HHMMSS.log`
- Contains detailed training progress and DSA statistics

## DSA Algorithm

DSA (Dynamic Sparse Attention) works by:
1. Computing standard attention weights
2. Selecting top-k most important connections
3. Dynamically adjusting sparsity during training
4. Maintaining model performance while reducing computation

## Notes

1. **Minimal Changes**: Original code is preserved, DSA is additive
2. **Backward Compatible**: Can disable DSA to use standard transformer
3. **Memory Efficient**: DSA reduces memory usage for large sequences
4. **Adaptive**: Sparsity adapts based on training progress

## Requirements

- PyTorch
- accelerate (for multi-GPU training)
- numpy
- Original transformer dependencies

## Testing

To test the implementation:

1. Ensure dataset is available at the specified path
2. Start with a small number of epochs to verify everything works
3. Check the log files for DSA statistics
4. Monitor training loss to ensure convergence

Example test command:
```bash
cd example
# Edit train_dsa.py to set num_epochs=2 for quick testing
python train_dsa.py
```