#!/usr/bin/env python3
"""
Consolidated Attention Collection System
整合的Attention数据收集和分析系统

This system collects attention weights from transformer models for DSA algorithm analysis.
Features:
- Attention hook system for collecting weights
- Training with attention collection
- Comprehensive attention analysis
- Complete output without truncation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import os
import sys
import logging
import json
import glob
from datetime import datetime
import argparse


# ============================ TRANSFORMER MODEL WITH ATTENTION HOOKS ============================

class MultiheadAttentionWithHook(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.01, layer_idx=None):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.layer_idx = layer_idx

        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # For storing attention weights
        self.attention_weights = None
        self.collect_attention = False

    def forward(self, Q, K, V, mask=None):
        bsz, seq_len, _ = Q.size()
        q = self.wq(Q).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(K).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(V).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        # Store attention weights if collection is enabled
        if self.collect_attention:
            self.attention_weights = attn_weights.detach().cpu().numpy()

        attn_weights = self.dropout(attn_weights)

        output = (
            torch.matmul(attn_weights, v)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.embed_dim)
        )
        return self.fc(output)

    def enable_attention_collection(self):
        self.collect_attention = True

    def disable_attention_collection(self):
        self.collect_attention = False

    def get_attention_weights(self):
        return self.attention_weights


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.01):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.silu = nn.SiLU()
        self.w2 = nn.Linear(ff_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w1(x)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class TransformerBlockWithHook(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, layer_idx=None):
        super().__init__()
        self.attn = MultiheadAttentionWithHook(embed_dim, num_heads, dropout, layer_idx)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.layer_idx = layer_idx

    def forward(self, x, mask=None):
        h = self.norm1(x + self.attn(x, x, x, mask))
        return self.norm2(h + self.ff(h))

    def enable_attention_collection(self):
        self.attn.enable_attention_collection()

    def disable_attention_collection(self):
        self.attn.disable_attention_collection()

    def get_attention_weights(self):
        return self.attn.get_attention_weights()


class TransformerModelWithHook(nn.Module):
    def __init__(
        self,
        input_dim=3,
        embed_dim=64,
        num_heads=2,
        ff_dim=128,
        num_layers=1,
        dropout=0.0,
        output_dim=6,
        max_seq_len=1000,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # feature embedding
        self.feature_projection = nn.Linear(input_dim, embed_dim)
        self.norm_after_position = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlockWithHook(embed_dim, num_heads, ff_dim, dropout, i)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, output_dim)

        # For storing all attention data
        self.attention_data = []

    def make_padding_mask(self, x):
        return x.abs().sum(dim=-1) == 0

    def create_positional_encoding(self, seq_len, device):
        """Generate sinusoidal position embeddings for the given sequence length."""
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float()
            * -(math.log(10000.0) / self.embed_dim)
        ).to(device)

        positional_encoding = torch.zeros(seq_len, self.embed_dim, device=device)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        return positional_encoding.unsqueeze(0)

    def enable_attention_collection(self):
        """Enable attention collection for all layers"""
        for block in self.blocks:
            block.enable_attention_collection()

    def disable_attention_collection(self):
        """Disable attention collection for all layers"""
        for block in self.blocks:
            block.disable_attention_collection()

    def collect_attention_data(self, batch_idx=None):
        """Collect attention data from all layers"""
        batch_data = {
            'batch_idx': batch_idx,
            'layers': []
        }

        for i, block in enumerate(self.blocks):
            weights = block.get_attention_weights()
            if weights is not None:
                # Store weights as numpy array for easier access
                weights_np = weights if isinstance(weights, np.ndarray) else weights.cpu().numpy()
                layer_data = {
                    'layer_idx': block.layer_idx,
                    'attention_weights': weights_np.tolist(),
                    'shape': list(weights_np.shape)
                }
                batch_data['layers'].append(layer_data)

        self.attention_data.append(batch_data)

    def forward(self, x, collect_attention=False, batch_idx=None):
        # mask
        mask = self.make_padding_mask(x)

        # token embedding
        x = self.feature_projection(x)

        # position embedding (time information)
        seq_len = x.size(1)
        device = x.device
        position_embeddings = self.create_positional_encoding(seq_len, device)
        x += position_embeddings

        x = self.norm_after_position(x)

        # Enable attention collection if requested
        if collect_attention:
            self.enable_attention_collection()

        for block in self.blocks:
            x = block(x, mask=mask)

        # Collect attention data after forward pass
        if collect_attention:
            self.collect_attention_data(batch_idx)

        x = self.norm(x)
        not_pad = (~mask).float()
        x_valid = x * not_pad.unsqueeze(-1)
        x_pooled = x_valid.sum(dim=1) / (not_pad.sum(dim=1, keepdim=True) + 1e-8)
        output = self.fc_out(x_pooled)
        return output

    def save_attention_data(self, save_dir, filename_prefix=None):
        """Save collected attention data to files - COMPLETE OUTPUT WITHOUT TRUNCATION"""
        print(f"[DEBUG] save_attention_data called")
        print(f"[DEBUG] attention_data length: {len(self.attention_data)}")
        print(f"[DEBUG] save_dir: {save_dir}")

        if not self.attention_data:
            print("[ERROR] No attention data to save")
            return

        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"attention_{timestamp}"

        print(f"[DEBUG] Filename prefix: {filename_prefix}")
        os.makedirs(save_dir, exist_ok=True)

        saved_files = []
        # Save each layer's attention data separately
        for layer_idx in range(self.num_layers):
            layer_filename = os.path.join(save_dir, f"{filename_prefix}_layer_{layer_idx}.txt")
            print(f"[DEBUG] Saving layer {layer_idx} to {layer_filename}")

            with open(layer_filename, 'w') as f:
                f.write(f"Complete Attention Data for Layer {layer_idx}\n")
                f.write("=" * 60 + "\n\n")

                layer_has_data = False
                for batch_idx, batch_data in enumerate(self.attention_data):
                    if layer_idx < len(batch_data['layers']):
                        layer_data = batch_data['layers'][layer_idx]
                        weights_list = layer_data['attention_weights']
                        shape = layer_data['shape']

                        f.write(f"Batch {batch_idx}, Layer {layer_idx}\n")
                        f.write(f"Shape: {shape}\n")
                        f.write(f"Complete Attention Weights (No Truncation):\n")
                        layer_has_data = True

                        # Convert list back to numpy array for easier processing
                        weights = np.array(weights_list)

                        # Save attention weights in a readable format - COMPLETE OUTPUT
                        for head_idx in range(shape[1]):  # num_heads
                            f.write(f"\nHead {head_idx}:\n")
                            head_weights = weights[0][head_idx]  # First batch, this head

                            # Print statistics for this head
                            head_stats = {
                                'mean': float(np.mean(head_weights)),
                                'std': float(np.std(head_weights)),
                                'min': float(np.min(head_weights)),
                                'max': float(np.max(head_weights))
                            }
                            f.write(f"  Statistics: mean={head_stats['mean']:.6f}, std={head_stats['std']:.6f}, min={head_stats['min']:.6f}, max={head_stats['max']:.6f}\n")
                            f.write(f"  Complete Attention Matrix ({head_weights.shape[0]}x{head_weights.shape[1]}):\n")

                            # Output COMPLETE attention matrix - NO TRUNCATION
                            for i, row in enumerate(head_weights):
                                row_str = ' '.join([f'{w:.6f}' for w in row])
                                f.write(f"    Row {i:4d}: {row_str}\n")

                        f.write("\n" + "=" * 60 + "\n\n")

                if layer_has_data:
                    saved_files.append(layer_filename)
                    print(f"[SUCCESS] Layer {layer_idx} complete data saved to {layer_filename}")
                else:
                    print(f"[WARNING] No data found for layer {layer_idx}")

        # Also save metadata
        metadata_filename = os.path.join(save_dir, f"{filename_prefix}_metadata.json")
        metadata = {
            'total_batches': len(self.attention_data),
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'collection_time': datetime.now().isoformat(),
            'data_files': saved_files,
            'model_config': {
                'input_dim': getattr(self, 'input_dim', 'unknown'),
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers
            },
            'output_mode': 'complete_no_truncation'
        }

        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[SUCCESS] Complete attention data saved to {save_dir}")
        print(f"[SUCCESS] Metadata saved to {metadata_filename}")
        print(f"[SUCCESS] Saved {len(saved_files)} layer files")
        print(f"[INFO] Output mode: COMPLETE (no truncation)")

        # Clear attention data after saving
        self.attention_data.clear()
        print(f"[DEBUG] Attention data cleared after saving")


# ============================ DATASET HANDLING ============================

class DataLoader:
    def __init__(self, dataset_path, input_dim=5):
        self.dataset_path = dataset_path
        self.input_dim = input_dim
        self.data = None

    def load_dataset(self):
        """Load dataset from specified path"""
        if os.path.exists(self.dataset_path) and os.path.isdir(self.dataset_path):
            # Try to load actual dataset files
            pkl_files = glob.glob(os.path.join(self.dataset_path, "*.pt")) + glob.glob(os.path.join(self.dataset_path, "*.pkl"))

            if pkl_files:
                # Load real data
                data_file = pkl_files[0]
                data = torch.load(data_file)

                if isinstance(data, dict):
                    self.data = {
                        'x_data': data.get('x_data', data.get('features', torch.randn(50, 500, self.input_dim))),
                        'y_data': data.get('y_data', data.get('labels', torch.randn(50, 6)))
                    }
                else:
                    # If it's just a tensor, create dummy data with similar structure
                    self.data = {
                        'x_data': torch.randn(50, min(500, data.shape[0] if len(data.shape) > 0 else 500), self.input_dim),
                        'y_data': torch.randn(50, 6)
                    }
            else:
                self._create_dummy_data()
        else:
            self._create_dummy_data()

        return self.data

    def _create_dummy_data(self):
        """Create dummy data for testing"""
        seq_len = 500  # Reasonable sequence length for complete output

        self.data = {
            'x_data': torch.randn(50, seq_len, self.input_dim),
            'y_data': torch.randn(50, 6)
        }


# ============================ ATTENTION COLLECTION AND ANALYSIS ============================

class AttentionCollector:
    def __init__(self, config, data_dir="./data_attention", log_dir="./log"):
        self.config = config
        self.data_dir = data_dir
        self.log_dir = log_dir

        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Initialize model
        self.model = TransformerModelWithHook(
            input_dim=config['input_dim'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            ff_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=6,
            dropout=0.0
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Load dataset
        self.data_loader = DataLoader(config['dataset_path'], config['input_dim'])
        self.train_data = self.data_loader.load_dataset()

        self.logger.info(f"AttentionCollector initialized with device: {self.device}")
        self.logger.info(f"Model parameters: {config}")
        self.logger.info(f"Dataset loaded: {len(self.train_data['x_data'])} samples")

    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.log_dir, f"attention_collection_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_filename}")

    def collect_attention_samples(self, num_samples=10):
        """Collect attention weights from sample inputs - COMPLETE OUTPUT"""
        self.logger.info(f"Starting complete attention collection for {num_samples} samples")
        self.logger.info("Output mode: COMPLETE (no truncation)")

        self.model.eval()

        sample_indices = np.random.choice(len(self.train_data['x_data']),
                                       min(num_samples, len(self.train_data['x_data'])),
                                       replace=False)

        for i, idx in enumerate(sample_indices):
            try:
                # Get sample data
                x_sample = self.train_data['x_data'][idx:idx+1].to(self.device)
                self.logger.info(f"Processing sample {i+1}/{num_samples}, index: {idx}")

                # Forward pass with attention collection
                with torch.no_grad():
                    output = self.model(x_sample, collect_attention=True, batch_idx=i)

                # Save attention data after each sample
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_prefix = f"sample_{i+1:03d}_idx_{idx}_{timestamp}"

                self.model.save_attention_data(self.data_dir, filename_prefix)

                self.logger.info(f"Sample {i+1} completed and saved")

            except Exception as e:
                self.logger.error(f"Error processing sample {i+1}: {e}")
                continue

        self.logger.info("Complete attention collection finished")

    def run_training_with_attention(self, num_epochs=5, collect_every_n_batches=5):
        """Run training while collecting attention periodically - COMPLETE OUTPUT"""
        self.logger.info(f"Starting training with complete attention collection for {num_epochs} epochs")
        self.logger.info("Output mode: COMPLETE (no truncation)")

        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()

        batch_size = self.config['batch_size']
        dataset_size = len(self.train_data['x_data'])

        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")

            # Shuffle data
            indices = torch.randperm(dataset_size)

            epoch_loss = 0.0
            num_batches = 0

            for batch_start in range(0, dataset_size, batch_size):
                batch_end = min(batch_start + batch_size, dataset_size)
                batch_indices = indices[batch_start:batch_end]

                # Get batch data
                x_batch = self.train_data['x_data'][batch_indices].to(self.device)
                y_batch = self.train_data['y_data'][batch_indices].to(self.device)

                # Training step
                self.model.train()
                optimizer.zero_grad()

                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Collect complete attention every n batches
                if (num_batches % collect_every_n_batches == 0):
                    self.logger.info(f"Collecting complete attention at batch {num_batches}")

                    # Switch to eval mode for attention collection
                    self.model.eval()
                    with torch.no_grad():
                        # Collect complete attention from first sample in batch
                        sample_x = x_batch[:1]  # First sample in batch
                        _ = self.model(sample_x, collect_attention=True,
                                     batch_idx=f"epoch{epoch+1}_batch{num_batches}")

                        # Save complete attention data
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename_prefix = f"epoch{epoch+1:02d}_batch{num_batches:04d}_{timestamp}"
                        self.model.save_attention_data(self.data_dir, filename_prefix)

                    # Switch back to train mode
                    self.model.train()

            avg_loss = epoch_loss / num_batches
            self.logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}")

        self.logger.info("Training with complete attention collection completed")

    def save_model(self, save_path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }, save_path)
        self.logger.info(f"Model saved to {save_path}")


class AttentionAnalyzer:
    def __init__(self, data_dir="./data_attention"):
        self.data_dir = data_dir
        self.attention_data = {}
        self.metadata = {}

    def load_attention_data(self):
        """Load all attention data files from the data directory"""
        print(f"Loading complete attention data from {self.data_dir}")

        # Find all metadata files
        metadata_files = glob.glob(os.path.join(self.data_dir, "*_metadata.json"))

        for metadata_file in metadata_files:
            base_name = os.path.basename(metadata_file).replace("_metadata.json", "")

            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.metadata[base_name] = metadata
            self.attention_data[base_name] = {}

            # Load layer data
            for layer_file in metadata['data_files']:
                layer_path = os.path.join(self.data_dir, layer_file)
                if os.path.exists(layer_path):
                    layer_idx = layer_file.split('_layer_')[1].split('.')[0]
                    self.attention_data[base_name][layer_idx] = self._parse_attention_file(layer_path)

        print(f"Loaded {len(self.attention_data)} complete attention datasets")

    def _parse_attention_file(self, file_path):
        """Parse attention weights from text file"""
        layers_data = []

        with open(file_path, 'r') as f:
            content = f.read()

        # Split by batch
        batches = content.split("Batch ")[1:]  # Skip header

        for batch_content in batches:
            lines = batch_content.strip().split('\n')

            # Find attention weights section
            attention_start = -1
            for i, line in enumerate(lines):
                if "Complete Attention Weights" in line:
                    attention_start = i + 2  # Skip header line
                    break

            if attention_start == -1:
                continue

            # Parse complete attention weights for each head
            head_data = []
            current_head = []

            for line in lines[attention_start:]:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("Head "):
                    if current_head:  # Save previous head if exists
                        head_data.append(current_head)
                    current_head = []
                elif line.startswith("Row ") or line.startswith("    Row "):  # Handle complete output format
                    # Parse numeric values
                    try:
                        # Extract row values from "Row N: ..." format
                        if ":" in line:
                            row_part = line.split(":", 1)[1].strip()
                        else:
                            row_part = line.strip()

                        values = [float(x) for x in row_part.split()]
                        current_head.append(values)
                    except ValueError:
                        continue

            if current_head:
                head_data.append(current_head)

            if head_data:
                layers_data.append(np.array(head_data))

        return layers_data

    def analyze_complete_attention_patterns(self):
        """Analyze patterns in the collected complete attention data"""
        print("Analyzing complete attention patterns...")

        analysis_results = {}

        for dataset_name, layers_data in self.attention_data.items():
            print(f"\nAnalyzing dataset: {dataset_name}")

            dataset_analysis = {
                'num_layers': len(layers_data),
                'layer_stats': {}
            }

            for layer_idx, batches_data in layers_data.items():
                if not batches_data:
                    continue

                print(f"  Layer {layer_idx}: {len(batches_data)} batches")

                # Combine all batches for this layer
                all_attention = np.concatenate(batches_data, axis=0)

                layer_stats = {
                    'num_heads': all_attention.shape[0],
                    'seq_length': all_attention.shape[1],
                    'mean_attention': np.mean(all_attention),
                    'std_attention': np.std(all_attention),
                    'max_attention': np.max(all_attention),
                    'min_attention': np.min(all_attention),
                    'sparsity': np.mean(all_attention < 0.01),  # Percentage of very low attention
                }

                # Head-specific statistics
                layer_stats['head_stats'] = []
                for head_idx in range(all_attention.shape[0]):
                    head_attention = all_attention[head_idx]
                    head_stats = {
                        'head_idx': head_idx,
                        'mean': np.mean(head_attention),
                        'std': np.std(head_attention),
                        'entropy': self._calculate_entropy(head_attention)
                    }
                    layer_stats['head_stats'].append(head_stat)

                dataset_analysis['layer_stats'][layer_idx] = layer_stats

            analysis_results[dataset_name] = dataset_analysis

        return analysis_results

    def _calculate_entropy(self, attention_matrix):
        """Calculate entropy of attention distribution"""
        # Normalize to probability distribution
        attention_flat = attention_matrix.flatten()
        attention_norm = attention_flat / np.sum(attention_flat)

        # Remove zeros to avoid log(0)
        attention_norm = attention_norm[attention_norm > 0]

        # Calculate entropy
        entropy = -np.sum(attention_norm * np.log2(attention_norm + 1e-8))
        return entropy

    def generate_complete_analysis_report(self, analysis_results, output_file=None):
        """Generate a summary report of complete attention analysis"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.data_dir, f"attention_complete_analysis_{timestamp}.txt")

        with open(output_file, 'w') as f:
            f.write("Complete Attention Analysis Summary Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Output Mode: COMPLETE (No Truncation)\n\n")

            for dataset_name, dataset_analysis in analysis_results.items():
                f.write(f"Dataset: {dataset_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Number of layers: {dataset_analysis['num_layers']}\n\n")

                for layer_idx, layer_stats in dataset_analysis['layer_stats'].items():
                    f.write(f"  Layer {layer_idx}:\n")
                    f.write(f"    Number of heads: {layer_stats['num_heads']}\n")
                    f.write(f"    Sequence length: {layer_stats['seq_length']}\n")
                    f.write(f"    Mean attention: {layer_stats['mean_attention']:.6f}\n")
                    f.write(f"    Std attention: {layer_stats['std_attention']:.6f}\n")
                    f.write(f"    Max attention: {layer_stats['max_attention']:.6f}\n")
                    f.write(f"    Min attention: {layer_stats['min_attention']:.6f}\n")
                    f.write(f"    Sparsity (<0.01): {layer_stats['sparsity']:.2%}\n")

                    f.write(f"    Head statistics (Complete Data):\n")
                    for head_stat in layer_stats['head_stats']:
                        f.write(f"      Head {head_stat['head_idx']}: "
                               f"mean={head_stat['mean']:.4f}, "
                               f"std={head_stat['std']:.4f}, "
                               f"entropy={head_stat['entropy']:.4f}\n")
                    f.write("\n")

                f.write("\n" + "=" * 60 + "\n\n")

        print(f"Complete analysis report saved to: {output_file}")
        return output_file


# ============================ MAIN FUNCTION ============================

def main():
    """Main function to run complete attention collection system"""

    # Parse command line arguments (optional - for backward compatibility)
    parser = argparse.ArgumentParser(description='Complete Attention Collection System')
    parser.add_argument('--mode', type=str, default='collect',
                       choices=['collect', 'train', 'analyze', 'all'],
                       help='Operation mode: collect, train, analyze, or all')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to collect')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--data_dir', type=str, default='./data_attention', help='Data directory')
    parser.add_argument('--log_dir', type=str, default='./log', help='Log directory')
    args = parser.parse_args()

    # Configuration
    config = {
        'input_dim': 5,
        'embed_dim': 64,
        'num_heads': 2,
        'num_layers': 1,
        'hidden_dim': 128,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'dataset_path': '../sample/dataset',
        'data_dir': args.data_dir,
        'log_dir': args.log_dir
    }

    print("=" * 70)
    print("COMPLETE ATTENTION COLLECTION SYSTEM")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Output Mode: COMPLETE (No Truncation)")
    print(f"Samples: {args.samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Log Directory: {args.log_dir}")
    print("=" * 70)

    try:
        # Initialize attention collector
        collector = AttentionCollector(config)

        if args.mode == 'collect' or args.mode == 'all':
            # Collect attention from samples
            print("\n" + "=" * 50)
            print("COLLECTING COMPLETE ATTENTION DATA")
            print("=" * 50)
            collector.collect_attention_samples(num_samples=args.samples)

        if args.mode == 'train' or args.mode == 'all':
            # Run training with attention collection
            print("\n" + "=" * 50)
            print("TRAINING WITH COMPLETE ATTENTION COLLECTION")
            print("=" * 50)
            collector.run_training_with_attention(num_epochs=args.epochs, collect_every_n_batches=5)

            # Save the model
            model_save_path = os.path.join(args.data_dir, f"attention_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            collector.save_model(model_save_path)

        if args.mode == 'analyze' or args.mode == 'all':
            # Analyze collected attention data
            print("\n" + "=" * 50)
            print("ANALYZING COMPLETE ATTENTION DATA")
            print("=" * 50)
            analyzer = AttentionAnalyzer(args.data_dir)
            analyzer.load_attention_data()

            if analyzer.attention_data:
                analysis_results = analyzer.analyze_complete_attention_patterns()
                analyzer.generate_complete_analysis_report(analysis_results)
            else:
                print("No attention data found for analysis. Run collection first.")

        print("\n" + "=" * 70)
        print("COMPLETE ATTENTION COLLECTION SYSTEM FINISHED")
        print("=" * 70)
        print("All attention data saved without truncation in:", args.data_dir)
        print("Ready for DSA algorithm analysis!")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: Complete attention collection system failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()