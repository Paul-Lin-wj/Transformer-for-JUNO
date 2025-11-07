"""
=============================================
TRAINER WITH DSA (Dynamic Sparse Attention)
DSA增强的训练器
=============================================

This trainer provides:
1. DSA-enabled model training
2. Checkpoint saving and resuming
3. Periodic model saving
4. Multi-GPU support via accelerate
5. Comprehensive logging
6. DSA statistics tracking
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from datetime import datetime
import glob
import logging
import psutil
import platform
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch.nn.functional as F

# Import custom modules
from TensorPre.TensorDataset import *
from TensorPre.MLModel import TransformerWithDSA, TransformerModel
from dsa_algorithm import DSALoss
from Fuc import Fuc


class TrainerWithDSA:
    """
    Enhanced trainer with DSA support
    """

    def __init__(self, config):
        self.config = config
        # Initialize log_dir early
        self.log_dir = config['log_dir']
        self.setup_logging()  # Setup logging first
        # System info is now printed by shell script before Python process starts
        self.setup_directories()
        self.setup_device()
        self.load_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_loss()
        self.setup_checkpoint()

        # Print dataset information after all setup is complete
        self.print_dataset_info()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.dsa_stats_history = []

    def setup_directories(self):
        """Create necessary directories"""
        # Create timestamped directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Log directory
        self.log_dir = self.config['log_dir']
        os.makedirs(self.log_dir, exist_ok=True)

        # Model directory with timestamp
        self.model_save_dir = os.path.join(
            self.config['model_dir'],
            timestamp
        )
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(self.model_save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        self.logger.info(f"Model save directory: {self.model_save_dir}")
        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"Configuration saved to: {config_path}")

    def setup_logging(self):
        """Setup logging configuration"""
        # Create logger
        self.logger = logging.getLogger('DSA_Trainer')
        self.logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler
        log_file = os.path.join(
            self.log_dir,
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info("Logging initialized")
        self.logger.info(f"Log file: {log_file}")

    def setup_device(self):
        """Setup device"""
        # Determine device
        if self.config.get('num_gpus', 0) > 0 and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"Using device: {self.device}")
            self.logger.info(f"GPU count: {torch.cuda.device_count()}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU")

        # Set seed for reproducibility
        if self.config.get('seed', None) is not None:
            torch.manual_seed(self.config['seed'])
            np.random.seed(self.config['seed'])

        # For single GPU/CPU training, no need for accelerate
        self.use_accelerate = False

    def load_data(self):
        """Load and prepare data"""
        self.logger.info("Loading dataset...")

        try:
            # Load dataset
            fuc = Fuc()
            pkl_files_path = self.config['dataset_path']

            # 检查数据路径
            if not os.path.exists(pkl_files_path):
                raise FileNotFoundError(f"Dataset path does not exist: {pkl_files_path}")

            pkl_files = fuc.get_file_paths(pkl_files_path)

            if not pkl_files:
                raise ValueError(f"No .pt files found in: {pkl_files_path}")

            self.logger.info(f"Found {len(pkl_files)} .pt files in dataset path")

            # 计算总数据大小
            total_size = 0
            valid_files = []
            for file in pkl_files:
                if os.path.exists(file):
                    file_size = os.path.getsize(file) / (1024*1024)  # MB
                    total_size += file_size
                    valid_files.append(file)
                else:
                    self.logger.warning(f"File not found, skipping: {file}")

            pkl_files = valid_files
            self.logger.info(f"Total dataset size: {total_size:.2f} MB")

            # Limit number of files if specified
            if 'max_files' in self.config and self.config['max_files']:
                original_count = len(pkl_files)
                pkl_files = pkl_files[:self.config['max_files']]
                self.logger.info(f"Using first {len(pkl_files)} files (limited from {original_count})")

            # 创建数据集（CombinedDataset会打印详细的加载信息）
            self.logger.info("Creating combined dataset...")
            dataset = CombinedDataset(pkl_files)

            # 获取数据集详细信息
            dataset_info = dataset.get_dataset_info()
            self.logger.info(f"Dataset loading completed:")
            self.logger.info(f"  - Successfully loaded: {dataset_info['loaded_files']}/{dataset_info['total_files']} files")
            self.logger.info(f"  - Total samples: {dataset_info['total_samples']:,}")

            # 检查数据集是否为空
            if len(dataset) == 0:
                raise ValueError("Dataset is empty after loading!")

        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise

        # Split dataset
        train_size = int(self.config['train_ratio'] * len(dataset))
        val_size = len(dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.get('seed', 42))
        )

        self.logger.info(f"Train set size: {len(self.train_dataset)}")
        self.logger.info(f"Validation set size: {len(self.val_dataset)}")

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        # Store dataset info for later printing
        self.dataset_info = {
            'pkl_files_path': pkl_files_path,
            'pkl_files': pkl_files,
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset),
            'train_loader_len': len(self.train_loader),
            'val_loader_len': len(self.val_loader)
        }

    def setup_model(self):
        """Setup model with or without DSA"""
        # Model configuration
        model_config = {
            'input_dim': self.config['input_dim'],
            'embed_dim': self.config['embed_dim'],
            'num_heads': self.config['num_heads'],
            'ff_dim': self.config.get('ff_dim', self.config['embed_dim'] * 2),
            'num_layers': self.config['num_layers'],
            'dropout': self.config.get('dropout', 0.1),
            'output_dim': self.config['output_dim'],
            'max_seq_len': self.config.get('seq_len', 1000)
        }

        # Create model
        if self.config.get('dsa_enabled', False):
            self.logger.info("Initializing Transformer with DSA")
            dsa_config = self.config.get('dsa_config', {})
            dsa_config['enabled'] = True
            model_config['dsa_config'] = dsa_config
            self.model = TransformerWithDSA(**model_config)
        else:
            self.logger.info("Initializing standard Transformer")
            self.model = TransformerModel(**model_config)

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model initialized with {total_params:,} total parameters")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

    def setup_optimizer(self):
        """Setup optimizer"""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()

        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0)
            )
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        self.logger.info(f"Optimizer: {optimizer_name} with lr={self.config['learning_rate']}")

    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'none').lower()

        if scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('scheduler_factor', 0.5),
                patience=self.config.get('scheduler_patience', 10),
                min_lr=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None

        if self.scheduler:
            self.logger.info(f"Scheduler: {scheduler_type}")
        else:
            self.logger.info("No scheduler configured")

    def print_system_info(self):
        """Display system and process information"""
        self.logger.info("=" * 70)
        self.logger.info("SYSTEM & PROCESS INFORMATION:")
        self.logger.info("=" * 70)

        # Process information
        process = psutil.Process(os.getpid())
        self.logger.info(f"Process ID (PID): {os.getpid()}")
        self.logger.info(f"Process name: {process.name()}")
        self.logger.info(f"Parent PID: {process.ppid()}")

        # Memory information
        memory_info = process.memory_info()
        self.logger.info(f"Memory usage: {memory_info.rss / (1024**3):.2f} GB RSS, {memory_info.vms / (1024**3):.2f} GB VMS")
        self.logger.info(f"Memory percent: {process.memory_percent():.1f}%")

        # CPU information
        self.logger.info(f"CPU count: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
        self.logger.info(f"CPU percent: {psutil.cpu_percent(interval=1):.1f}%")

        # System information
        self.logger.info(f"System: {platform.system()} {platform.release()}")
        self.logger.info(f"Python version: {platform.python_version()}")

        # PyTorch information
        self.logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA available: Yes (version {torch.version.cuda})")
            self.logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                self.logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            self.logger.info("CUDA available: No")

        # Disk information
        disk_usage = psutil.disk_usage('.')
        self.logger.info(f"Disk usage: {disk_usage.used / (1024**3):.1f} GB used / {disk_usage.total / (1024**3):.1f} GB total ({disk_usage.percent:.1f}%)")

        self.logger.info("=" * 70)

    def print_dataset_info(self):
        """Print detailed dataset information after Data Configuration"""
        if not hasattr(self, 'dataset_info'):
            return

        info = self.dataset_info

        self.logger.info("\n" + "=" * 70)
        self.logger.info("DATA CONFIGURATION:")
        self.logger.info("=" * 70)

        # Data files information
        self.logger.info(f"Dataset path: {info['pkl_files_path']}")
        self.logger.info(f"Total .pt files found: {len(info['pkl_files'])}")
        if len(info['pkl_files']) > 0:
            self.logger.info("Data files:")
            for i, file in enumerate(info['pkl_files'][:5]):  # 只显示前5个文件
                file_size = os.path.getsize(file) / (1024*1024)  # MB
                self.logger.info(f"  [{i+1}] {os.path.basename(file)} ({file_size:.2f} MB)")
            if len(info['pkl_files']) > 5:
                self.logger.info(f"  ... and {len(info['pkl_files'])-5} more files")

        # Sample data analysis
        sample_x, sample_y = self.train_dataset[0]
        self.logger.info("-" * 50)
        self.logger.info("SAMPLE DATA ANALYSIS:")
        self.logger.info("-" * 50)
        self.logger.info(f"Sample input shape: {sample_x.shape}")
        self.logger.info(f"Sample target shape: {sample_y.shape}")
        self.logger.info(f"Input data type: {sample_x.dtype}")
        self.logger.info(f"Target data type: {sample_y.dtype}")

        # Input statistics
        if hasattr(sample_x, 'min') and hasattr(sample_x, 'max'):
            self.logger.info(f"Input value range: [{sample_x.min():.6f}, {sample_x.max():.6f}]")
            self.logger.info(f"Input mean: {sample_x.mean():.6f}, std: {sample_x.std():.6f}")

        # Target statistics
        if hasattr(sample_y, 'min') and hasattr(sample_y, 'max'):
            self.logger.info(f"Target value range: [{sample_y.min():.6f}, {sample_y.max():.6f}]")
            self.logger.info(f"Target mean: {sample_y.mean():.6f}, std: {sample_y.std():.6f}")

        # Check for NaN or Inf
        if torch.isnan(sample_x).any():
            self.logger.warning("⚠️  WARNING: Input data contains NaN values!")
        if torch.isinf(sample_x).any():
            self.logger.warning("⚠️  WARNING: Input data contains Inf values!")
        if torch.isnan(sample_y).any():
            self.logger.warning("⚠️  WARNING: Target data contains NaN values!")
        if torch.isinf(sample_y).any():
            self.logger.warning("⚠️  WARNING: Target data contains Inf values!")

        # Data split information
        self.logger.info("-" * 50)
        self.logger.info("DATA SPLIT:")
        self.logger.info("-" * 50)
        total_samples = info['train_size'] + info['val_size']
        self.logger.info(f"Total samples: {total_samples}")
        self.logger.info(f"Train set size: {info['train_size']} ({info['train_size']/total_samples*100:.1f}%)")
        self.logger.info(f"Validation set size: {info['val_size']} ({info['val_size']/total_samples*100:.1f}%)")
        self.logger.info(f"Train ratio: {self.config['train_ratio']}")

        # Dataloader configuration
        self.logger.info("-" * 50)
        self.logger.info("DATALOADER CONFIGURATION:")
        self.logger.info("-" * 50)
        self.logger.info(f"Batch size: {self.config['batch_size']}")
        self.logger.info(f"Train batches per epoch: {info['train_loader_len']}")
        self.logger.info(f"Validation batches per epoch: {info['val_loader_len']}")
        self.logger.info(f"Number of workers: {self.config['num_workers']}")
        self.logger.info("=" * 70)

    def setup_loss(self):
        """Setup loss function"""
        # Base loss function
        if self.config.get('task_type', 'regression') == 'classification':
            base_loss = nn.CrossEntropyLoss()
        elif self.config.get('task_type') == 'binary_classification':
            base_loss = nn.BCEWithLogitsLoss()
        else:
            base_loss = nn.MSELoss()

        # Add DSA loss if DSA is enabled
        if self.config.get('dsa_enabled', False):
            self.criterion = DSALoss(
                base_loss_fn=base_loss,
                sparsity_weight=self.config.get('sparsity_weight', 0.001),
                entropy_weight=self.config.get('entropy_weight', 0.0001)
            )
            self.logger.info("Using DSA-enhanced loss function")
        else:
            self.criterion = base_loss
            self.logger.info(f"Using base loss function: {type(base_loss).__name__}")

    def setup_checkpoint(self):
        """Setup checkpoint loading if resuming"""
        if self.config.get('resume_training', False):
            self.load_checkpoint()

    def load_checkpoint(self):
        """Load checkpoint if available"""
        # Find latest checkpoint
        checkpoint_pattern = os.path.join(
            self.config['model_dir'],
            "*/checkpoint_latest.pth"
        )
        checkpoint_files = glob.glob(checkpoint_pattern)

        if checkpoint_files:
            # Sort by modification time and get the latest
            checkpoint_files.sort(key=os.path.getmtime, reverse=True)
            latest_checkpoint = checkpoint_files[0]

            self.logger.info(f"Loading checkpoint: {latest_checkpoint}")

            # Load checkpoint
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')

            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                if self.scheduler and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                self.current_epoch = checkpoint.get('epoch', 0)
                self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                self.train_losses = checkpoint.get('train_losses', [])
                self.val_losses = checkpoint.get('val_losses', [])

                self.logger.info(f"Resumed from epoch {self.current_epoch}")
                self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
            else:
                self.logger.warning("Invalid checkpoint format")
        else:
            self.logger.warning("No checkpoint found for resuming")

    def save_checkpoint(self, epoch, is_best=False, is_latest=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save with different names
        if is_best:
            torch.save(checkpoint, os.path.join(self.model_save_dir, 'model_best.pth'))
            self.logger.info("Saved best model")

        if is_latest:
            torch.save(checkpoint, os.path.join(self.model_save_dir, 'checkpoint_latest.pth'))

        # Regular periodic save
        save_name = f"model_epoch_{epoch}.pth"
        torch.save(checkpoint, os.path.join(self.model_save_dir, save_name))

        # Save DSA stats if available
        if hasattr(self.model, 'get_dsa_stats'):
            dsa_stats = self.model.get_dsa_stats()
            stats_file = os.path.join(self.model_save_dir, f'dsa_stats_epoch_{epoch}.json')
            with open(stats_file, 'w') as f:
                json.dump(dsa_stats, f, indent=2)

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            # Move data to device
            data = data.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(data)

            # Calculate loss
            if self.config.get('dsa_enabled', False):
                # Get DSA stats from model if available
                dsa_stats = {}
                if hasattr(self.model, 'get_dsa_stats'):
                    dsa_stats = self.model.get_dsa_stats()

                loss, loss_components = self.criterion(outputs, targets, dsa_stats)
            else:
                loss = self.criterion(outputs, targets)
                loss_components = {'total_loss': loss.item()}

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for data, targets in self.val_loader:
                # Move data to device
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(data)

                # Calculate loss
                if self.config.get('dsa_enabled', False):
                    dsa_stats = {}
                    if hasattr(self.model, 'get_dsa_stats'):
                        dsa_stats = self.model.get_dsa_stats()

                    loss, loss_components = self.criterion(outputs, targets, dsa_stats)
                else:
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self):
        """Main training loop"""
        # System and data information have already been printed during initialization
        self.logger.info("\n" + "=" * 70)
        self.logger.info("TRAINING CONFIGURATION:")
        self.logger.info("=" * 70)
        self.logger.info(f"Total epochs: {self.config['num_epochs']}")
        self.logger.info(f"Batch size: {self.config['batch_size']}")
        self.logger.info(f"Learning rate: {self.config['learning_rate']}")
        self.logger.info(f"Optimizer: {self.config.get('optimizer', 'adamw')}")
        self.logger.info(f"DSA enabled: {self.config.get('dsa_enabled', False)}")
        self.logger.info("=" * 70)

        self.logger.info("\nStarting training...")

        # Move model to device
        self.model.to(self.device)

        # Note: For single GPU/CPU training, no need for accelerate
        # self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
        #     self.model, self.optimizer, self.train_loader, self.val_loader
        # )

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log epoch results
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}"
            )

            # Update DSA sparsity if enabled
            if self.config.get('dsa_enabled', False) and hasattr(self.model, 'update_dsa_sparsity'):
                self.model.update_dsa_sparsity(val_loss)

                # Log DSA stats
                if hasattr(self.model, 'get_dsa_stats'):
                    dsa_stats = self.model.get_dsa_stats()
                    self.logger.info(f"DSA Stats: {dsa_stats}")
                    self.dsa_stats_history.append(dsa_stats)

            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.logger.info(f"New best validation loss: {self.best_val_loss:.6f}")

            # Save checkpoint
            save_every = self.config.get('save_every', 10)
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best, is_latest=True)

            # Early stopping check
            patience = self.config.get('early_stopping_patience', 50)
            if patience > 0 and len(self.val_losses) > patience:
                recent_losses = self.val_losses[-patience:]
                if all(loss > self.best_val_loss + self.config.get('min_delta', 1e-6) for loss in recent_losses):
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")

        # Save final model
        self.save_checkpoint(self.config['num_epochs'] - 1, is_best=True, is_latest=True)

        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'config': self.config
        }

        if self.dsa_stats_history:
            history['dsa_stats_history'] = self.dsa_stats_history

        history_file = os.path.join(self.model_save_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        self.logger.info(f"Training history saved to: {history_file}")

        return history