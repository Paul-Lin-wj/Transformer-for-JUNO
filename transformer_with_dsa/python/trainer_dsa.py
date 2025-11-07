"""
=============================================
TRAINER FOR TRANSFORMER WITH DSA
DSA优化Transformer的训练器
=============================================

功能特性：
1. 完整的训练循环
2. 断点续训功能
3. 自动模型保存
4. 训练统计和可视化
5. 早停和学习率调度
6. DSA稀疏度动态调整
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from accelerate import Accelerator
import numpy as np
import os
import json
import time
import datetime
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from pathlib import Path

from transformer_dsa import TransformerWithDSA, create_default_dsa_config, DSALoss
from data_loader import DataLoaderWithDSA


class CheckpointManager:
    """
    检查点管理器
    负责保存和加载模型检查点
    """

    def __init__(self, checkpoint_dir, save_every=50, keep_last=5):
        """
        Args:
            checkpoint_dir: 检查点保存目录
            save_every: 每隔多少epoch保存一次
            keep_last: 保留最近几个检查点
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every = save_every
        self.keep_last = keep_last
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 检查点信息文件
        self.info_file = self.checkpoint_dir / "checkpoint_info.json"

    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss,
                        metrics=None, dsa_stats=None, training_state=None):
        """
        保存模型检查点

        Args:
            model: 模型实例
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            loss: 当前损失
            metrics: 评估指标
            dsa_stats: DSA统计信息
            training_state: 训练状态信息
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'metrics': metrics or {},
            'dsa_stats': dsa_stats or {},
            'training_state': training_state or {},
            'timestamp': datetime.datetime.now().isoformat()
        }

        # 保存检查点文件
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # 更新检查点信息
        self._update_checkpoint_info(epoch, checkpoint_path.name, loss, metrics)

        # 清理旧检查点
        self._cleanup_old_checkpoints()

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, model, optimizer, scheduler=None, checkpoint_path=None):
        """
        加载检查点

        Args:
            model: 模型实例
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
            checkpoint_path: 指定检查点路径，如果为None则加载最新的

        Returns:
            checkpoint_info: 检查点信息
        """
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            return None

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载调度器状态
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resumed from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")

        return checkpoint

    def _update_checkpoint_info(self, epoch, filename, loss, metrics):
        """更新检查点信息文件"""
        info = {
            'latest_epoch': epoch,
            'latest_checkpoint': filename,
            'latest_loss': loss,
            'latest_metrics': metrics,
            'checkpoints': []
        }

        # 读取现有信息
        if self.info_file.exists():
            try:
                with open(self.info_file, 'r') as f:
                    existing_info = json.load(f)
                info['checkpoints'] = existing_info.get('checkpoints', [])
            except:
                pass

        # 添加新检查点信息
        info['checkpoints'].append({
            'epoch': epoch,
            'filename': filename,
            'loss': loss,
            'metrics': metrics,
            'timestamp': datetime.datetime.now().isoformat()
        })

        # 保存信息
        with open(self.info_file, 'w') as f:
            json.dump(info, f, indent=2)

    def _get_latest_checkpoint(self):
        """获取最新的检查点路径"""
        if not self.info_file.exists():
            return None

        try:
            with open(self.info_file, 'r') as f:
                info = json.load(f)
            latest_filename = info.get('latest_checkpoint')
            if latest_filename:
                return self.checkpoint_dir / latest_filename
        except:
            pass

        return None

    def _cleanup_old_checkpoints(self):
        """清理旧的检查点文件"""
        if not self.info_file.exists():
            return

        try:
            with open(self.info_file, 'r') as f:
                info = json.load(f)

            checkpoints = info.get('checkpoints', [])
            if len(checkpoints) > self.keep_last:
                # 删除多余的检查点
                for checkpoint in checkpoints[:-self.keep_last]:
                    checkpoint_path = self.checkpoint_dir / checkpoint['filename']
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                        print(f"Removed old checkpoint: {checkpoint_path}")

                # 更新信息文件
                info['checkpoints'] = checkpoints[-self.keep_last:]
                with open(self.info_file, 'w') as f:
                    json.dump(info, f, indent=2)
        except Exception as e:
            print(f"Error cleaning up checkpoints: {e}")


class TrainerWithDSA:
    """
    DSA优化Transformer的训练器
    """

    def __init__(self, config):
        """
        Args:
            config: 训练配置字典
        """
        self.config = config

        # 初始化accelerator（使用默认配置）
        self.accelerator = Accelerator()

        # 禁用DDP的buffer同步以避免内存冲突
        if self.accelerator.distributed_type.value == "MULTI_GPU":
            import torch.distributed as dist
            # 保存原始的find_unused_parameters设置
            # 注意：这需要在prepare_model之前设置

        # 设置日志
        self._setup_logging()

        # 初始化模型
        self._init_model()

        # 初始化数据加载器
        self._init_data_loader()

        # 初始化优化器和损失函数
        self._init_optimizer()

        # 初始化检查点管理器
        self._init_checkpoint_manager()

        # 训练状态
        self.training_history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.start_epoch = 0

        self.logger.info("Trainer with DSA initialized")
        self.logger.info(f"Device: {self.accelerator.device}")
        self.logger.info(f"Number of processes: {self.accelerator.num_processes}")
        self.logger.info(f"Distributed type: {self.accelerator.distributed_type}")
        # 获取原始模型来显示模型大小
        original_model = self.accelerator.unwrap_model(self.model)
        self.logger.info(f"Model: {original_model.get_model_size()}")

    def _setup_logging(self):
        """设置日志"""
        log_dir = Path(self.config.get('log_dir', './logs'))
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _init_model(self):
        """初始化模型"""
        self.model = TransformerWithDSA(
            input_dim=self.config['input_dim'],
            embed_dim=self.config['embed_dim'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            ff_dim=self.config['ff_dim'],
            output_dim=self.config['output_dim'],
            dropout=self.config.get('dropout', 0.1),
            dsa_enabled=self.config.get('dsa_enabled', True),
            dsa_config=self.config.get('dsa_config', create_default_dsa_config()),
            task_type=self.config.get('task_type', 'regression')
        )

        # 使用accelerator准备模型（会自动处理多GPU分布）
        self.model = self.accelerator.prepare(self.model)

        # 对于多GPU训练，禁用buffer同步以避免内存冲突
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'broadcast_buffers'):
            self.model.module.broadcast_buffers = False

    def _init_data_loader(self):
        """初始化数据加载器"""
        self.data_loader = DataLoaderWithDSA(
            dataset_path=self.config['dataset_path'],
            input_dim=self.config['input_dim'],
            target_dim=self.config['output_dim'],
            seq_len=self.config.get('seq_len'),
            batch_size=self.config['batch_size'],
            train_ratio=self.config.get('train_ratio', 0.8),
            normalize=self.config.get('normalize', True),
            augment_train=self.config.get('augment_train', False),
            num_workers=self.config.get('num_workers', 4),
            max_files=self.config.get('max_files', None)
        )

        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()
        # 自动获取测试数据加载器（4:1分割）
        self.test_loader = self.data_loader.get_test_loader()

        # 使用accelerator准备数据加载器（会自动处理分布式采样）
        self.train_loader, self.val_loader, self.test_loader = self.accelerator.prepare(
            self.train_loader, self.val_loader, self.test_loader
        )

        self.logger.info(f"Test dataset loaded automatically: {len(self.test_loader.dataset)} test samples")

    def _init_optimizer(self):
        """初始化优化器和损失函数"""
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        # 学习率调度器
        scheduler_type = self.config.get('scheduler', 'plateau')
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('scheduler_factor', 0.5),
                patience=self.config.get('scheduler_patience', 10)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=self.config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None

        # 使用accelerator准备优化器
        self.optimizer = self.accelerator.prepare(self.optimizer)

        # 损失函数
        if self.config.get('task_type', 'regression') == 'classification':
            base_loss = nn.CrossEntropyLoss()
        else:
            base_loss = nn.MSELoss()

        self.criterion = DSALoss(
            base_loss_fn=base_loss,
            sparsity_weight=self.config.get('sparsity_weight', 0.001),
            entropy_weight=self.config.get('entropy_weight', 0.0001)
        )

    def _init_checkpoint_manager(self):
        """初始化检查点管理器"""
        model_dir = Path(self.config['model_dir'])
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = model_dir / timestamp

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            save_every=self.config.get('save_every', 50),
            keep_last=self.config.get('keep_last', 5)
        )

        self.current_model_dir = checkpoint_dir

    def _print_training_config(self):
        """打印训练配置信息"""
        import json
        self.logger.info("="*50)
        self.logger.info("TRAINING CONFIGURATION")
        self.logger.info("="*50)

        # 训练参数
        self.logger.info("Training Parameters:")
        self.logger.info(f"  Epochs: {self.config['num_epochs']}")
        self.logger.info(f"  Batch Size: {self.config['batch_size']}")
        self.logger.info(f"  Learning Rate: {self.config['learning_rate']}")
        self.logger.info(f"  Save Every: {self.config.get('save_every', 50)} epochs")
        self.logger.info(f"  Resume Training: {self.config.get('resume_training', False)}")

        # 模型参数
        self.logger.info("\nModel Parameters:")
        self.logger.info(f"  Input Dim: {self.config['input_dim']}")
        self.logger.info(f"  Embed Dim: {self.config['embed_dim']}")
        self.logger.info(f"  Num Heads: {self.config['num_heads']}")
        self.logger.info(f"  Num Layers: {self.config['num_layers']}")
        self.logger.info(f"  FF Dim: {self.config['ff_dim']}")
        self.logger.info(f"  Output Dim: {self.config['output_dim']}")
        self.logger.info(f"  Dropout: {self.config.get('dropout', 0.1)}")

        # DSA参数
        self.logger.info("\nDSA Parameters:")
        self.logger.info(f"  DSA Enabled: {self.config.get('dsa_enabled', True)}")
        if self.config.get('dsa_enabled', True):
            dsa_config = self.config.get('dsa_config', {})
            self.logger.info(f"  Initial Sparsity: {dsa_config.get('sparsity_ratio', 0.1)}")
            self.logger.info(f"  Target Sparsity: {dsa_config.get('target_sparsity', 0.05)}")
            self.logger.info(f"  Sparsity Weight: {self.config.get('sparsity_weight', 0.001)}")
            self.logger.info(f"  Entropy Weight: {self.config.get('entropy_weight', 0.0001)}")

        # 数据参数
        self.logger.info("\nData Parameters:")
        self.logger.info(f"  Dataset Path: {self.config['dataset_path']}")
        self.logger.info(f"  Data Split: 4:1 (Train+Val:Test)")
        self.logger.info(f"  Train/Val Split: {self.config.get('train_ratio', 0.8):.0%}:{1-self.config.get('train_ratio', 0.8):.0%} (Train:Val)")
        self.logger.info(f"  Sequence Length: {self.config.get('seq_len', 'None')}")
        self.logger.info(f"  Normalize: {self.config.get('normalize', True)}")
        self.logger.info(f"  Augment Train: {self.config.get('augment_train', False)}")

        # 系统参数
        self.logger.info("\nSystem Parameters:")
        self.logger.info(f"  Device: {self.accelerator.device}")
        self.logger.info(f"  Number of Processes: {self.accelerator.num_processes}")
        self.logger.info(f"  Distributed Type: {self.accelerator.distributed_type}")
        self.logger.info(f"  Num Workers: {self.config.get('num_workers', 4)}")
        self.logger.info(f"  Log Dir: {self.config.get('log_dir', './logs')}")
        self.logger.info(f"  Model Dir: {self.config['model_dir']}")

        # 打印数据集信息
        if hasattr(self, 'data_loader'):
            dataset_info = self.data_loader.get_dataset_info()
            self.logger.info("\nDataset Info:")
            self.logger.info(f"  Train Samples: {dataset_info['train_samples']}")
            self.logger.info(f"  Validation Samples: {dataset_info['val_samples']}")
            self.logger.info(f"  Test Samples: {dataset_info['test_samples']}")
            self.logger.info(f"  Data Shape: {dataset_info['data_shape']}")
            self.logger.info(f"  Target Shape: {dataset_info['target_shape']}")

        self.logger.info("="*50)

    def train(self):
        """主训练循环"""
        self.logger.info("Starting training...")

        # 打印训练配置
        self._print_training_config()

        # 尝试加载检查点
        resume_training = self.config.get('resume_training', False)
        if resume_training:
            checkpoint = self.checkpoint_manager.load_checkpoint(
                self.model, self.optimizer, self.scheduler
            )
            if checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_val_loss = checkpoint.get('metrics', {}).get('val_loss', float('inf'))
                self.training_history = defaultdict(list, checkpoint.get('training_state', {}))
                self.logger.info(f"Resumed training from epoch {self.start_epoch}")
            else:
                self.logger.info("No checkpoint found, starting from scratch")
        else:
            self.logger.info("Starting fresh training")

        # 训练循环
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            train_loss, train_metrics = self._train_epoch(epoch)
            val_loss, val_metrics = self._validate_epoch(epoch)

            # 测试集评估（如果有测试数据）
            test_metrics = self._test_evaluate(epoch)

            # 更新DSA稀疏度
            # 使用unwrapped model访问原始模型属性
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            if unwrapped_model.dsa_enabled:
                self.model.update_dsa_sparsity(val_loss)

            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 记录训练历史
            self._record_training_history(epoch, train_loss, train_metrics, val_loss, val_metrics, test_metrics)

            # 保存检查点（只在主进程中保存）
            if (epoch + 1) % self.checkpoint_manager.save_every == 0:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    training_state = dict(self.training_history)
                    # 获取unwrapped模型（移除accelerator包装）
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    self.checkpoint_manager.save_checkpoint(
                        model=unwrapped_model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        loss=val_loss,
                        metrics={'val_loss': val_loss, 'train_loss': train_loss},
                        dsa_stats=train_metrics.get('dsa_stats', {}),
                        training_state=training_state
                    )

            # 早停检查
            if self._should_early_stop(val_loss):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        # 保存最终模型（只在主进程中保存）
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._save_final_model()
            # 绘制训练曲线
            self._plot_training_curves()
            # 训练结束后进行详细验证
            self.logger.info("Starting final model validation...")
            self._validate_model_detailed()

        self.accelerator.wait_for_everyone()
        self.logger.info("Training completed!")

    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_metrics = defaultdict(list)

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            # accelerator会自动处理数据移动，不需要手动.to(device)
            pass

            self.optimizer.zero_grad()

            # 前向传播（使用unwrap_model访问原始模型属性）
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            if unwrapped_model.dsa_enabled:
                output, stats = self.model(data, training=True, return_attention=True)
                dsa_stats = stats['global_stats']
            else:
                output = self.model(data, training=True)
                dsa_stats = {}

            # 对于回归任务，只使用最后一个时间步的输出
            if self.config.get('task_type', 'regression') == 'regression':
                output = output[:, -1, :]  # 取最后一个时间步 [batch_size, output_dim]

            # 处理形状不匹配的情况
            # 如果targets是3维而output是2维，压缩targets的中间维度
            if len(targets.shape) == 3 and len(output.shape) == 2:
                if targets.shape[1] == 1:
                    targets = targets.squeeze(1)  # [batch_size, 1, dim] -> [batch_size, dim]

            # 如果output是1维而targets是2维，扩展output
            if len(output.shape) == 1 and len(targets.shape) == 2:
                output = output.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]

            # 确保output和targets的形状兼容
            # 如果output的最后一个维度是1而targets更大，或反之
            if output.shape[-1] == 1 and targets.shape[-1] > 1:
                # 这种情况下，假设output应该重复匹配targets的维度
                output = output.expand(-1, targets.shape[-1])
            elif targets.shape[-1] == 1 and output.shape[-1] > 1:
                targets = targets.expand(-1, output.shape[-1])

            # 计算损失
            loss, loss_components = self.criterion(output, targets, dsa_stats)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

            self.optimizer.step()

            total_loss += loss.item()

            # 记录指标
            for key, value in loss_components.items():
                all_metrics[key].append(value)
            if dsa_stats:
                all_metrics['sparsity'].append(dsa_stats.get('avg_sparsity', 0))

            # 定期输出（只在主进程中输出）
            if batch_idx % 100 == 0 and self.accelerator.is_main_process:
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                               f"Loss: {loss.item():.6f}")

        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

        return avg_loss, avg_metrics

    def _validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        all_metrics = defaultdict(list)

        with torch.no_grad():
            for data, targets in self.val_loader:
                # accelerator会自动处理数据移动，不需要手动.to(device)
                pass

                # 检查输入数据是否包含NaN
                if torch.isnan(data).any():
                    self.logger.error("Input data contains NaN! Skipping this batch.")
                    continue
                if torch.isnan(targets).any():
                    self.logger.error("Targets contain NaN! Skipping this batch.")
                    continue

                # 前向传播
                # 使用unwrapped model访问原始模型属性
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                if unwrapped_model.dsa_enabled:
                    output, stats = self.model(data, training=False, return_attention=True)
                    dsa_stats = stats['global_stats']
                else:
                    output = self.model(data, training=False)
                    dsa_stats = {}

                # 对于回归任务，只使用最后一个时间步的输出
                if self.config.get('task_type', 'regression') == 'regression':
                    output = output[:, -1, :]  # 取最后一个时间步 [batch_size, output_dim]

                # 检查模型输出是否包含NaN
                if torch.isnan(output).any():
                    self.logger.error("Model output contains NaN! Skipping this batch.")
                    continue

                # 处理形状不匹配的情况
                # 如果targets是3维而output是2维，压缩targets的中间维度
                if len(targets.shape) == 3 and len(output.shape) == 2:
                    if targets.shape[1] == 1:
                        targets = targets.squeeze(1)  # [batch_size, 1, dim] -> [batch_size, dim]

                # 如果output是1维而targets是2维，扩展output
                if len(output.shape) == 1 and len(targets.shape) == 2:
                    output = output.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]

                # 确保output和targets的形状兼容
                # 如果output的最后一个维度是1而targets更大，或反之
                if output.shape[-1] == 1 and targets.shape[-1] > 1:
                    # 这种情况下，假设output应该重复匹配targets的维度
                    output = output.expand(-1, targets.shape[-1])
                elif targets.shape[-1] == 1 and output.shape[-1] > 1:
                    targets = targets.expand(-1, output.shape[-1])

                # 调试信息（只记录第一个batch）
                if epoch == 0 and total_loss == 0:
                    self.logger.debug(f"Validation - Output shape: {output.shape}, Targets shape: {targets.shape}")

                # 计算损失
                loss, loss_components = self.criterion(output, targets, dsa_stats)

                # 检查loss是否为NaN
                if torch.isnan(loss):
                    self.logger.error(f"NaN loss detected! Output shape: {output.shape}, Targets shape: {targets.shape}")
                    self.logger.error(f"Output sample: {output[0] if len(output) > 0 else 'Empty'}")
                    self.logger.error(f"Target sample: {targets[0] if len(targets) > 0 else 'Empty'}")
                    continue  # 跳过这个batch

                total_loss += loss.item()

                # 记录指标
                for key, value in loss_components.items():
                    # 处理NaN值
                    if isinstance(value, (int, float)):
                        if np.isnan(value):
                            value = 0.0
                    all_metrics[key].append(value)
                if dsa_stats:
                    all_metrics['sparsity'].append(dsa_stats.get('avg_sparsity', 0))

        # 处理没有有效batch的情况
        if len(self.val_loader) == 0:
            return float('nan'), {}

        # 计算平均指标（如果有有效batch）
        if total_loss == 0 and len(all_metrics) == 0:
            # 所有batch都被跳过了
            self.logger.warning("All validation batches were skipped due to NaN values.")
            return 0.0, {}

        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {}
        for k, v in all_metrics.items():
            if v:  # 如果列表不为空
                avg_metrics[k] = np.mean(v)
            else:
                avg_metrics[k] = 0.0

        self.logger.info(f"Epoch {epoch} - Val Loss: {avg_loss:.6f}")

        return avg_loss, avg_metrics

    def _test_evaluate(self, epoch):
        """测试集评估 - 计算入口点和出口点的MSE误差"""
        if self.test_loader is None:
            return None

        self.model.eval()
        total_mse_loss = 0
        entry_point_loss = 0
        exit_point_loss = 0
        num_samples = 0

        with torch.no_grad():
            for data, targets in self.test_loader:
                # accelerator会自动处理数据移动，不需要手动.to(device)
                pass

                # 前向传播
                # 使用unwrapped model访问原始模型属性
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                if unwrapped_model.dsa_enabled:
                    output, _ = self.model(data, training=False, return_attention=True)
                else:
                    output = self.model(data, training=False)

                # 对于回归任务，只使用最后一个时间步的输出
                if self.config.get('task_type', 'regression') == 'regression':
                    output = output[:, -1, :]  # [batch_size, output_dim]

                # 处理形状不匹配的情况
                if len(targets.shape) == 3 and len(output.shape) == 2:
                    if targets.shape[1] == 1:
                        targets = targets.squeeze(1)

                if len(output.shape) == 1 and len(targets.shape) == 2:
                    output = output.unsqueeze(-1)

                # 确保output和targets的形状兼容
                if output.shape[1] != targets.shape[1]:
                    min_dim = min(output.shape[1], targets.shape[1])
                    output = output[:, :min_dim]
                    targets = targets[:, :min_dim]

                # 检查是否包含NaN
                if torch.isnan(output).any() or torch.isnan(targets).any():
                    continue

                # 计算入口点和出口点的MSE
                # 假设targets和output都是6维：[entry_x, entry_y, entry_z, exit_x, exit_y, exit_z]
                if output.shape[1] >= 6:
                    # 入口点误差（前3维）
                    pred_entry = output[:, :3]
                    true_entry = targets[:, :3]
                    entry_mse = F.mse_loss(pred_entry, true_entry)

                    # 出口点误差（后3维）
                    pred_exit = output[:, 3:6]
                    true_exit = targets[:, 3:6]
                    exit_mse = F.mse_loss(pred_exit, true_exit)

                    # 总体MSE
                    total_mse = F.mse_loss(output[:, :6], targets[:, :6])

                    entry_point_loss += entry_mse.item() * data.size(0)
                    exit_point_loss += exit_mse.item() * data.size(0)
                    total_mse_loss += total_mse.item() * data.size(0)
                    num_samples += data.size(0)

        if num_samples == 0:
            self.logger.warning("No valid test samples found")
            return None

        # 计算平均损失
        avg_entry_loss = entry_point_loss / num_samples
        avg_exit_loss = exit_point_loss / num_samples
        avg_total_loss = total_mse_loss / num_samples

        # 打印测试结果
        self.logger.info(f"Epoch {epoch} - Test Loss (Entry Point): {avg_entry_loss:.6f}")
        self.logger.info(f"Epoch {epoch} - Test Loss (Exit Point): {avg_exit_loss:.6f}")
        self.logger.info(f"Epoch {epoch} - Test Loss (Total MSE): {avg_total_loss:.6f}")

        # 返回测试结果
        return {
            'test_entry_loss': avg_entry_loss,
            'test_exit_loss': avg_exit_loss,
            'test_total_loss': avg_total_loss
        }

    def _record_training_history(self, epoch, train_loss, train_metrics, val_loss, val_metrics, test_metrics=None):
        """记录训练历史"""
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['epoch'].append(epoch)

        # 记录其他指标
        for key, value in train_metrics.items():
            self.training_history[f'train_{key}'].append(value)
        for key, value in val_metrics.items():
            self.training_history[f'val_{key}'].append(value)

        # 记录测试指标
        if test_metrics:
            for key, value in test_metrics.items():
                self.training_history[key].append(value)

    def _should_early_stop(self, val_loss):
        """早停判断"""
        patience = self.config.get('early_stopping_patience', 50)
        min_delta = self.config.get('min_delta', 1e-6)

        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            return False
        else:
            self.epochs_no_improve += 1
            return self.epochs_no_improve >= patience

    def _save_final_model(self):
        """保存最终模型"""
        final_model_path = self.current_model_dir / "final_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': dict(self.training_history),
            'best_val_loss': self.best_val_loss
        }, final_model_path)

        self.logger.info(f"Final model saved: {final_model_path}")

    def _plot_training_curves(self):
        """绘制训练曲线"""
        try:
            plt.figure(figsize=(15, 10))

            # 损失曲线
            plt.subplot(2, 3, 1)
            plt.plot(self.training_history['epoch'], self.training_history['train_loss'], label='Train Loss')
            # 只有当val_loss不为空时才绘制
            if self.training_history['val_loss'] and len(self.training_history['val_loss']) > 0:
                plt.plot(self.training_history['epoch'], self.training_history['val_loss'], label='Val Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            # 稀疏度曲线（如果启用DSA）
            if 'train_sparsity' in self.training_history:
                plt.subplot(2, 3, 2)
                plt.plot(self.training_history['epoch'], self.training_history['train_sparsity'])
                plt.title('DSA Sparsity Over Time')
                plt.xlabel('Epoch')
                plt.ylabel('Sparsity Ratio')
                plt.grid(True)

            # 其他指标
            if 'train_task_loss' in self.training_history:
                plt.subplot(2, 3, 3)
                plt.plot(self.training_history['epoch'], self.training_history['train_task_loss'], label='Train')
                plt.plot(self.training_history['epoch'], self.training_history['val_task_loss'], label='Val')
                plt.title('Task Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)

            # 保存图像
            plot_path = self.current_model_dir / "training_curves.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Training curves saved: {plot_path}")

        except Exception as e:
            self.logger.error(f"Error plotting training curves: {e}")

    def _validate_model_detailed(self):
        """详细验证模型，计算几何误差指标并保存结果"""
        import numpy as np
        import matplotlib.pyplot as plt

        self.model.eval()

        # 收集所有预测结果和真实值
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, targets in self.val_loader:
                # accelerator会自动处理数据移动，不需要手动.to(device)
                pass

                # 前向传播
                # 使用unwrapped model访问原始模型属性
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                if unwrapped_model.dsa_enabled:
                    output, _ = self.model(data, training=False, return_attention=True)
                else:
                    output = self.model(data, training=False)

                # 对于回归任务，只使用最后一个时间步的输出
                if self.config.get('task_type', 'regression') == 'regression':
                    output = output[:, -1, :]

                # 处理形状不匹配
                if len(targets.shape) == 3 and len(output.shape) == 2:
                    if targets.shape[1] == 1:
                        targets = targets.squeeze(1)

                if len(output.shape) == 1 and len(targets.shape) == 2:
                    output = output.unsqueeze(-1)

                # 确保形状兼容
                if output.shape[1] != targets.shape[1]:
                    min_dim = min(output.shape[1], targets.shape[1])
                    output = output[:, :min_dim]
                    targets = targets[:, :min_dim]

                # 转换为numpy并收集
                all_predictions.append(output.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # 合并所有结果
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # 计算各种误差指标
        results = self._calculate_geometric_errors(predictions, targets)

        # 保存结果到文件
        self._save_validation_results(results)

        # 绘制误差分布图
        self._plot_error_distributions(results)

        self.logger.info("Detailed validation completed!")

    def _calculate_geometric_errors(self, predictions, targets):
        """计算几何误差指标"""
        results = []

        for i in range(len(predictions)):
            if targets.shape[1] < 6:
                # 如果目标维度不足6，跳过或填充
                continue

            pred = predictions[i]
            true = targets[i]

            # 提取入口点和出口点
            pred_entry = pred[:3]
            pred_exit = pred[3:6] if pred.shape[0] >= 6 else pred[3:]
            true_entry = true[:3]
            true_exit = true[3:6] if true.shape[0] >= 6 else true[3:]

            # 计算坐标差
            entry_diff = np.linalg.norm(pred_entry - true_entry)
            exit_diff = np.linalg.norm(pred_exit - true_exit)

            # 计算角度差（两条直线的夹角）
            pred_direction = pred_exit - pred_entry
            true_direction = true_exit - true_entry

            # 归一化方向向量
            pred_norm = np.linalg.norm(pred_direction)
            true_norm = np.linalg.norm(true_direction)

            if pred_norm > 0 and true_norm > 0:
                pred_direction = pred_direction / pred_norm
                true_direction = true_direction / true_norm

                # 计算夹角（弧度）
                cos_angle = np.clip(np.dot(pred_direction, true_direction), -1.0, 1.0)
                angle_diff = np.arccos(cos_angle)
                angle_diff_deg = np.degrees(angle_diff)
            else:
                angle_diff_deg = 0.0

            # 计算中点差
            pred_mid = (pred_entry + pred_exit) / 2
            true_mid = (true_entry + true_exit) / 2
            mid_diff = np.linalg.norm(pred_mid - true_mid)

            # 保存结果
            result = {
                'sample_id': i,
                'pred_entry': pred_entry,
                'pred_exit': pred_exit,
                'true_entry': true_entry,
                'true_exit': true_exit,
                'entry_diff': entry_diff,
                'exit_diff': exit_diff,
                'angle_diff_deg': angle_diff_deg,
                'mid_diff': mid_diff,
                'pred_line_length': pred_norm,
                'true_line_length': true_norm
            }
            results.append(result)

        return results

    def _save_validation_results(self, results):
        """保存验证结果到txt文件"""
        results_file = self.current_model_dir / "validation_results.txt"

        with open(results_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED MODEL VALIDATION RESULTS\n")
            f.write("="*80 + "\n\n")

            # 写入统计信息
            if results:
                entry_diffs = [r['entry_diff'] for r in results]
                exit_diffs = [r['exit_diff'] for r in results]
                angle_diffs = [r['angle_diff_deg'] for r in results]
                mid_diffs = [r['mid_diff'] for r in results]

                f.write("SUMMARY STATISTICS:\n")
                f.write("-"*40 + "\n")
                f.write(f"Total samples: {len(results)}\n")
                f.write(f"Entry point error - Mean: {np.mean(entry_diffs):.6f}, Std: {np.std(entry_diffs):.6f}\n")
                f.write(f"Exit point error - Mean: {np.mean(exit_diffs):.6f}, Std: {np.std(exit_diffs):.6f}\n")
                f.write(f"Angle difference - Mean: {np.mean(angle_diffs):.6f}°, Std: {np.std(angle_diffs):.6f}°\n")
                f.write(f"Midpoint error - Mean: {np.mean(mid_diffs):.6f}, Std: {np.std(mid_diffs):.6f}\n\n")

            # 写入每个样本的详细结果
            f.write("DETAILED RESULTS:\n")
            f.write("-"*40 + "\n")
            f.write("ID | Entry_Err | Exit_Err | Angle_Diff(°) | Mid_Err | LineLen_Pred | LineLen_True\n")
            f.write("-"*85 + "\n")

            for r in results:
                f.write(f"{r['sample_id']:3d} | ")
                f.write(f"{r['entry_diff']:10.6f} | ")
                f.write(f"{r['exit_diff']:9.6f} | ")
                f.write(f"{r['angle_diff_deg']:12.6f} | ")
                f.write(f"{r['mid_diff']:8.6f} | ")
                f.write(f"{r['pred_line_length']:12.6f} | ")
                f.write(f"{r['true_line_length']:12.6f}\n")

                # 写入坐标信息（可选）
                f.write(f"    Pred Entry: [{r['pred_entry'][0]:.6f}, {r['pred_entry'][1]:.6f}, {r['pred_entry'][2]:.6f}]\n")
                f.write(f"    True Entry: [{r['true_entry'][0]:.6f}, {r['true_entry'][1]:.6f}, {r['true_entry'][2]:.6f}]\n")
                f.write(f"    Pred Exit:  [{r['pred_exit'][0]:.6f}, {r['pred_exit'][1]:.6f}, {r['pred_exit'][2]:.6f}]\n")
                f.write(f"    True Exit:  [{r['true_exit'][0]:.6f}, {r['true_exit'][1]:.6f}, {r['true_exit'][2]:.6f}]\n")
                f.write("\n")

        self.logger.info(f"Validation results saved: {results_file}")

    def _plot_error_distributions(self, results):
        """绘制误差分布统计图"""
        if not results:
            self.logger.warning("No results to plot")
            return

        import matplotlib.pyplot as plt

        # 提取误差数据
        entry_diffs = [r['entry_diff'] for r in results]
        exit_diffs = [r['exit_diff'] for r in results]
        angle_diffs = [r['angle_diff_deg'] for r in results]
        mid_diffs = [r['mid_diff'] for r in results]

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Error Distribution Statistics', fontsize=16)

        # 入口点误差分布
        axes[0, 0].hist(entry_diffs, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Entry Point Error Distribution')
        axes[0, 0].set_xlabel('Error (mm)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(np.mean(entry_diffs), color='red', linestyle='--',
                          label=f'Mean: {np.mean(entry_diffs):.4f}')
        axes[0, 0].legend()

        # 出口点误差分布
        axes[0, 1].hist(exit_diffs, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Exit Point Error Distribution')
        axes[0, 1].set_xlabel('Error (mm)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(np.mean(exit_diffs), color='red', linestyle='--',
                          label=f'Mean: {np.mean(exit_diffs):.4f}')
        axes[0, 1].legend()

        # 角度差分布
        axes[1, 0].hist(angle_diffs, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Angle Difference Distribution')
        axes[1, 0].set_xlabel('Angle (degrees)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(np.mean(angle_diffs), color='red', linestyle='--',
                          label=f'Mean: {np.mean(angle_diffs):.2f}°')
        axes[1, 0].legend()

        # 中点误差分布
        axes[1, 1].hist(mid_diffs, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_title('Midpoint Error Distribution')
        axes[1, 1].set_xlabel('Error (mm)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(np.mean(mid_diffs), color='red', linestyle='--',
                          label=f'Mean: {np.mean(mid_diffs):.4f}')
        axes[1, 1].legend()

        plt.tight_layout()

        # 保存图像
        plot_path = self.current_model_dir / "error_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Error distribution plots saved: {plot_path}")


def test_trainer():
    """测试训练器"""
    print("Testing Trainer with DSA...")

    # 创建测试配置
    config = {
        'input_dim': 5,
        'embed_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'ff_dim': 128,
        'output_dim': 1,
        'batch_size': 16,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'dataset_path': '/data/juno/lin/JUNO/transformer/muon_track_reco_transformer/sample/dataset',
        'model_dir': './test_models',
        'log_dir': './test_logs',
        'dsa_enabled': True,
        'save_every': 2,  # 每2个epoch保存一次（测试用）
        'resume_training': False
    }

    try:
        # 创建训练器
        trainer = TrainerWithDSA(config)

        # 开始训练
        trainer.train()

        print("Trainer test passed!")

    except Exception as e:
        print(f"Trainer test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_trainer()