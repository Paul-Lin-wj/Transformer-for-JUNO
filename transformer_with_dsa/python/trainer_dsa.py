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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.model.get_model_size()}")

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
        ).to(self.device)

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
            num_workers=self.config.get('num_workers', 4)
        )

        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()

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

    def train(self):
        """主训练循环"""
        self.logger.info("Starting training...")

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

            # 更新DSA稀疏度
            if self.model.dsa_enabled:
                self.model.update_dsa_sparsity(val_loss)

            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 记录训练历史
            self._record_training_history(epoch, train_loss, train_metrics, val_loss, val_metrics)

            # 保存检查点
            if (epoch + 1) % self.checkpoint_manager.save_every == 0:
                training_state = dict(self.training_history)
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
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

        # 保存最终模型
        self._save_final_model()

        # 绘制训练曲线
        self._plot_training_curves()

        self.logger.info("Training completed!")

    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_metrics = defaultdict(list)

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            if self.model.dsa_enabled:
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

            # 定期输出
            if batch_idx % 100 == 0:
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
                data, targets = data.to(self.device), targets.to(self.device)

                # 检查输入数据是否包含NaN
                if torch.isnan(data).any():
                    self.logger.error("Input data contains NaN! Skipping this batch.")
                    continue
                if torch.isnan(targets).any():
                    self.logger.error("Targets contain NaN! Skipping this batch.")
                    continue

                # 前向传播
                if self.model.dsa_enabled:
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

    def _record_training_history(self, epoch, train_loss, train_metrics, val_loss, val_metrics):
        """记录训练历史"""
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['epoch'].append(epoch)

        # 记录其他指标
        for key, value in train_metrics.items():
            self.training_history[f'train_{key}'].append(value)
        for key, value in val_metrics.items():
            self.training_history[f'val_{key}'].append(value)

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