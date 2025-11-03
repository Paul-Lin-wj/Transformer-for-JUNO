import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data import Dataset, ConcatDataset, TensorDataset
from accelerate import Accelerator
from accelerate import notebook_launcher
from torchsummary import summary
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import io
import subprocess
import time

from TensorPre.TensorDataset import *
from TensorPre.MLModel import *
from Fuc import Fuc
from SendMail import SendMail
import os

accelerator = Accelerator()
pmt_radius = 19433.975


class ModelTrain:
    def __init__(self, arguments):
        self.arguments = arguments

        self.mission_name = arguments["mission_name"]
        self.output_path = arguments["output_path"]
        self.pre_method = arguments["pre_method"]
        self.model_name = arguments["train_model_name"]
        self.num_epochs = arguments["num_epochs"]
        self.train_test_num = arguments["train_test_num"]
        self.test_size = arguments["test_size"]
        self.learning_rate = arguments["learning_rate"]
        self.scheduler_step = arguments["scheduler_step"]
        self.batch_size = arguments["batch_size"]
        self.GPUid = arguments["GPUid"]
        self.is_fine_tuning = arguments.get("ModelTuning", False)
        self.pretrained_model_path = arguments.get("pretrained_model_path", None)

        if self.is_fine_tuning:
            self.result_save_path = (
                f"{self.output_path}/{self.mission_name}/tunning_results/result"
            )
        else:
            self.result_save_path = f"{self.output_path}/{self.mission_name}/result"
        result = subprocess.run(f"mkdir -p {self.result_save_path}", shell=True)

        # data load
        self.train_set_size = 0
        self.test_set_size = 0
        self.train_dataset, self.val_dataset = self.load_dataset()
        # run
        self.run()

    def load_dataset(self):
        fuc = Fuc()
        pkl_files_path = self.arguments["pklfile_train_path"]
        pkl_files = fuc.get_file_paths(pkl_files_path)
        print("Using existing dataset")
        dataset = CombinedDataset(pkl_files)

        print(f"Using method: {self.pre_method}")
        print(f"test_size: {self.test_size}")

        # test underfitting (select 10 events for train)
        # dataset = Subset(dataset, list(range(10)))

        test_ratio = self.test_size
        train_size = int((1 - test_ratio) * len(dataset))
        test_size = len(dataset) - train_size

        self.train_set_size = train_size
        self.test_set_size = test_size

        # torch.manual_seed(self.id + 1)
        train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

        sample_data, _ = train_dataset[0]
        print(f"Event data shape: {sample_data.shape}")
        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Val dataset length: {len(val_dataset)}")

        return train_dataset, val_dataset

    def run(self):
        # dataloader
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=True
        )
        model = self.define_model(self.pre_method, self.arguments)
        print(f"Model: {self.model_name}")
        criterion = CustomLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.scheduler_step, min_lr=1e-5
        )
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
        # print model parameters
        self.print_model_parameters(model)
        # training
        start_time = time.time()
        print(f"GPU num: {torch.cuda.device_count()}")

        # Print training mode information
        if self.is_fine_tuning:
            print("Running in FINE-TUNING mode")
            if self.pretrained_model_path:
                print(f"Using pretrained model: {self.pretrained_model_path}")
        else:
            print("Running in NORMAL TRAINING mode")

        train_loss_history = []
        val_loss_history = []

        for epoch in range(self.num_epochs):
            model.train()
            # 训练loss累加
            local_train_loss_sum = torch.tensor(0.0, device=accelerator.device)
            local_train_batches = torch.tensor(0, device=accelerator.device)
            for i, data in enumerate(train_loader):
                X_train, y_train = data
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                local_train_loss_sum += loss.detach()
                local_train_batches += 1
                if (i + 1) % 2000 == 0:
                    accelerator.print(
                        f"Epoch {epoch+1} Step {i+1}/{len(train_loader)} Loss {loss.item():.4f}"
                    )
            # 汇总训练 loss
            gathered_loss = accelerator.gather(local_train_loss_sum)
            gathered_batches = accelerator.gather(local_train_batches)
            total_loss_sum = gathered_loss.sum().item()
            total_batch_count = int(gathered_batches.sum().item())
            if total_batch_count > 0:
                avg_train_loss = total_loss_sum / total_batch_count
            else:
                avg_train_loss = float("nan")
            # validation
            model.eval()
            local_val_loss_sum = torch.tensor(0.0, device=accelerator.device)
            local_val_batches = torch.tensor(0, device=accelerator.device)
            with torch.no_grad():
                for j, data in enumerate(val_loader):
                    X_val, y_val = data
                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    local_val_loss_sum += loss.detach()
                    local_val_batches += 1
            gathered_val_loss = accelerator.gather(local_val_loss_sum)
            gathered_val_batches = accelerator.gather(local_val_batches)
            total_val_loss_sum = gathered_val_loss.sum().item()
            total_val_batch_count = int(gathered_val_batches.sum().item())
            if total_val_batch_count > 0:
                avg_val_loss = total_val_loss_sum / total_val_batch_count
            else:
                avg_val_loss = float("nan")
            # 仅主进程记录/打印/保存
            if accelerator.is_main_process:
                train_loss_history.append(avg_train_loss)
                val_loss_history.append(avg_val_loss)
                accelerator.print(
                    f"Epoch [{epoch+1}/{self.num_epochs}] Train loss: {avg_train_loss:.6f}  Val loss: {avg_val_loss:.6f} learning rate: {scheduler.get_last_lr()[0]:.6f}"
                )

            scheduler.step(avg_val_loss)

            # ===== 每100 epoch保存模型和loss曲线，以及在验证集做性能分析 ======
            if (epoch + 1) % 100 == 0 or (epoch == self.num_epochs - 1):
                if accelerator.is_main_process:
                    # 保存loss曲线
                    plt.figure(figsize=(10, 5))
                    plt.plot(
                        range(1, len(train_loss_history) + 1),
                        train_loss_history,
                        label="Train Loss",
                    )
                    plt.plot(
                        range(1, len(val_loss_history) + 1),
                        val_loss_history,
                        label="Val Loss",
                    )
                    plt.title("Training and Validation Loss Curve")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.grid()
                    plt.savefig(f"{self.result_save_path}/train_val_loss_curve.jpg")
                    plt.close()
                    # checkpoint: 保存模型权重
                    torch.save(
                        accelerator.unwrap_model(model).state_dict(),
                        f"{self.result_save_path}/{self.model_name}_checkpoint_epoch{epoch+1}.pth",
                    )
                    # 在验证集做预测和性能评估
                    predictions = []
                    actuals = []
                    model.eval()
                    with torch.no_grad():
                        for data in val_loader:
                            X_val, y_val = data
                            X_val = X_val.float()
                            y_val = y_val.float()
                            outputs = model(X_val)
                            predictions.append(outputs.cpu().numpy())
                            actuals.append(y_val.cpu().numpy())
                    # 性能统计
                    (
                        test_angle_quantile_68,
                        test_dist_quantile_68,
                        test_angles,
                        test_midpoint_dist,
                    ) = self.check_result(predictions, actuals, epoch=epoch + 1)
                    with open(
                        f"{self.result_save_path}/reconstruction_accuracy_epoch{epoch+1}.txt",
                        "w",
                    ) as f:
                        f.write(f"Angle 68% quantile: {test_angle_quantile_68}\n")
                        f.write(
                            f"midpoints distance 68% quantile: {test_dist_quantile_68}\n"
                        )

        # 训练结束后，主进程写入最终文件
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(
                unwrapped_model.state_dict(),
                f"{self.result_save_path}/{self.model_name}_final.pth",
            )

        # 清理分布式环境，避免资源泄漏
        accelerator.wait_for_everyone()
        try:
            accelerator.free_memory()
            if hasattr(torch.distributed, 'destroy_process_group'):
                torch.distributed.destroy_process_group()
        except:
            pass

        end_time = time.time()
        if accelerator.is_main_process:
            print(f"Training time: {(end_time - start_time)/60:.2f} minutes")
            # 最终loss曲线
            plt.figure(figsize=(10, 5))
            plt.plot(
                range(1, self.num_epochs + 1), train_loss_history, label="Train Loss"
            )
            plt.plot(range(1, self.num_epochs + 1), val_loss_history, label="Val Loss")
            plt.title("Training and Validation Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid()
            plt.savefig(f"{self.result_save_path}/train_val_loss_curve.jpg")
            plt.close()
            # test
            predictions = []
            actuals = []
            with torch.no_grad():
                for data in val_loader:
                    X_val, y_val = data
                    X_val = X_val.float()
                    y_val = y_val.float()
                    outputs = model(X_val)
                    predictions.append(outputs.cpu().numpy())
                    actuals.append(y_val.cpu().numpy())
            (
                test_angle_quantile_68,
                test_dist_quantile_68,
                test_angles,
                test_midpoint_dist,
            ) = self.check_result(predictions, actuals, save_prefix="")
            print(f"Test angle 68% quantile: {test_angle_quantile_68}")
            print(f"Test midpoints distance 68% quantile: {test_dist_quantile_68}")
            with open(f"{self.result_save_path}/reconstruction_accuracy.txt", "w") as f:
                f.write(f"Angle 68% quantile: {test_angle_quantile_68}\n")
                f.write(f"midpoints distance 68% quantile: {test_dist_quantile_68}\n")
            training_mode = "fine-tuning" if self.is_fine_tuning else "training"
            print(f"Model {training_mode} finished")

    def define_model(self, pre_method, arguments):
        if pre_method == "TensorPre":
            model_name = arguments["train_model_name"]
            if model_name == "Transformer":
                model = TransformerModel(
                    embed_dim=arguments["embed_dim"],
                    num_heads=arguments["num_heads"],
                    num_layers=arguments["num_layers"],
                    ff_dim=arguments["hidden_dim"],
                    input_dim=arguments["input_dim"],
                    # max_seq_len=arguments["max_hits"],
                )
        else:
            print("Can't identify the pre_method or model name")

        # Load pretrained model if fine-tuning
        if self.is_fine_tuning and self.pretrained_model_path:
            if os.path.exists(self.pretrained_model_path):
                print(f"Loading pretrained model from: {self.pretrained_model_path}")
                model.load_state_dict(
                    torch.load(self.pretrained_model_path, map_location="cpu")
                )
                print("Pretrained model loaded successfully")
            else:
                print(
                    f"Warning: Pretrained model path does not exist: {self.pretrained_model_path}"
                )
                print("Starting with random initialization")

        return model

    def print_model_parameters(self, model):
        total = 0
        print("Layer-wise parameter count:")
        for name, param in model.named_parameters():
            print(
                f"{name:40}: {param.numel()} ({'train' if param.requires_grad else 'freeze'})"
            )
            total += param.numel()
        print(f"Total model parameters: {total}")

    def check_result(self, predictions, actuals, epoch=None, save_prefix=""):
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)

        # get enter point and exit point
        pred_enter = predictions[:, :3]
        pred_exit = predictions[:, 3:6]
        act_enter = actuals[:, :3]
        act_exit = actuals[:, 3:6]

        # caculate angle between predicted track and actual track
        pred_vectors = pred_exit - pred_enter
        act_vectors = act_exit - act_enter
        norms_act = np.linalg.norm(act_vectors, axis=1)
        norms_pred = np.linalg.norm(pred_vectors, axis=1)
        dot_products = np.sum(pred_vectors * act_vectors, axis=1)
        cosine_angles = dot_products / (norms_pred * norms_act)
        cosine_angles = np.clip(cosine_angles, -1.0, 1.0)
        angles = np.arccos(cosine_angles) * 180 / np.pi

        sorted_cosangle = sorted(angles)
        index_68percent = int(len(sorted_cosangle) * 0.68)
        quantile_angle_68 = sorted_cosangle[index_68percent]

        # distance between mid points of predicted track and actual track
        pre_mid_point = (pred_exit * pmt_radius + pred_enter * pmt_radius) / 2
        act_mid_point = (act_exit * pmt_radius + act_enter * pmt_radius) / 2
        mid_point_dist = np.linalg.norm(pre_mid_point - act_mid_point, axis=1)

        mid_point_dist_sorted = sorted(mid_point_dist)
        quantile_dist_68 = mid_point_dist_sorted[index_68percent]

        # Draw angle distribution
        if epoch is not None:
            title = f"Angle Distribution (epoch {epoch})"
            filename = (
                f"{self.result_save_path}/reco_angle_distribution_epoch{epoch}.jpg"
            )
        else:
            title = "Angle Distribution"
            filename = (
                f"{self.result_save_path}/{save_prefix}reco_angle_distribution.jpg"
            )

        plt.figure(figsize=(12, 6))
        plt.hist(angles, bins=100, range=(0, 180), alpha=0.7, color="blue")
        plt.title(title)
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Frequency")
        plt.savefig(filename)
        plt.close()

        # Draw midpoints distance distribution
        if epoch is not None:
            title = f"Midpoints Distance Distribution (epoch {epoch})"
            filename = (
                f"{self.result_save_path}/reco_midpoints_distance_epoch{epoch}.jpg"
            )
        else:
            title = "Midpoints Distance Distribution"
            filename = f"{self.result_save_path}/{save_prefix}reco_midpoints_distance_distribution.jpg"

        plt.figure(figsize=(12, 6))
        plt.hist(mid_point_dist, bins=100, alpha=0.7, color="green")
        plt.title(title)
        plt.xlabel("Distance (mm)")
        plt.ylabel("Frequency")
        plt.savefig(filename)
        plt.close()

        return (
            quantile_angle_68,
            quantile_dist_68,
            angles,
            mid_point_dist,
        )


# define loss function
class CustomLoss(nn.Module):
    def __init__(self, w_position=1.0):
        super(CustomLoss, self).__init__()
        self.w_position = w_position

    def forward(self, outputs, labels):
        true_in = labels[:, :3]
        true_out = labels[:, 3:]
        pred_in = outputs[:, :3]
        pred_out = outputs[:, 3:]

        loss_in = ((true_in - pred_in) ** 2).sum(dim=1)  # (batch_size,)
        loss_out = ((true_out - pred_out) ** 2).sum(dim=1)
        total_position_loss = torch.mean(self.w_position * (loss_in + loss_out))

        return total_position_loss


# Dataset Combine
class CombinedDataset(Dataset):
    def __init__(self, pt_files):
        self.datasets = []
        for file in pt_files:
            data = torch.load(file)  # data should be {'x':..., 'y':...}
            x = data["x_data"]
            y = data["y_data"]
            self.datasets.append(TensorDataset(x, y))
        self.combined_dataset = ConcatDataset(self.datasets)

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        return self.combined_dataset[idx]
