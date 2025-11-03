import torch
import torchvision.models as models
from torch.utils.data import Dataset, ConcatDataset, TensorDataset
import numpy as np
import pandas as pd
import subprocess
from matplotlib import pyplot as plt
from TensorPre.MLModel import *
from TensorPre.TensorDataset import *

# from ModelTrain import CombinedDataset
from torch.utils.data import DataLoader
from Fuc import *


class ModelPredict:
    def __init__(self, options):
        self.pmt_radius = 19433.975
        self.fuc = Fuc()
        self.arguments = options
        save_path = f'{options["output_path"]}/{options["mission_name"]}'
        self.result_save_path = f"{save_path}/predict_results"
        result = subprocess.run(f"mkdir -p {self.result_save_path}", shell=True)
        self.model = self.load_model(options["predict_model_path"])
        self.dataset = self.load_dataset(options["pklfile_predict_path"])
        self.predict()
        # self.examine_predict()
        self.save_results()

    def load_dataset(self, dataset_path):
        files = self.fuc.get_file_paths(dataset_path)
        dataset = CombinedDataset(files)
        test_data = DataLoader(dataset, batch_size=32, shuffle=False)
        return test_data

    def load_model(self, model_path):
        model_name = self.arguments["predict_model_name"]
        arguments = self.arguments
        print("Model loaded from", model_path)
        state_dict = torch.load(model_path)
        if self.arguments["pre_method"] == "TensorPre":
            if model_name == "Transformer":
                model = TransformerModel(
                    embed_dim=arguments["embed_dim"],
                    num_heads=arguments["num_heads"],
                    num_layers=arguments["num_layers"],
                    ff_dim=arguments["hidden_dim"],
                    input_dim=arguments["input_dim"],
                )
        else:
            print("No other preprocess method now, use TensorPre to create dataset")
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded")
        return model

    def predict(self):
        model = self.model
        test_loader = self.dataset
        # predict
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        predictions = []
        track_times = []
        totalPEs = []
        # actuals = []

        model.to(device)
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                x_data, track_time, totalPE = data
                x_data = x_data.float().to(device)
                track_time = track_time.to(device)
                totalPE = totalPE.to(device)
                # labels = labels.float().to(device)

                outputs = model(x_data)
                predictions.append(outputs.cpu().numpy())
                track_times.append(track_time.cpu().numpy())
                totalPEs.append(totalPE.cpu().numpy())
                # actuals.append(labels.cpu().numpy())

        self.predictions = np.concatenate(predictions)
        # transform to actural position (previously normalized by pmt_radius)
        self.predictions = self.predictions * self.pmt_radius
        self.track_times = np.concatenate(track_times)
        self.totalPEs = np.concatenate(totalPEs)
        # print(self.predictions)
        # self.actuals = np.concatenate(actuals)

    def examine_predict(self):
        pred_vectors = self.predictions

        # caculate angle between prediction track and z axis
        pre_enter_pos = pred_vectors[:, 0:3]
        pre_exit_pos = pred_vectors[:, 3:6]
        track_direction = pre_exit_pos - pre_enter_pos
        track_direction_norm = np.linalg.norm(track_direction, axis=1)
        z_axis = np.array([0, 0, -1])
        cos_angle = np.dot(track_direction, z_axis) / track_direction_norm

        # caculate distance between prediction track and center
        center = np.array([0, 0, 0])
        distance = (
            np.linalg.norm(
                np.cross(pre_exit_pos - pre_enter_pos, pre_enter_pos - center),
                axis=1,
            )
            / track_direction_norm
        )

        # plot histogram of cos_angle and distance
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(cos_angle, bins=50, alpha=0.7, color="blue")
        plt.xlabel("Cosine of Angle with Z-axis")
        plt.ylabel("Frequency")
        plt.title("Distribution of Cosine of Angle with Z-axis")
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.hist(distance, bins=50, alpha=0.7, color="green")
        plt.xlabel("Distance from Center (mm)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Distance from Center")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.result_save_path}/prediction_examination.png")
        plt.show()

    def save_results(self):
        # Save predictions and track_times
        df = pd.DataFrame(
            {
                "track_time": self.track_times.flatten(),
                "enter_x": self.predictions[:, 0],
                "enter_y": self.predictions[:, 1],
                "enter_z": self.predictions[:, 2],
                "exit_x": self.predictions[:, 3],
                "exit_y": self.predictions[:, 4],
                "exit_z": self.predictions[:, 5],
                "totalPE": self.totalPEs.flatten(),
            }
        )
        self.dataframe = df

        # Save as CSV for easy viewing
        df.to_csv(f"{self.result_save_path}/predict_results.csv", index=False)

        # Also save as numpy arrays for easy loading
        np.savez(
            f"{self.result_save_path}/predict_results.npz",
            predictions=self.predictions,
            track_times=self.track_times,
            totalPEs=self.totalPEs,
        )

        print(f"Results saved in {self.result_save_path}")
        print(f"- CSV file: {self.result_save_path}/predict_results.csv")
        print(f"- NPZ file: {self.result_save_path}/predict_results.npz")

    def get_pred_vectors(self):
        return self.pred_vectors

    def get_act_vectors(self):
        return self.act_vectors

    def get_solar_predirect_angle(self):
        return self.solar_predirect_cos_angle

    def get_solar_actdirect_angle(self):
        return self.solar_actdirect_cos_angle

    def get_edep(self):
        return self.edep

    def get_sun_position(self):
        return self.sun_position

    def get_dataframe(self):
        return self.dataframe


# Dataset Combine
class CombinedDataset(Dataset):
    def __init__(self, pt_files):
        self.datasets = []
        for file in pt_files:
            data = torch.load(file)  # data should be {'x':..., 'y':...}
            x = data["x_data"]
            track_t = data["track_t"]
            totalPE = data["totalPE"]
            self.datasets.append(TensorDataset(x, track_t, totalPE))
        self.combined_dataset = ConcatDataset(self.datasets)

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        return self.combined_dataset[idx]
