import numpy as np
import pandas as pd
import os
# import iminuit  # Commented out due to missing dependency
# from iminuit import Minuit


# ====================================Fuc====================================#
def crystal_ball_function(x, norm, mean, sigma, alpha, n):
    A = (n / np.abs(alpha)) ** n * np.exp(-(alpha**2) / 2)
    B = n / np.abs(alpha) - np.abs(alpha)

    delta_x = (-x + mean) / sigma
    gaussian_part = norm * np.exp(-(delta_x**2) / 2)
    exponential_part = norm * A * (B - delta_x) ** (-n)

    return np.where(delta_x > -alpha, gaussian_part, exponential_part)


class Fuc:
    def __init__(self):
        main_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(main_dir, "..")

        pmt_info = pd.read_csv(os.path.join(base_path, "PMTInfo.csv"), sep="\s+")
        self.pmt_dict = pmt_info.set_index("PMTID")[["X", "Y", "Z"]].to_dict("index")
        self.pmt_type = pmt_info.set_index("PMTID")["Type"]
        self.pmt_2d_dict = pmt_info.set_index("PMTID")[["Theta", "Phi"]].to_dict(
            "index"
        )

    def get_tof(self, ev_xyz, pmt_xyz):
        source_x, source_y, source_z = ev_xyz

        R = 17820
        c = 299.792458
        pmt_x = pmt_xyz[:, 0]
        pmt_y = pmt_xyz[:, 1]
        pmt_z = pmt_xyz[:, 2]

        dist = np.sqrt(
            (source_x - pmt_x) ** 2 + (source_y - pmt_y) ** 2 + (source_z - pmt_z) ** 2
        )
        evt_r0 = np.sqrt(source_x**2 + source_y**2 + source_z**2)
        pmt_r0 = np.sqrt(pmt_x**2 + pmt_y**2 + pmt_z**2)

        cos_theta = (pmt_r0**2 + dist**2 - evt_r0**2) / (2 * pmt_r0 * dist)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        dist_water = pmt_r0 * cos_theta - np.sqrt(R**2 - pmt_r0**2 * np.sin(theta) ** 2)
        dist_LS = dist - dist_water
        tof = (dist_LS * 1.543 / c) + (dist_water * 1.33 / c)
        return tof

    def time_alignment_bin(
        self, time
    ):  # align the starting time of event by searching bin
        counts, bins = np.histogram(time, bins=100, range=(150, 350))
        new_time_point = 0
        if len(time) > 10000:
            conut_threshold = 10
        elif len(time) < 10000 and len(time) > 3000:
            conut_threshold = 8
        else:
            conut_threshold = 6

        for i in range(len(counts)):
            if counts[i] >= conut_threshold:
                new_time_point = bins[i]
                break

        if new_time_point == 0:
            new_time_point = bins[np.argmax(counts)]

        time = time - new_time_point
        return time

    def time_alignment_fit(self, time):  # align the starting time of event by fitting
        counts, bins = np.histogram(time, bins=100, range=(200, 400))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        new_time_point = 0

        def cost_function(norm, mean, sigma, alpha, n):
            y_pred = crystal_ball_function(bin_centers, norm, mean, sigma, alpha, n)
            return np.sum((counts - y_pred) ** 2)

        # m = Minuit(cost_function, norm=300, mean=275, sigma=2.6, alpha=0.8, n=1)  # Commented out - iminuit not available
        # m.migrad()
        # new_time_point = m.values["mean"] - 2 * m.values["sigma"]
        # Use simple method instead
        new_time_point = bins[np.argmax(counts)]

        if new_time_point == 0:
            new_time_point = bins[np.argmax(counts)]

        time = time - new_time_point
        return time

    def time_alignment(self, time, method="fit"):
        if method == "bin":
            return self.time_alignment_bin(time)
        elif method == "fit":
            return self.time_alignment_fit(time)
        else:
            return time

    def calculate_cos_theta(self, pmt_positions, vertex_position, momentum):
        vectors = pmt_positions - vertex_position
        norms = np.linalg.norm(vectors, axis=1)
        normed_vectors = vectors / norms[:, np.newaxis]
        momentum_norm = momentum / np.linalg.norm(momentum)
        cos_theta = np.dot(normed_vectors, momentum_norm)
        return cos_theta

    def project_to_image(self, pos, image_size):
        x = int((pos[0] + np.pi) / (2 * np.pi) * image_size)
        y = int((pos[1] + np.pi / 2) / np.pi * image_size)
        return x, y

    def compute_track_points(self, ref_pos, direction):
        transfer_z = 26452  # mm
        ref_pos = [
            ref_pos[0],
            ref_pos[1],
            ref_pos[2] #+ transfer_z,
        ]  # shift the origin to the center of the sphere
        direction = direction / np.linalg.norm(direction)
        intersection1, intersection2 = self.find_sphere_intersection(
            sphere_center=(0, 0, 0),
            #sphere_radius=19433.975,
            sphere_radius= 17700.000,
            point=ref_pos,
            direction=direction,
        )
        return intersection1, intersection2

    def find_sphere_intersection(self, sphere_center, sphere_radius, point, direction):
        x0, y0, z0 = sphere_center
        r = sphere_radius
        x1, y1, z1 = point
        dx, dy, dz = direction
        a = dx**2 + dy**2 + dz**2
        b = 2 * (dx * (x1 - x0) + dy * (y1 - y0) + dz * (z1 - z0))
        c = (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2 - r**2

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return None, None

        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)

        # t = max(t1, t2)
        # if t < 0:
        #     return None

        # intersection = (x1 + t * dx, y1 + t * dy, z1 + t * dz)
        intersection1 = (x1 + t1 * dx, y1 + t1 * dy, z1 + t1 * dz)
        intersection2 = (x1 + t2 * dx, y1 + t2 * dy, z1 + t2 * dz)
        return intersection1, intersection2

    def mercator_projection(self, positions):
        x = np.arctan2(positions[:, 1], positions[:, 0])
        y = np.arcsinh(positions[:, 2] / np.linalg.norm(positions[:, :2], axis=1))
        return np.vstack((x, y)).T

    def inverse_mercator_projection(self, positions_2d, radius):
        lon = positions_2d[:, 0]
        lat = np.arcsin(np.tanh(positions_2d[:, 1]))
        x = radius * np.cos(lat) * np.cos(lon)
        y = radius * np.cos(lat) * np.sin(lon)
        z = radius * np.sin(lat)
        return np.vstack((x, y, z)).T

    def pixel_to_latlon(self, cherenkov_center_x, cherenkov_center_y, image_size):
        lon = cherenkov_center_x / image_size * 2 * np.pi - np.pi
        lat = cherenkov_center_y / image_size * np.pi - np.pi / 2
        return lon, lat

    def center_to_direction(
        self, predictions, vertex_positions, momentum_list, pre_method, image_size
    ):

        if pre_method == "PixelPre":
            predictions = np.array(
                [self.pixel_to_latlon(x, y, image_size) for x, y in predictions]
            )

        predicted_3d = self.inverse_mercator_projection(predictions, radius=19433.975)

        predicted_directions = predicted_3d - vertex_positions
        actual_directions = momentum_list

        pred_vectors = predicted_directions
        act_vectors = actual_directions
        return pred_vectors, act_vectors

    def search_pmt(self, pmt_ids):
        result = []
        for pmt_id in pmt_ids:
            coords = self.pmt_dict.get(pmt_id, {"X": np.nan, "Y": np.nan, "Z": np.nan})
            result.append([coords["X"], coords["Y"], coords["Z"]])
        result_array = np.array(result)
        return result_array

    def search_pmt_pos(self, pmt_id):
        coords = self.pmt_dict.get(pmt_id, {"X": np.nan, "Y": np.nan, "Z": np.nan})
        return np.array([coords["X"], coords["Y"], coords["Z"]])

    def search_pmt_2dpos(self, pmt_id):
        coords = self.pmt_2d_dict.get(pmt_id, {"Theta": np.nan, "Phi": np.nan})
        return np.array([coords["Theta"], coords["Phi"]])

    def is_pmttype(self, pmt_id):
        if pmt_id not in self.pmt_type.index:
            return None
        else:
            return self.pmt_type[pmt_id]

    def time_smearing(self, time, sigma):
        nosie = np.random.normal(0, sigma, len(time))
        time = time + nosie
        return time

    def get_solar_position(self, timestamp):
        solar_position = np.array(0, 0, 0)
        return solar_position

    def get_file_paths(self, directory):
        file_paths = []

        for root, directories, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)

        return file_paths
