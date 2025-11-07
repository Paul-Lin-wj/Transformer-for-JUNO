import numpy as np


class selector_num:
    def __init__(self, time_notof_list, min_time, keep_ph_num):
        mask = time_notof_list > min_time
        filter_time_notof_list = time_notof_list[mask]
        ori_index = np.where(mask)[0]
        sorted_indices = np.argsort(filter_time_notof_list)
        top_100_indices = sorted_indices[:keep_ph_num]
        select_indices = ori_index[top_100_indices]
        self.select_indices = select_indices

    def get_select_indices(self):
        return self.select_indices


class TimeCutPoint:
    def __init__(self, time_list, ph_num_threshold=100, time_lolimits=-5.0):
        mask = time_list >= time_lolimits
        time_list = time_list[mask]
        self.time_list = np.array(time_list)
        self.ph_num_threshold = ph_num_threshold
        self.time_lolimits = time_lolimits

    def search_best_time_cut(self):
        sorted_time_list = np.sort(self.time_list)

        if len(sorted_time_list) >= self.ph_num_threshold:
            time_cut_point = sorted_time_list[self.ph_num_threshold - 1]
        else:
            time_cut_point = sorted_time_list[-1]

        return time_cut_point
