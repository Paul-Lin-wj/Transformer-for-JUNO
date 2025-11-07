import numpy as np
import os
import pickle
import subprocess
import ROOT
import torch
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
import sys
from bisect import bisect_left
from matplotlib import pyplot as plt

sys.path.append("..")
from concurrent.futures import ProcessPoolExecutor
from Fuc import Fuc
from HitSelection import selector_num


class HitTokenizer:
    def __init__(self):
        pass

    def tokenize(self, hits, hit_length=17612):
        embeddings = []

        # for hit in hits:
        #     position = hit
        #     embedding = list(position)
        #     embeddings.append(embedding)

        for hit in hits:
            time, charge, position = hit
            embedding = [time, charge] + list(position)
            embeddings.append(embedding)

        # max length padding
        if len(embeddings) < hit_length:
            embeddings += [[0] * len(embeddings[0])] * (hit_length - len(embeddings))
        elif len(embeddings) > hit_length:
            embeddings = embeddings[:hit_length]

        return embeddings


# =================dataset to train model===================
class EvtDataset(Dataset):
    def __init__(self, file=[], options={}):
        ROOT.gSystem.Load("libEDMUtil")
        self.max_hits = options["max_hits"]
        self.label_type = options["label_type"]
        self.file_names = file
        self.fuc = Fuc()
        (
            self.x_data,
            self.y_data,
            self.track_time_list,
            self.totalPE_list,
        ) = self.prepare_data()[:4]
        print("X_data shape:", self.x_data.shape)
        if len(self.x_data) > 0:
            print("x_data first example:", self.x_data[0])
        else:
            print("WARNING: No correlated muon events found in this file!")
        if len(self.y_data) > 0:
            print("Y_data shape:", self.y_data.shape)
            print("y_data first example:", self.y_data[0])

    def prepare_data(self):
        embedding_list = []
        label_list = []
        track_t_list = []
        totalpe_list = []
        #pmt_radius = 19433.975
        pmt_radius = 17700.000
        

        pmt_edm_file = ROOT.TFile.Open(self.file_names[0], "READ")
        track_user_file = ROOT.TFile.Open(self.file_names[1], "READ")
        navi_tree = pmt_edm_file.Get("Meta/navigator")
        pmt_tree = pmt_edm_file.Get("Event/CdLpmtCalib/CdLpmtCalibEvt")
        navi_evt = ROOT.JM.EvtNavigator()
        pmt_evt = ROOT.JM.CdLpmtCalibEvt()
        navi_tree.SetBranchAddress("EvtNavigator", ROOT.AddressOf(navi_evt))
        pmt_tree.SetBranchAddress("CdLpmtCalibEvt", ROOT.AddressOf(pmt_evt))
        total_entries = pmt_tree.GetEntries()

        # construct muon track map (same as before) and sorted list of times
        if self.label_type == "Wp":
            muon_track_map, correlation_map, track_times = self.Wp_muon_map_construct(
                track_user_file
            )
        elif self.label_type == "TT":
            muon_track_map, correlation_map, track_times = self.TT_muon_map_construct(
                track_user_file
            )
        else:
            raise ValueError("Invalid label_type. Use 'Wp' or 'TT'.")
        print("Total muon tracks:", len(track_times))

        # window (ns)
        TIME_WINDOW = 800  # 1 second window to allow for timestamp mismatches

        pmt_count = 0
        # first step: loop navi events to record event count
        for count in range(navi_tree.GetEntries()):
            navi_tree.GetEntry(count)
            detector_type = navi_evt.getDetectorType()
            if detector_type != 0:
                continue
            # find closest track time using bisect
            navi_evt_t = (
                navi_evt.TimeStamp().GetNanoSec()
                + navi_evt.TimeStamp().GetSec() * 1000000000
            )
            idx = bisect_left(track_times, navi_evt_t)
            candidate = None
            min_diff = None
            # check idx and idx-1 as possible nearest neighbors
            for check_idx in (idx - 1, idx):
                if 0 <= check_idx < len(track_times):
                    t = track_times[check_idx]
                    d = abs(navi_evt_t - t)
                    if d <= TIME_WINDOW:
                        if (min_diff is None) or (d < min_diff):
                            min_diff = d
                            candidate = t

            if candidate is not None:
                correlation_map[candidate].append((count, pmt_count))

            pmt_count += 1

        # grouped pmt event count
        for k, v in correlation_map.items():
            print(f"Track time {k} has {len(v)} correlated pmt events.")
            track_t = k
            track_data = muon_track_map[track_t]
            print(f"Track data: {track_data}")
            norm_track = np.array(track_data) #/ pmt_radius
            hits = []
            totalPE = 0.0
            used_pmt = set()
            for c_idx, p_idx in v:
                navi_tree.GetEntry(c_idx)
                pmt_tree.GetEntry(p_idx)
                cur_evt_t = (
                    navi_evt.TimeStamp().GetNanoSec()
                    + navi_evt.TimeStamp().GetSec() * 1000000000
                )
                for pmt in pmt_evt.calibPMTCol():
                    pmt_id = pmt.pmtId()
                    pmt_q = pmt.sumCharge()
                    totalPE += pmt_q
                    if pmt_id in used_pmt:
                        continue
                    used_pmt.add(pmt_id)
                    pmt_pos = self.fuc.search_pmt_pos(pmt_id)
                    pmt_2d_pos = self.fuc.search_pmt_2dpos(pmt_id)
                    norm_pmt_pos = pmt_pos / pmt_radius
                    t = pmt.time()[0]  # first hit
                    abs_t = cur_evt_t + int(t)
                    log_q = np.log10(pmt_q + 1)
                    hits.append((abs_t, t, log_q, norm_pmt_pos, pmt_id, pmt_2d_pos))

            print(f"Total hits for track at time {track_t}: {len(hits)}")

            if len(hits) <= 10000:
                continue

            # process hits: sort, truncate, -
            if len(hits) > self.max_hits:
                hits.sort(key=lambda x: x[0])  # 直接按时间升序排序
                #self.max_hits = len(hits)
                n = self.max_hits
                n_first = int(round(0.6 * n))
                n_last = n - n_first
                selected_hits = hits[:n_first]   + hits[-n_last:]          #########  time_check_point
                hits = selected_hits

                # # draw histograms for analysis
                # self.draw_hist(hits, track_t)

                # 3. 归一化和组装
                abs_times = np.array([h[0] for h in hits], dtype=np.uint64)
                charges = np.array([h[2] for h in hits], dtype=np.float32)
                norm_times = abs_times - abs_times.min()
                if np.std(norm_times) > 1e-6:
                    norm_times = norm_times / np.std(norm_times)
                else:
                    norm_times[:] = 0.0
                charges = np.clip(charges, 0, np.percentile(charges, 99))
                if charges.max() > charges.min():
                    norm_charges = (charges - charges.min()) / (
                        charges.max() - charges.min()
                    )
                else:
                    norm_charges = np.zeros_like(charges)
                token_hits = [
                    (norm_times[i], norm_charges[i], hits[i][3])
                    # (hits[i][3])
                    for i in range(len(hits))
                ]
            else:
                # 只有1个hit
                token_hits = [(0.0, 0.0, hits[0][3])]

            # tokenize and append
            embedding = HitTokenizer().tokenize(token_hits, hit_length=self.max_hits)
            label = norm_track
            embedding_list.append(embedding)
            label_list.append(label)
            track_t_list.append(track_t)
            totalpe_list.append(totalPE)

        print("Total correlated muon events:", len(embedding_list))

        # transform to tensor
        x_data = torch.tensor(np.array(embedding_list), dtype=torch.float32)
        y_data = torch.tensor(np.array(label_list), dtype=torch.float32)
        track_t_list = torch.tensor(np.array(track_t_list), dtype=torch.uint64)
        totalPE = torch.tensor(np.array(totalpe_list), dtype=torch.float32)

        return x_data, y_data, track_t_list, totalPE, len(embedding_list)

    def Wp_muon_map_construct(self, track_user_file):
        track_tree = track_user_file.Get("WpMuonClassifyRecTool")
        muon_track_map = {}
        correlation_map = {}
        for count in range(track_tree.GetEntries()):
            track_tree.GetEntry(count)
            totalPE = track_tree.totalPE
            track_t = track_tree.time
            enter_x = track_tree.recoenterx
            enter_y = track_tree.recoentery
            enter_z = track_tree.recoenterz
            exit_x = track_tree.recoexitx
            exit_y = track_tree.recoexity
            exit_z = track_tree.recoexitz
            recoclassid = track_tree.recoclassid
            if recoclassid == 1:  # only for through-going muon (single muon)
                # print("Find muon event with totalPE:", totalPE)
                muon_track_map[track_t] = (
                    enter_x[0],
                    enter_y[0],
                    enter_z[0],
                    exit_x[0],
                    exit_y[0],
                    exit_z[0],
                )
                correlation_map[track_t] = []

        track_times = sorted(muon_track_map.keys())
        # print("Total muon track:", len(track_times))
        return muon_track_map, correlation_map, track_times

    def TT_muon_map_construct(self, track_user_file):
        track_tree = track_user_file.Get("TT_selected")
        if not track_tree:
            # Try alternative name
            track_tree = track_user_file.Get("TT")
        muon_track_map = {}
        correlation_map = {}
        for count in range(track_tree.GetEntries()):
            track_tree.GetEntry(count)
            n_track = 1  # track_tree.NTracks
            n_points = 3  #track_tree.NTotPoints

            # Handle different timestamp formats
            if hasattr(track_tree, 'start_TS'):
                track_t = (
                    track_tree.start_TS.GetSec() * 1000000000
                    + track_tree.start_TS.GetNanoSec()
                )
            elif hasattr(track_tree, 'fSec') and hasattr(track_tree, 'fNanoSec'):
                track_t = track_tree.fSec * 1000000000 + track_tree.fNanoSec
            else:
                continue

            # Check if we have multiple tracks or single track
            if n_track == 1 and n_points >= 3:
                # Process single track
                # Check if Coeff0 is an array or single value
                coeff0 = track_tree.Coeff0
                if hasattr(coeff0, '__len__') and len(coeff0) > 0:
                    # Coeff0 is an array
                    ref_pos = [coeff0[0], track_tree.Coeff1[0], track_tree.Coeff2[0]]
                    direction = [track_tree.Coeff3[0], track_tree.Coeff4[0], track_tree.Coeff5[0]]
                else:
                    # Coeff0 is a single value
                    ref_pos = [coeff0, track_tree.Coeff1, track_tree.Coeff2]
                    direction = [track_tree.Coeff3, track_tree.Coeff4, track_tree.Coeff5]

                intersection1, intersection2 = self.fuc.compute_track_points(
                    ref_pos, direction
                )

                if intersection1 is None or intersection2 is None:  # track must be through-going
                    continue

                intersection1 = list(intersection1)
                intersection2 = list(intersection2)

                intersection1[0] = track_tree.Coeff0
                intersection1[1] = track_tree.Coeff1
                intersection1[2] = track_tree.Coeff2
                intersection2[0] = track_tree.Coeff0 + track_tree.Coeff3
                intersection2[1] = track_tree.Coeff1 + track_tree.Coeff4
                intersection2[2] = track_tree.Coeff2 + track_tree.Coeff5

                intersection1 = tuple(intersection1)
                intersection2 = tuple(intersection2)

                # Store track information: entry/exit points
                muon_track_map[track_t] = (
                    intersection1[0],  # entry x
                    intersection1[1],  # entry y
                    intersection1[2],  # entry z
                    intersection2[0],  # exit x
                    intersection2[1],  # exit y
                    intersection2[2],  # exit z
                    0.0,  # placeholder for second track entry x
                    0.0,  # placeholder for second track entry y
                    0.0,  # placeholder for second track entry z
                    0.0,  # placeholder for second track exit x
                    0.0,  # placeholder for second track exit y
                    0.0,  # placeholder for second track exit z
                )
                correlation_map[track_t] = []

        track_times = sorted(muon_track_map.keys())
        # print("Total muon track:", len(track_times))
        return muon_track_map, correlation_map, track_times

    def draw_hist(self, hits, track_t=0):
        # draw npe histogram
        npe_list = [h[2] for h in hits]
        plt.hist(npe_list, bins=50, range=(0, np.percentile(npe_list, 99)))
        plt.xlabel("PMT Charge (PE)")
        plt.ylabel("Counts")
        plt.title("PMT Charge Distribution for Muon Event")
        plt.grid(True)
        plt.savefig(f"figures/npe_hist_event_{track_t}.png")
        plt.clf()
        plt.close()
        # draw npe vs pmt_id
        pmt_ids = [h[4] for h in hits]
        plt.scatter(pmt_ids, npe_list, alpha=0.6)
        plt.xlabel("PMT ID")
        plt.ylabel("PMT Charge (PE)")
        plt.title("PMT Charge vs PMT ID for Muon Event")
        plt.grid(True)
        plt.savefig(f"figures/npe_vs_pmtid_event_{track_t}.png")
        plt.clf()
        plt.close()
        # draw 2d-hist theta vs phi (npe as color)
        thetas = [h[5][0] for h in hits]
        cos_thetas = np.cos(thetas)
        phis = [h[5][1] for h in hits]

        plt.figure(figsize=(8, 6))
        plt.hist2d(cos_thetas, phis, bins=50, weights=npe_list, cmap="viridis")
        plt.colorbar(label="Total PMT Charge (PE)")
        plt.xlabel("Cos(Theta)")
        plt.ylabel("Phi (radians)")
        plt.title("PMT Charge Distribution on Sphere for Muon Event")
        plt.grid(True)
        plt.savefig(f"figures/npe_2dhist_event_{track_t}.png")
        plt.clf()
        plt.close()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], (
            self.y_data[idx],
            self.track_time_list[idx],
            self.totalPE_list[idx],
        )


# ================ create dataset and save as pickle file =================
class DatasetCreate:
    def __init__(self, options):
        self.options = options

        # 打印数据集创建信息
        print("=" * 70)
        print("DATASET CREATION CONFIGURATION")
        print("=" * 70)
        print(f"Output file: {options['output_file']}")
        print(f"Track file: {options['track_file']}")
        print(f"Hits file: {options.get('hits_file', 'Not specified')}")
        print(f"Hits file list: {options.get('hits_file_list', 'Not specified')}")
        print(f"Max hits per event: {options['max_hits']}")
        print(f"Label type: {options.get('label_type', 'Not specified')}")

        # 检查必要文件
        if os.path.exists(options['track_file']):
            track_size = os.path.getsize(options['track_file']) / (1024*1024)
            print(f"\nTrack file size: {track_size:.2f} MB")
        else:
            print(f"\n⚠️  WARNING: Track file not found: {options['track_file']}")

        if options.get('hits_file') and os.path.exists(options['hits_file']):
            hits_size = os.path.getsize(options['hits_file']) / (1024*1024)
            print(f"Hits file size: {hits_size:.2f} MB")
        elif options.get('hits_file_list') and os.path.exists(options['hits_file_list']):
            with open(options['hits_file_list'], 'r') as f:
                file_count = len([line for line in f if line.strip()])
            print(f"File list contains {file_count} files")

        print("=" * 70)

        self.output_file = options["output_file"]
        self.track_file = options["track_file"]
        self.hits_file = options.get("hits_file")
        self.hits_file_list = options.get("hits_file_list")
        self.max_hits = options["max_hits"]

        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        self.create_dataset()

    def create_dataset(self):
        # Initialize files_to_process list
        files_to_process = []

        # Check if hits_file_list is provided
        if self.hits_file_list:
            # Load hits files from the list
            with open(self.hits_file_list, 'r') as f:
                hits_files = [line.strip() for line in f.readlines() if line.strip()]
            hits_files.sort()  # Sort files alphabetically
            print(f"Found {len(hits_files)} hits files in list: {self.hits_file_list}")

            if not hits_files:
                raise ValueError(f"No hits files found in list: {self.hits_file_list}")

            files_to_process = hits_files

        elif os.path.isdir(self.hits_file):
            # Get all .esd files in the directory
            esd_files = [os.path.join(self.hits_file, f) for f in os.listdir(self.hits_file) if f.endswith('.esd')]
            esd_files.sort()  # Sort files alphabetically
            print(f"Found {len(esd_files)} .esd files in directory: {self.hits_file}")

            if not esd_files:
                raise ValueError(f"No .esd files found in directory: {self.hits_file}")

            files_to_process = esd_files

        else:
            # Original single file processing
            file = [self.hits_file, self.track_file]
            dataset = EvtDataset(file, self.options)
            torch.save(
                {
                    "x_data": dataset.x_data,
                    "y_data": dataset.y_data,
                    "track_t": dataset.track_time_list,
                    "totalPE": dataset.totalPE_list,
                },
                self.output_file,
            )
            print("Dataset saved to", self.output_file)
            return

        # Initialize counters
        processed_files = 0
        skipped_files = 0
        saved_files = 0
        total_events = 0
        total_hits = 0
        total_pe = 0

        print(f"\n" + "=" * 70)
        print("PROCESSING FILES")
        print("=" * 70)
        print("File [progress]    | Events | Hits   | PE      | Output")
        print("-" * 70)

        # Process each hits file and save separately
        for i, hits_file in enumerate(files_to_process):
            filename = os.path.basename(hits_file)

            file = [hits_file, self.track_file]

            # Create dataset and check event count
            dataset = EvtDataset(file, self.options)
            event_count = len(dataset.x_data)

            processed_files += 1

            # 统计数据信息
            event_hits = [len(d) for d in dataset.x_data] if hasattr(dataset, 'x_data') else []
            event_pe = dataset.totalPE_list if hasattr(dataset, 'totalPE_list') else []
            avg_hits = sum(event_hits) / len(event_hits) if event_hits else 0
            total_hits_file = sum(event_hits)
            total_pe_file = sum(event_pe)

            total_events += event_count
            total_hits += total_hits_file
            total_pe += total_pe_file

            # Check if this file has any correlated muon events
            if event_count == 0:
                print(f"{filename[:15]:15} [{i+1:03d}/{len(files_to_process):03d}] | 0      | 0      | 0      | SKIPPED (no events)")
                skipped_files += 1
                continue

            # Create numbered output filename (001, 002, 003, ...)
            dirname = os.path.dirname(self.output_file)
            basename = os.path.basename(self.output_file)
            name, ext = os.path.splitext(basename)
            numbered_filename = f"{saved_files+1:03d}_{name}{ext}"
            numbered_output_file = os.path.join(dirname, numbered_filename + ".pt")

            # Save individual dataset
            torch.save(
                {
                    "x_data": dataset.x_data,
                    "y_data": dataset.y_data,
                    "track_t": dataset.track_time_list,
                    "totalPE": dataset.totalPE_list,
                },
                numbered_output_file,
            )
            saved_files += 1

            # 格式化输出
            print(f"{filename[:15]:15} [{i+1:03d}/{len(files_to_process):03d}] | {event_count:6d} | {total_hits_file:6d} | {total_pe_file:7.0f} | {os.path.basename(numbered_output_file)}")

        print("-" * 70)
        print("PROCESSING SUMMARY:")
        print("-" * 70)
        print(f"  - Total files found: {len(files_to_process)}")
        print(f"  - Files processed: {processed_files}")
        print(f"  - Files skipped (zero events): {skipped_files}")
        print(f"  - Files saved: {saved_files}")
        print(f"\n  - Total events: {total_events:,}")
        print(f"  - Total hits: {total_hits:,}")
        print(f"  - Total PE: {total_pe:,.0f}")
        if total_events > 0:
            print(f"\n  - Average hits per event: {total_hits/total_events:.1f}")
            print(f"  - Average PE per event: {total_pe/total_events:.1f}")
        print("=" * 70)


# Dataset Combine - Memory efficient version
class CombinedDataset(Dataset):
    def __init__(self, pt_files):
        self.pt_files = pt_files
        self.file_indices = []
        self.dataset_lengths = []
        self.cumulative_lengths = [0]
        self._all_data = None
        self._all_targets = None
        self._loaded = False

        print(f"Loading {len(pt_files)} dataset files...", flush=True)
        print("-" * 98, flush=True)
        print("File [progress]    | Samples | Size   | Input Shape      | Target Shape | Data Type       | Status", flush=True)
        print("-" * 98, flush=True)

        total_samples = 0
        total_size = 0
        all_data_list = []
        all_targets_list = []

        # Add progress ticker for large datasets
        import time
        start_time = time.time()
        last_progress_time = start_time

        for i, file_path in enumerate(pt_files):
            # Show progress every 10 files or every 30 seconds
            current_time = time.time()
            if i > 0 and (i % 10 == 0 or current_time - last_progress_time > 30):
                elapsed = current_time - start_time
                progress = (i + 1) / len(pt_files) * 100
                print(f"\nProgress: {i+1}/{len(pt_files)} files ({progress:.1f}%) - {elapsed:.1f}s elapsed", flush=True)
                last_progress_time = current_time
            try:
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    print(f"{os.path.basename(file_path)[:15]:15} [{i+1:03d}/{len(pt_files):03d}] | -      | -      | -               | -            | -               | ERROR: File not found", flush=True)
                    continue

                # 获取文件大小
                file_size = os.path.getsize(file_path) / (1024*1024)  # MB

                # 加载数据，使用map_location减少内存使用
                try:
                    data = torch.load(file_path, map_location='cpu')
                except Exception as e:
                    print(f"{os.path.basename(file_path)[:15]:15} [{i+1:03d}/{len(pt_files):03d}] | -      | {file_size:6.1f} | -               | -            | -               | ERROR: {str(e)[:50]}", flush=True)
                    continue

                # 验证数据格式
                if not isinstance(data, dict) or 'x_data' not in data or 'y_data' not in data:
                    print(f"{os.path.basename(file_path)[:15]:15} [{i+1:03d}/{len(pt_files):03d}] | -      | {file_size:6.1f} | -               | -            | -               | ERROR: Invalid data format", flush=True)
                    continue

                x = data["x_data"]
                y = data["y_data"]

                # 验证数据形状
                if len(x) != len(y):
                    print(f"{os.path.basename(file_path)[:15]:15} [{i+1:03d}/{len(pt_files):03d}] | -      | {file_size:6.1f} | -               | -            | -               | ERROR: Shape mismatch", flush=True)
                    continue

                # 检查数据是否为空
                if len(x) == 0:
                    print(f"{os.path.basename(file_path)[:15]:15} [{i+1:03d}/{len(pt_files):03d}] | 0      | {file_size:6.1f} | -               | -            | -               | WARNING: Empty file", flush=True)
                    continue

                # 累积数据而不是创建TensorDataset
                all_data_list.append(x)
                all_targets_list.append(y)
                self.file_indices.append(i)
                dataset_length = len(x)
                self.dataset_lengths.append(dataset_length)
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + dataset_length)

                total_samples += dataset_length
                total_size += file_size

                # Get shape and type information
                input_shape = str(list(x.shape[1:])) if len(x.shape) > 1 else str(x.shape[0])
                target_shape = str(list(y.shape[1:])) if len(y.shape) > 1 else str(y.shape[0])
                input_type = str(x.dtype).split('.')[-1]
                target_type = str(y.dtype).split('.')[-1]

                # Format shape strings to fit column width
                input_shape_str = input_shape[:15].ljust(15)
                target_shape_str = target_shape[:12].ljust(12)
                type_str = f"{input_type}/{target_type}"[:15].ljust(15)

                print(f"{os.path.basename(file_path)[:15]:15} [{i+1:03d}/{len(pt_files):03d}] | {dataset_length:6d} | {file_size:6.1f} | {input_shape_str} | {target_shape_str} | {type_str} | OK", flush=True)

                # 定期清理内存
                if i % 10 == 0:
                    import gc
                    gc.collect()

            except Exception as e:
                print(f"{os.path.basename(file_path)[:15]:15} [{i+1:03d}/{len(pt_files):03d}] | -      | -      | -               | -            | -               | ERROR: {str(e)[:50]}", flush=True)
                continue

        if len(all_data_list) == 0:
            raise ValueError("No valid dataset files found!")

        # 一次性合并所有数据（更高效）
        print("\nMerging loaded data...", flush=True)

        # Show progress during merging
        print("Merging progress:", end="", flush=True)
        self._all_data = torch.cat(all_data_list, dim=0)
        print(" [DONE]", flush=True)
        print("Merging targets:", end="", flush=True)
        self._all_targets = torch.cat(all_targets_list, dim=0)
        print(" [DONE]", flush=True)

        self._loaded = True

        # 清理列表释放内存
        del all_data_list
        del all_targets_list
        import gc
        gc.collect()

        print("-" * 98, flush=True)
        print(f"SUMMARY: Successfully loaded {len(self.file_indices)}/{len(pt_files)} files", flush=True)
        print(f"Total samples: {total_samples:,}", flush=True)
        print(f"Total size: {total_size:.1f} MB", flush=True)
        print("=" * 98, flush=True)

    def __len__(self):
        if not self._loaded:
            return 0
        return len(self._all_data)

    def __getitem__(self, idx):
        if not self._loaded:
            raise RuntimeError("Dataset not loaded")
        return self._all_data[idx], self._all_targets[idx]

    def get_dataset_info(self):
        """获取数据集详细信息"""
        info = {
            'total_files': len(self.pt_files),
            'loaded_files': len(self.file_indices),
            'total_samples': len(self._all_data) if self._loaded else 0,
            'file_details': []
        }

        for i, (file_idx, length) in enumerate(zip(self.file_indices, self.dataset_lengths)):
            file_path = self.pt_files[file_idx]
            info['file_details'].append({
                'file_name': os.path.basename(file_path),
                'file_path': file_path,
                'samples': length,
                'size_mb': os.path.getsize(file_path) / (1024*1024)
            })

        return info
