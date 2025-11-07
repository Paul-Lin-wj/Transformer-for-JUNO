class args_set:
    def __init__(self):
        self.options = {
            "mission_name": "test",
            "output_path": "../output",
            "solar_analysis": False,
            "pre_method": "TensorPre",
            # ==================DatasetCreate options==================
            "Createdataset": False,
            "label_type": "Wp",
            "track_file": "",
            "hits_file": "",
            "only_xdata": False,
            "output_file": "",
            # "select_pmt": "Hamamatsu",
            "max_hits": 500,
            "muon_pe_threshold": 10000,
            # ==================ModelTrain options==================
            "TrainModel": False,
            "send_email": None,
            "GPUid": "0",
            # Transformer
            "patch_size": 4,
            "embed_dim": 128,
            "num_heads": 4,
            "num_layers": 2,
            "hidden_dim": 128,
            "input_dim": 3,
            # Parameters for training
            "num_epochs": 150,
            "train_test_num": -1,
            "test_size": 0.05,
            "learning_rate": 1e-3,
            "scheduler_step": 150,
            "pklfile_train_path": "",
            "batch_size": 64,
            # ==================ModelPredict options==================
            "PredictModel": False,
            "predict_model_path": "model_path",
            "predict_model_name": "ResNet",
            "pklfile_predict_path": "dataset_path",
        }

    def get_args(self):
        return self.options

    def set_args(self, options):
        self.options = options

    def get_args_value(self, key):
        return self.options[key]

    def set_args_value(self, key, value):
        self.options[key] = value

    def contains_key(self, key):
        return key in self.options

    def update(self, options):
        self.options.update(options)
