import argparse
from ShareArgs import args_set
from ModelTrain import ModelTrain
from TensorPre.TensorDataset import DatasetCreate as TensorDatasetCreate
from ModelPredict import ModelPredict


class RunModule:
    def __init__(self, run_options):
        self.args_set = run_options["direct_reco"]

        print("Top mission name: ", self.args_set.options["mission_name"])

    def run(self):
        if self.args_set.options["Createdataset"]:
            if self.args_set.options["pre_method"] == "TensorPre":
                print("Start to create dataset, Dataset format is Tensor")
                TensorDatasetCreate(self.args_set.options)
            else:
                print("No other preprocess method now, use TensorPre to create dataset")
        elif self.args_set.options["TrainModel"]:
            if self.args_set.options.get("ModelTuning", False):
                print("Start to fine-tune model")
            else:
                print("Start to train model")

            # Build DSA config if enabled
            if self.args_set.options.get("dsa_enabled", False):
                dsa_config = {
                    "sparsity_ratio": self.args_set.options.get("sparsity_ratio", 0.1),
                    "target_sparsity": self.args_set.options.get("target_sparsity", 0.05),
                    "adaptive_threshold": True,
                    "min_connections": 5,
                    "warmup_epochs": 10,
                    "schedule_type": "adaptive"
                }
                self.args_set.options["dsa_config"] = dsa_config
                print(f"DSA enabled with config: {dsa_config}")

            ModelTrain(self.args_set.options)
        elif self.args_set.options["Predict"]:
            print("Start to predict")
            ModelPredict(self.args_set.options)
        else:
            print("Please set the correct options of run module")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RunModule")

    parser.add_argument("--mission_name", type=str, default="test")
    parser.add_argument("--output_path", type=str, default="../output")
    # Predict method
    parser.add_argument("--pre_method", type=str, default="TensorPre")
    # ==================DatasetCreate options==================
    parser.add_argument(
        "--Createdataset",
        action="store_true",
        default=False,
        help="Create dataset module",
    )
    parser.add_argument(
        "--label_type",
        type=str,
        default="Wp",
        help="label type: Wp, TT",
    )
    parser.add_argument(
        "--track_file",
        type=str,
        help="input file path, if isCreatedataset is True, input_file_path is required",
    )
    parser.add_argument("--hits_file", type=str, help="input_file of hit info")
    parser.add_argument("--hits_file_list", type=str, help="list file containing paths to hits files")
    parser.add_argument("--output_file", type=str, help="output pkl file path")
    parser.add_argument("--only_xdata", action="store_true", default=False)
    # parser.add_argument("--select_pmt", type=str, default="Hamamatsu")
    parser.add_argument("--max_hits", type=int, default=500)
    parser.add_argument("--muon_pe_threshold", type=int, default=10000)

    # ==================ModelTrain options==================
    # Model Train
    parser.add_argument(
        "--TrainModel",
        action="store_true",
        default=False,
        help="run model train module",
    )
    # Model Fine-tuning
    parser.add_argument(
        "--ModelTuning",
        action="store_true",
        default=False,
        help="run model fine-tuning module",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="path to pretrained model for fine-tuning",
    )

    parser.add_argument(
        "--send_email",
        type=str,
        default=None,
        help="send mail to the email address",
    )
    # Parameters for image
    parser.add_argument("--channels", type=int, default=6, help="detsim:5 fullsim:6")

    # Parameters for model
    parser.add_argument(
        "--train_model_name",
        type=str,
        default="CRCP",
        help="VGG16  CRCP  VisTransformer Transformer ResNet",
    )
    # VisionTransformer
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--input_dim", type=int, default=3)

    # DSA (Dynamic Sparse Attention) parameters
    parser.add_argument(
        "--dsa_enabled",
        action="store_true",
        default=False,
        help="Enable DSA (Dynamic Sparse Attention)"
    )
    parser.add_argument("--sparsity_ratio", type=float, default=0.1, help="Initial sparsity ratio for DSA")
    parser.add_argument("--target_sparsity", type=float, default=0.05, help="Target sparsity for DSA")
    parser.add_argument("--sparsity_weight", type=float, default=0.001, help="Sparsity regularization weight")
    parser.add_argument("--entropy_weight", type=float, default=0.0001, help="Entropy regularization weight")

    # Parameters for training
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--GPUid", type=str, default="0")
    parser.add_argument(
        "--train_test_num",
        type=int,
        default=-1,
        help="num of data for train and test, if -1, use all data",
    )
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--scheduler_step", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_every", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--run_timestamp", type=str, default=None, help="Run timestamp for creating unique output directory")
    parser.add_argument(
        "--pklfile_train_path",
        type=str,
        default="",
        help="if isCreatedataset is False, pkl_files_path is required",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of .pt files to load (None for all files)",
    )

    # ============== Parameters for predict =================
    parser.add_argument(
        "--Predict",
        action="store_true",
        default=False,
        help="run model predict module",
    )
    # parser.add_argument("--isTrain", action="store_true", default=False)
    parser.add_argument("--predict_model_name", type=str)
    parser.add_argument("--predict_model_path", type=str)
    parser.add_argument("--pklfile_predict_path", type=str)

    args = parser.parse_args()

    args_default = vars(args)
    args_set = args_set()
    args_set.update(args_default)

    run_option = {
        "direct_reco": args_set,
    }

    run_module = RunModule(run_option)
    run_module.run()
