"""
    Parse input arguments
"""

import argparse


class Options:
    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='Network for merging focal stack.')

        # Training Parameter
        parser.add_argument("--train", action='store_true', help="train or test", default=False)
        parser.add_argument("--testReal", action='store_true', help="test real data or synthesized", default=False)
        parser.add_argument("--BegEpoch", type=int, help="The Begin Epoch", default=1)
        parser.add_argument("--NumEpoch", type=int, help="The Number of Epoch", default=100)
        parser.add_argument("--lr", type=float, help="Learning Rate", default=5e-4)
        parser.add_argument("--num_workers", type=int, help="The number of loader workers", default=1)
        parser.add_argument("--milestone", type=list, help="Learning Rate Scheduler Milestones",
                            default=[x for x in range(50, 151, 20)])
        parser.add_argument("--gamma", type=float, help="Learning Rate Scheduler Gamma", default=0.1)
        parser.add_argument("--split_scale", type=float, help="Validation set proportion", default=0.1)
        parser.add_argument("--CropSize", type=int, help="Training image crop size", default=256)
        parser.add_argument("--batch_size", type=int, default=10)
        parser.add_argument("--StackSize", type=int, default=64, help="How many frames to input to each focal stack.")

        # Data Parameter
        parser.add_argument("--RGB", type=bool, help="Is RGB image", default=True)
        parser.add_argument("--use_gpus", action='store_true', help="Usage of GPUs", default=True)

        parser.add_argument("--load_weight", action='store_true', help="Load model weight before training")
        parser.add_argument("--ckp", type=str, help="The path of model weight file",
                            default="pretrained/model_best.pth")

        parser.add_argument("--LogPath", type=str, help="The path of log info",
                            default="LogFiles")

        parser.add_argument("--DataPath", type=str, help="The path to data.",
                            default="D:/Research/FocusDataset_Images/")
        parser.add_argument("--TestSavePath", type=str, help="Path to save results.",
                    default="result_pics")


        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
