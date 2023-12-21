import torch
import wandb
import argparse

from pytorch_lightning import seed_everything
from datasets import DATASETS
from config import *
# from model import *
from model.minimal_LRU import LRU, ClassificationModel
from dataloader import *
from trainer import *

import torch
import torchvision.transforms as transforms




def create_sc_classification_dataset(bsz=128):
    print("[*] Generating SC Classification Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 8000, 10, 1
    import os
    from torchaudio.datasets import SPEECHCOMMANDS
    from torchaudio.transforms import MuLawEncoding, Resample

    # # Create a transformation pipeline to apply to the recordings
    tf = transforms.Compose(
        [
            # transforms.Lambda(lambda x: x if print('Initial shape: ', x.shape) else x),
            Resample(16000, SEQ_LENGTH),
            # transforms.Lambda(lambda x: x if print('After resample: ', x.shape) else x),
            MuLawEncoding(quantization_channels=512),
            # transforms.Lambda(lambda x: x if print('After mulaw: ', x.shape) else x),
            transforms.Lambda(
                lambda x: torch.nn.functional.pad(
                    x, (0, SEQ_LENGTH - x.shape[1])
                ).view(-1, 1)
            ),
            # transforms.Lambda(lambda x: x if print('After padding: ', x.shape) else x),

        ]
    )


    class SubsetSC(SPEECHCOMMANDS):
        def __init__(self, subset: str = None):
            super().__init__("./", download=True)
            digits = [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            ]

            self.label2int = dict(zip(digits, range(len(digits))))

            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as fileobj:
                    return [
                        os.path.join(self._path, line.strip())
                        for line in fileobj
                        if line.split("/")[0] in digits
                    ]

            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = load_list("validation_list.txt") + load_list(
                    "testing_list.txt"
                )
                excludes = set(excludes)
                self._walker = [
                    w
                    for w in self._walker
                    if w not in excludes
                    if w.split("/")[-2] in digits
                ]

        def __getitem__(self, n):
            (
                waveform,
                sample_rate,
                label,
                speaker_id,
                utterance_number,
            ) = super().__getitem__(n)
            out = tf(waveform)
            label = self.label2int[label]
            return out, label

    # Create training and testing split of the data. We do not use validation in this tutorial.
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")

    waveform, label = train_set[0]
    print(waveform.shape, label)
    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


def train(args, export_root=None):
    seed_everything(args.seed)


    trainloader, testloader, n_classes, l_max, d_input = create_sc_classification_dataset(
        bsz=args.train_batch_size
    )
    args.num_items = 10

    lru = LRU(in_features=64,out_features=64, state_features=64)
    model = ClassificationModel(lru=lru,
                                d_output=10,
                                d_model=64,
                                n_layers=2)
    

    model.to("cuda")

    trainer = LRUTrainer(args, model, trainloader, testloader, None, export_root="./export_dir", use_wandb=True)
    trainer.train()
    # trainer.test()


if __name__ == "__main__":
    set_template(args)
    print(args)
    train(args)
