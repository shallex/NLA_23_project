import os
import jax
import numpy as np
import torch
import torchtext
import torchvision
import torchvision.transforms as transforms
from datasets import DatasetDict, load_dataset
from torch.utils.data import TensorDataset, random_split
from tqdm import tqdm


# ### MNIST Sequence Modeling
# **Task**: Predict next pixel value given history, in an autoregressive fashion (784 pixels x 256 values).
#
def create_mnist_dataset(bsz=128):
    print("[*] Generating MNIST Sequence Modeling Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 256, 1

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x.view(IN_DIM, SEQ_LENGTH).t() * 255).int()
            ),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train,
        batch_size=bsz,
        shuffle=True,
    )
    testloader = torch.utils.data.DataLoader(
        test,
        batch_size=bsz,
        shuffle=False,
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### FSDD Sequence Modeling
# **Task**: Predict next wav value given history, in an autoregressive fashion (6400 pixels x 256 values).
#
def create_fsdd_dataset(bsz=128):
    print("[*] Generating FSDD Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 6400, 256, 1

    from torchaudio.transforms import MuLawEncoding
    from torchfsdd import TorchFSDDGenerator, TrimSilence

    # Create a transformation pipeline to apply to the recordings
    tf = transforms.Compose(
        [
            TrimSilence(threshold=1e-6),
            MuLawEncoding(quantization_channels=255),
            transforms.Lambda(
                lambda x: torch.nn.functional.pad(
                    x.view(-1), 
                    (0, SEQ_LENGTH - x.shape[0]), 
                    "constant", 
                    255
                ).view(-1, 1)
            ),
        ]
    )

    # Fetch the latest version of FSDD and initialize a generator with those files
    fsdd = TorchFSDDGenerator("local", "recordings/", transforms=tf)

    # Create two Torch datasets for a train-test split from the generator
    train, test = fsdd.train_test_split(test_size=0.1)
    
    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### Speech Commands Sequence Modeling
# **Task**: Predict next wav value given history, in an autoregressive fashion (8000 samples x 256 values).
#
def create_sc_dataset(bsz=128):
    print("[*] Generating SC Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 8000, 256, 1
    import os
    from torchaudio.datasets import SPEECHCOMMANDS
    from torchaudio.transforms import MuLawEncoding, Resample

    # # Create a transformation pipeline to apply to the recordings
    tf = transforms.Compose(
        [
            Resample(16000, SEQ_LENGTH),
            MuLawEncoding(quantization_channels=255),
            transforms.Lambda(
                lambda x: torch.nn.functional.pad(
                    x.view(-1),
                    (0, SEQ_LENGTH - x.view(-1).shape[0]),
                    "constant",
                    255,
                ).view(-1, 1)
            ),
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
            return out, 0

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


# ### Speech Commands Sequence Modeling
# **Task**: Predict next wav value given history, in an autoregressive fashion (8000 samples x 256 values).
#
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


# ### MNIST Classification
# **Task**: Predict MNIST class given sequence model over pixels (784 pixels => 10 classes).
def create_mnist_classification_dataset(bsz=128):
    print("[*] Generating MNIST Classification Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 10, 1
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
            transforms.Lambda(lambda x: x.view(IN_DIM, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### FSDD Classification
# **Task**: Predict FSDD class given sequence model over pixels (6400 wav => 10 classes).
def create_fsdd_classification_dataset(bsz=128):
    print("[*] Generating FSDD Classification Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 6400, 10, 1

    from torchaudio.transforms import MuLawEncoding
    from torchfsdd import TorchFSDDGenerator, TrimSilence

    # Create a transformation pipeline to apply to the recordings
    tf = transforms.Compose(
        [
            TrimSilence(threshold=1e-6),
            MuLawEncoding(quantization_channels=512),
            transforms.Lambda(
                lambda x: torch.nn.functional.pad(
                    x, (0, 6400 - x.shape[0])
                ).view(-1, 1)
            ),
        ]
    )

    # Fetch the latest version of FSDD and initialize a generator with those files
    fsdd = TorchFSDDGenerator(version="master", transforms=tf)

    # Create two Torch datasets for a train-test split from the generator
    train, test = fsdd.train_test_split(test_size=0.1)

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


Datasets = {
    "mnist": create_mnist_dataset,
    "fsdd": create_fsdd_dataset,
    "sc": create_sc_dataset,
    "mnist-classification": create_mnist_classification_dataset,
    "fsdd-classification": create_fsdd_classification_dataset,
    "sc-classification": create_sc_classification_dataset
}
