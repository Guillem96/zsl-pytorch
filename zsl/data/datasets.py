from pathlib import Path
from typing import Any, Mapping, Collection, Callable, Sequence, Union

import torch
from torch.utils.data import Dataset

from torchvision.datasets import ImageFolder, folder


class ZSLImageFolder(ImageFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Parameters
    ----------
    root: str 
        Root directory path.
    class_to_repr: Callable[str, torch.FloatTensor]
        Function that given a class name returns its semantic representation
    zero_shot_classes: List[str]
        List of ZS classes. 
    load_unseen: bool, default False
        Wether to work with unseen classes too. 
    load_only_unseen: bool, default False
        If set to true only Zero Shot classes are going to be loaded
    transform: Callable, default None
        A function/transform that takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform: Callable, default None 
        A function/transform that takes in the target and transforms it.
    loader: Callable, default None  
        A function to load an image given its path.
    is_valid_file: Callable, default None  
        A function that takes path of an Image file and check if the file 
        is a valid file (used to check of corrupt files)

    Attributes
    ----------
    classes: List[str] 
        List of the class names.
    class_to_idx: Mapping[str, int]
        Dict with items (class_name, class_index).
    imgs: List[Tuple[str, int]] 
        List of (image path, class_index) tuples
    """

    def __init__(self,
                 root: str,
                 class_to_repr: Callable[[str], torch.FloatTensor],
                 zero_shot_classes: Collection[str],
                 load_unseen: bool = False,
                 load_only_unseen: bool = False,
                 *args, **kwargs):

        is_valid_fn = kwargs.get('is_valid_file')

        def is_valid_file_and_is_seen(path):
            path = Path(path)
            # If the image is a zero shot file, wo do not load it
            if path.parent.stem in zero_shot_classes:
                return False

            # If the image is not a zero shot class, delegate the decision of
            # valid file to is_valid_file function
            if is_valid_fn is not None:
                return is_valid_fn(path)
            else:
                return True

        def is_valid_and_unseen(path):
            path = Path(path)
            # If the image is a zero shot file, wo do not load it
            if path.parent.stem not in zero_shot_classes:
                return False

            # If the image is not a zero shot class, delegate the decision of
            # valid file to is_valid_file function
            if is_valid_fn is not None:
                return is_valid_fn(path)
            else:
                return True

        if not load_unseen:
            kwargs['is_valid_file'] = is_valid_file_and_is_seen

        if load_only_unseen:
            kwargs['is_valid_file'] = is_valid_and_unseen

        super(ZSLImageFolder, self).__init__(root, *args, **kwargs)
        self.class_to_repr = class_to_repr
        self.zero_shot_classes = zero_shot_classes
        self.load_unseen = load_unseen

    @property
    def valid_classes(self):
        def path_2_label(p): return Path(p).parent.stem
        return set(path_2_label(o[0]) for o in self.samples)

    def semantic_representations(self) -> Sequence[Any]:
        return [self.class_to_repr(o) for o in self.classes]

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        semantic = self.class_to_repr(self.classes[target])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, semantic


class AwAFeaturesDataset(Dataset):
    _train_names = [
        "killer+whale", "beaver", "dalmatian", "persian+cat", "german+shepherd", 
        "siamese+cat", "skunk", "mole", "tiger", "hippopotamus", "leopard", 
        "spider+monkey", "elephant", "gorilla", "ox", "chimpanzee", "hamster", 
        "fox", "squirrel", "rabbit", "wolf", "chihuahua", "weasel", "otter", 
        "buffalo", "zebra", "giant+panda", "pig", "lion", "polar+bear", "collie", 
        "cow", "deer", "mouse", "humpback+whale", "antelope", "grizzly+bear", 
        "rhinoceros", "raccoon", "moose"]

    _test_names = ["sheep", "dolphin", "bat", "seal", "blue+whale", "rat", "horse", 
        "walrus", "giraffe", "bobcat"]

    def __init__(self,
                 root: str,
                 features_type: str = 'ResNet-101',
                 load_unseen: bool = False,
                 load_only_unseen: bool = False):

        root = Path(root)
        assert root.exists()

        features_root = root / 'Features' / features_type
        assert features_root.exists()

        # Build class to idx
        def file_to_mapping(f):
            split_lines = f.open().readlines()
            split_lines = (o.split('\t') for o in split_lines)
            return {c.strip(): int(i) - 1 for i, c in split_lines}

        self.class_to_idx = file_to_mapping(root / 'classes.txt')
        self.classes = [k for k, _ in 
                        sorted(self.class_to_idx.items(), 
                               key=lambda item: item[1])]
        
        # Get Zero Shot classes and Non Zero Shot
        self.training_casses = AwAFeaturesDataset._train_names
        self.zero_shot_classes = AwAFeaturesDataset._test_names

        # Generate attributes name to idx
        self.attr_to_idx = file_to_mapping(root / 'predicates.txt')
        self.attrs = [k for k, _ in 
                      sorted(self.attr_to_idx.items(), 
                             key=lambda item: item[1])]

        # Load mapping class to features
        self.attr_matrix = root / 'predicate-matrix-continuous.txt'
        self.attr_matrix = self.attr_matrix.open().readlines()
        self.attr_matrix = [list(map(float, l.split()))
                            for l in self.attr_matrix]
        self.attr_matrix = torch.FloatTensor(self.attr_matrix)
        self.attr_matrix = (self.attr_matrix -
                            self.attr_matrix.mean(0)) / self.attr_matrix.std(0)

        # Create the dataset samples from features files
        features = (features_root / 'AwA2-features.txt').open().readlines()
        targets = (features_root / 'AwA2-labels.txt').open().readlines()
        targets = [int(l) - 1 for l in targets]
        self.samples = list(zip(features, targets))

        if not load_unseen:
            self.samples = [(p, t) for p, t in self.samples
                            if self.classes[t] in self.training_casses]

        if load_only_unseen:
            self.samples = [(p, t) for p, t in self.samples
                            if self.classes[t] in self.zero_shot_classes]

    @property
    def valid_classes(self):
        return set(self.classes[t] for _, t in self.samples)

    def class_to_repr(self, class_: Union[str, int]):
        if isinstance(class_, str):
            class_ = self.class_to_idx[class_]
        return self.attr_matrix[class_]

    def __getitem__(self, index: int):
        features, target = self.samples[index]
        sample = torch.FloatTensor([float(o) for o in features.split()])
        semantic = self.class_to_repr(target)

        return sample, target, semantic

    def __len__(self):
        return len(self.samples)