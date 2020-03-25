from pathlib import Path
from typing import Any, Mapping, Collection, Callable, Sequence

import torch
from torch.utils.data import Dataset

from torchvision.datasets import ImageFolder


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
        path_2_label = lambda p: Path(p).parent.stem
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
