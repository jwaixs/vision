import torch.utils.data as data

from PIL import Image
import os
import os.path

from .folder import default_loader

class ImageList(data.Dataset):
    """A generic data loader where the input is an image list in this way: ::

        [
            [/path/to/image/1, label_1],
            [/path/to/image/2, label_2],
            ...
            [/path/to/image/n, label_n]
        ]

    Args:
        image_list (list): List of image paths and labels.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, source_transform = None,
                 target_transform = None, loader = default_loader):
        self.csv_path = csv_path
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.image_list = image_list

    def __getitem__(self, index):
        path, target = self.image_list[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.image_list)
