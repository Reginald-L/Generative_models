from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torch
from torch.nn.functional import tanh
from torchvision.transforms.functional import to_tensor


class CustomDataset(Dataset):
    def __init__(self, name:str) -> None:
        super().__init__()
        self.name = name


class ImgDataset(CustomDataset):
    def __init__(self, name='img', dir=None, transform=None, norm=True) -> None:
        super().__init__(name)
        assert dir is not None, 'dir must be a directory containing images'
        self.dir = dir
        self.extensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.JPEG']
        self.dataset = self.load_dataset()
        self.transform = transform
        self.to_tensor = to_tensor
        self.norm = norm

    def load_dataset(self):
        dataset = []
        root = Path(self.dir)
        for ext in self.extensions:
            dataset.extend(root.rglob(ext))
        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def pil_imread(self, impath):
        return Image.open(impath).convert('RGB')
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.pil_imread(sample)
        if self.transform is not None:
            image = self.transform(image)
        if not isinstance(image, torch.Tensor):
            image = self.to_tensor(image)
        if self.norm:
            image = tanh(image)
        return image