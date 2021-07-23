import random
import torch
import glob
from PIL import Image
from torchvision import transforms as T


class ImageNet_5Class(torch.utils.data.Dataset):
    """Some Information about ImageNet_5Class"""

    def __init__(self, train: bool = True, augmentation: bool = False, annotation: bool = False, ratio: float = 1.0):
        super(ImageNet_5Class, self).__init__()
        data_dir = 'data/train/' if train else 'data/test/'
        image_list = glob.glob(f'{data_dir}/*.jpg')
        self.image_list = random.sample(image_list, int(ratio*len(image_list)))
        self.augmentation = augmentation
        self.annotation = annotation

    def __getitem__(self, index):
        img_dir = self.image_list[index]
        pil_img = Image.open(img_dir)
        no_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        basic_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((224, 224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        tensor_img = basic_transform(
            pil_img) if self.augmentation else no_transform(pil_img)

        if self.annotation:
            img_label = (img_dir.split('/')[-1]).split('_')[0]
            label = ["airplane", "car", "cat",
                     "dog", "elephant"].index(img_label)
            return tensor_img, label
        else:
            return tensor_img

    def __len__(self):
        return len(self.image_list)
