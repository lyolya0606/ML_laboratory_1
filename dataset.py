import os
from scipy.io import loadmat
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config


class ImageDataset(Dataset):
    def __init__(self, image_dir, annotation_path, image_size=224, mode="Train"):
        self.image_dir, self.image_size, self.mode = image_dir, image_size, mode
        self.annotations = self._load_annotations(annotation_path)  # Загружаем аннотации
        self.transform = self._get_transform()  # Преобразования

    def _load_annotations(self, annotation_path):
        annotations = loadmat(annotation_path)["annotations"][0]
        return [{
            "fname": ann[5][0],
            "bbox": tuple(ann[i][0][0] for i in range(4)),
            "class_id": ann[4][0][0] - 1
        } for ann in annotations]

    def _get_transform(self):
        transforms_list = [transforms.Resize((self.image_size, self.image_size))]
        if self.mode == "Train":
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.extend([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = Image.open(os.path.join(self.image_dir, ann["fname"])).convert("RGB").crop(ann["bbox"])
        return {"image": self.transform(image), "target": ann["class_id"]}


def load_dataset():
    def create_dataloader(image_dir, annotation_path, mode, shuffle, drop_last):
        dataset = ImageDataset(image_dir, annotation_path, config.image_size, mode)
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                          num_workers=config.num_workers, pin_memory=True,
                          drop_last=drop_last, persistent_workers=True)

    return (
        create_dataloader(config.train_image_dir, config.train_annotation_path, "Train", True, True),
        create_dataloader(config.valid_image_dir, config.valid_annotation_path, "Valid", False, False)
    )
