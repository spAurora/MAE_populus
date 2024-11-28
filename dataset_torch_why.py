import warnings
from pathlib import Path

import rasterio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# python c:\Users\xianyu\Documents\Source\proj_rs_mae\mae-main\main_pretrain.py --model mae_vit_large_patch16_populus --batch_size 64 --data_path e:\Populus_MAE_dataset\image-batch-1\train

mean_pixel_populus = [0.1767787, 0.19072463, 0.20309546, 0.2463202]
std_pixel_populus = [0.04486727, 0.05505818, 0.06724718, 0.07047508]

transforms_train = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean_pixel_populus, std=std_pixel_populus),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=(0, 360), interpolation=v2.InterpolationMode.BILINEAR),
        v2.RandomResizedCrop(size=(256, 256), scale=(0.2, 1.0), ratio=(0.8, 1.25), interpolation=v2.InterpolationMode.BICUBIC),
    ]
)

transforms_test = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean_pixel_populus, std=std_pixel_populus),
    ]
)


# 定义自定义数据集类
class MyPopulusDataset(Dataset):
    def __init__(self, home, transforms):
        super(MyPopulusDataset, self).__init__()
        self.transforms = transforms
        self.dataset = list(Path(home).rglob("*.tif"))

    def __getitem__(self, index):
        with rasterio.open(self.dataset[index]) as src:
            img = src.read()

        img = torch.from_numpy(img)

        return self.transforms(img)

    def __len__(self):
        return len(self.dataset)


# if __name__ == "__main__":
#     ds = MyPopulusDataset(r"e:\Populus_MAE_dataset\image-batch-1\train", transforms=transforms_train)
#     print(len(ds))

#     sampler = RandomSampler(ds)

#     dataloader = DataLoader(ds, batch_size=16, sampler=sampler, pin_memory=True)

#     import time

#     time_start = time.time()
#     # 查看 WeightedRandomSampler 的效果
#     for batch_idx, data in enumerate(dataloader):
#         time_end = time.time()
#         print(f"Batch {batch_idx + 1}: {time_end - time_start:.2f} seconds")
#         if batch_idx == 3:  # 输出前三个批次
#             break

#         data = data.to("cuda")
#         data = None

#         time_start = time.time()

#     # 训练循环
#     for epoch in range(10):
#         for batch in dataloader:
#             inputs, labels = batch
#             print(inputs.shape, labels.shape)
