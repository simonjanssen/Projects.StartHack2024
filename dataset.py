import torch
import pandas as pd
from pathlib import Path 
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from timm import create_model


class NumpyToTensor(object):
    """Convert a numpy.ndarray to a tensor."""
    
    def __call__(self, np_array):
        """
        Args:
            np_array (numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return torch.from_numpy(np_array)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class N5kRealSense(Dataset):
    def __init__(self, path_imagery, path_labels_csv, path_split_txt, transform=None, target_transform=None):
        self.path_imagery = Path(path_imagery)
        assert self.path_imagery.is_dir()

        dish_id_to_image_path = {}
        for path_dish in Path(path_imagery).glob("*"):
          dish_id = path_dish.name
          path_img = Path(path_dish, "rgb.png")
          assert path_img.is_file()
          #print(path_img)
          dish_id_to_image_path[dish_id] = str(path_img)
        self.dish_id_to_image_path = dish_id_to_image_path
        
        self.labels = pd.read_csv(path_labels_csv, usecols=range(6), header=None, index_col=0)

        with open(path_split_txt, "r") as fp:
          _split_ids = fp.read()
        _split_ids = _split_ids.split("\n")
        self.split_ids = []
        for _split_id in _split_ids:
          if _split_id in self.dish_id_to_image_path:
            self.split_ids.append(_split_id)

        print(f"Split size: {len(self.split_ids)} (orginal: {len(_split_ids)})")
        
        #self.split_ids = pd.read_csv(path_split_txt, header=None, index_col=None)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # get next dish id from split list
        dish_id = self.split_ids[idx]
        #print(dish_id)

        # get image for this dish_id
        path_image = self.dish_id_to_image_path[dish_id]
        #assert path_image.is_file(), path_image
        image = Image.open(path_image)
        #image = image.convert("RGB")

        # get label for this dish_id
        target = self.labels.loc[dish_id].to_numpy()

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

input_transform = transforms.Compose([
    transforms.Resize(255),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    NumpyToTensor()
])

dataset = N5kRealSense(
    path_imagery="./nutrition5k_dataset/imagery/realsense_overhead", 
    path_labels_csv="./nutrition5k_dataset/metadata/dish_metadata_cafe1.csv",
    path_split_txt="./nutrition5k_dataset/dish_ids/splits/rgb_train_ids.txt",
    transform=input_transform,
    target_transform=target_transform
)

len(dataset)
(input, target) = dataset.__getitem__(0)
print(input.size(), target)
#print(target)
#print(np.array(input).shape)
#plt.imshow(np.array(input))
#plt.show()

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

for (image, target) in dataloader:
  print(type(image), target)
  break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model('efficientnet_b0', pretrained=True, num_classes=5)
model.to(device)
optimizer = Adam(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

def train_one_epoch(model):
  model.train()

  for input, target in dataloader:
    input = input.float().to(device)
    target = target.float().to(device)
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    break


epochs = 1
for e in range(epochs):
  print(f"{e}/{epochs}")
  train_one_epoch(model)

