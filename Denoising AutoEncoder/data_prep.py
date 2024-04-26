from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image

class dataset(Dataset):
    def __init__(self, root= 'data', split= 'train'):
       self.img_paths = []
       self.transforms = transforms.PILToTensor()

       if split == 'train':
           train_path = os.path.join(root, split)

           for path in os.listdir(train_path):
               img_path = os.path.join(train_path, path)
               img_path = self.img_paths.append(img_path)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])

        img_tensor = self.transforms(img).float()

        return img_tensor

    def __len__(self):
        return len(self.img_paths)
    
def prep_data(batch_size= 40, shuffle= True):
    train_set = dataset()
    train_loader = DataLoader(train_set, batch_size, shuffle= shuffle)
    
    return train_loader
