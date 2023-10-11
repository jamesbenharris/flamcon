from torchvision.io import read_video
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import os
import torch
import math
import logging
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class WebVidDataset(Dataset):
    """WebVid dataset."""

    def __init__(self, csv_file, root_dir, max_frames, tokenizer, max_tokens, samples=None, test=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if samples == None:
            self.webvideos = pd.read_csv(os.path.join(root_dir,csv_file))
        else:
            self.webvideos = pd.read_csv(os.path.join(root_dir,csv_file)).head(samples)
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.max_tokens=max_tokens
        self.test = test
        
    def csv(self):
        return self.webvideos

    def __len__(self):
        return len(self.webvideos)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_name = os.path.join(self.root_dir,'videos',
                                str(self.webvideos.iloc[idx, 1])+'.mp4')
        video = read_video(video_name,pts_unit='sec')[0].permute(0,3,1,2)[:self.max_frames]
        l,c,h,w = video.shape
        video = video.reshape(1,l,c,h,w).to(torch.float32)
        X = self.webvideos.iloc[idx, 6].strip()
        y = self.webvideos.iloc[idx, 7].strip()
        return (video, X, y, video_name)

def prepWebVid(data):
    rows = []
    for idx,row in data.iterrows():
        text = re.findall(r"[\w']+|[.,!?;]",row["name"]) 
        for i in range(len(text)-1):
            rw = row.copy()
            rw['X'] = ' '.join(text[:i+1])
            rw['y'] = text[i+1]
            rows.append(rw)
    rows = pd.DataFrame(rows)
    rows.reset_index()
    return rows

class RandomVideos(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, length=10,frames=10,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dialogue = torch.randint(0, 20000, (length, 512))
        self.videos = torch.randn(length, 1, frames, 3, 256, 256)
        self.transform = transform
        
    def __len__(self):
        return len(self.dialogue)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.videos[idx],self.dialogue[idx])
    
class RandomData(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, length=10,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dialogue = torch.randint(0, 20000, (length, 512))
        self.images = torch.randn(length, 2, 3, 256, 256)
        self.transform = transform
        
    def __len__(self):
        return len(self.dialogue)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.images[idx],self.dialogue[idx])