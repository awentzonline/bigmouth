import glob
import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset


class Flickr8k(VisionDataset):
    """For the structure of whatever I found on Kaggle.

    * root/
     * Flickr8k_text/*.txt
     * Flickr8k_Dataset/Flicker8k_Dataset/*.jpg
    """
    def __init__(self, root, image_transform=None, caption_transform=None, **dataset_kwargs):
        super().__init__(root, **dataset_kwargs)
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self._img_cache = {}
        self._caption_cache = {}
        self._setup()

    def _setup(self):
        # load image information
        filenames = glob.glob(os.path.join(self.image_root, '*.jpg'))
        names = [
            os.path.basename(f)
            for f in filenames
        ]
        self.img_data = {
            name: dict(path=path)
            for name, path in zip(names, filenames)
        }
        # load captions
        caption_filename = os.path.join(self.root, 'Flickr8k_text', 'Flickr8k.token.txt')
        captions_df = pd.read_csv(
            caption_filename, delimiter='\t', names=['caption_id', 'caption']
        )
        captions_df['path'] = captions_df['caption_id'].apply(
            lambda x: x.rsplit('#', 1)[0]
        )
        captions_df = captions_df[captions_df['path'].isin(names)]
        captions_df.reset_index(inplace=True, drop=True)
        self.captions_df = captions_df

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, index):
        if index not in self._img_cache:
            row = self.captions_df.iloc[index]
            # image
            filename = os.path.join(self.image_root, row.path)
            img = Image.open(filename)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_arr = self.image_transform(img)
            self._img_cache[index] = img_arr

            # caption
            caption = row.caption
            caption_arr = self.caption_transform(caption)
            self._caption_cache[index] = caption_arr
        img_arr = self._img_cache[index]
        caption_vec = self._caption_cache[index]
        return img_arr, caption_arr

    @property
    def image_root(self):
        return os.path.join(self.root, 'Flickr8k_Dataset', 'Flicker8k_Dataset')


class PreprocessFlickr8k(Flickr8k):
    def _setup(self):
        super()._setup()
        self.captions_df = self.captions_df.groupby('path').first().reset_index()

    def __getitem__(self, index):
        row = self.captions_df.iloc[index]
        filename = os.path.join(self.image_root, row.path)
        img = Image.open(filename)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_arr = self.image_transform(img)

        return img_arr, row.path


class PreprocessedFlickr8k(Flickr8k):
    def __init__(self, *args, gan_encodings_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        with open(gan_encodings_path, 'rb') as infile:
            self.gan_encodings = pickle.load(infile)

    def __getitem__(self, index):
        row = self.captions_df.iloc[index]
        if index not in self._img_cache:
            # image
            filename = os.path.join(self.image_root, row.path)
            img = Image.open(filename)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_arr = self.image_transform(img)
            self._img_cache[index] = img_arr

            # caption
            caption = row.caption
            caption_arr = self.caption_transform(caption)
            self._caption_cache[index] = caption_arr
        img_arr = self._img_cache[index]
        caption_arr = self._caption_cache[index]
        z, classes = self.gan_encodings[row.path]
        return img_arr, z, classes, caption_arr
