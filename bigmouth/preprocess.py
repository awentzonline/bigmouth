from collections import deque
import itertools
import os
import pickle
import re

import gensim
import numpy as np
import pandas as pd
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)
from sklearn import preprocessing
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import tqdm

from .app import App
from .flickr8k_dataset import PreprocessFlickr8k
from .models import ZApproximator, SemanticZEncoder


class BigMouthPreprocessorApp(App):
    """
    Find BigGAN encodings for batches of images to be used in bigmouth.
    """
    def run(self):
        os.makedirs(self.args.output_path, exist_ok=True)
        self.logger.info('Preparing model...')
        self.build_models()
        self.logger.info('Generating images...')
        self.generate()

    def generate(self):
        image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

        dataset = PreprocessFlickr8k(
            self.args.flickr_path,
            image_transform=image_transform,
            caption_transform=self.encode_document
        )
        data_loader = DataLoader(
            dataset, batch_size=self.args.batch_size,
        )
        step = 0
        loss_history = deque([], 20)
        files_encoded = {}
        for batch_i, (img_arr, path) in tqdm.tqdm(enumerate(data_loader)):
            img_arr = img_arr.to(self.args.device)

            p_z_img, p_classes_img, approx_img, loss = self.z_approximator(
                img_arr, truncation=self.args.truncation,
                iterations=self.args.approx_iters, lr=self.args.approx_lr
            )
            p_z_img = p_z_img.cpu().detach().numpy()
            p_classes_img = p_classes_img.cpu().detach().numpy()
            for p, z, c in zip(path, p_z_img, p_classes_img):
                files_encoded[p] = (z, c)
            loss_history.append(float(loss))
            step += 1
            if step % self.args.report_freq == 0:
                mean_loss = np.mean(loss_history)
                #print(f'Batch {batch_i} loss {mean_loss}')
                imgs = self.render_samples(
                    img_arr, approx_img,
                    step=step
                )
        with open(self.args.encoded, 'wb') as outfile:
            pickle.dump(files_encoded, outfile)

    def render_samples(self, real, approx, step=0):
        filename = f'batch_{step}.png'
        imgs = torch.cat([real, approx])
        torchvision.utils.save_image(
            imgs, os.path.join(self.args.output_path, filename),
            normalize=True, nrow=real.shape[0]
        )

    def build_models(self):
        print('Loading BigGAN...')
        self.biggan = BigGAN.from_pretrained(self.args.model).eval().to(self.args.device)
        print('Loading vgg...')
        self.vgg = torchvision.models.vgg.vgg16(pretrained=True).eval().to(self.args.device)
        print('Building other models...')
        bgc = self.biggan.config
        self.z_approximator = ZApproximator(self.biggan, vgg=self.vgg)

    def encode_document(self, text):
        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        vecs = []
        for word in text:
            if word in self.w2v:
                vecs.append(self.w2v[word])
        return np.mean(vecs, axis=0)

    @property
    def image_size(self):
        return (128, 128)

    @classmethod
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument('flickr_path')
        parser.add_argument('--output-path', default='./generated/')
        parser.add_argument('--encoded', default='encodings.pkl')
        parser.add_argument('--model', default='biggan-deep-128')
        parser.add_argument('--device', default='cpu')
        parser.add_argument('--approx-iters', type=int, default=10)
        parser.add_argument('--approx-lr', type=float, default=0.1)
        parser.add_argument('--truncation', type=float, default=1.)
        parser.add_argument('--report-freq', type=int, default=1)
        parser.add_argument('--batch-size', type=int, default=3)


if __name__ == '__main__':
    app = BigMouthPreprocessorApp.create_from_args()
    app.run()
