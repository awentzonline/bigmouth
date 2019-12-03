from collections import deque
import itertools
import os
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
from .flickr8k_dataset import PreprocessedFlickr8k
from .models import ZApproximator, SemanticZEncoder


class BigMouthApp(App):
    """
    Render images from BigGAN based on a description.
    """
    def run(self):
        os.makedirs(self.args.output_path, exist_ok=True)
        self.logger.info('Preparing model...')
        self.build_models()
        self.logger.info('Generating images...')
        try:
            self.train()
        except KeyboardInterrupt:
            pass
        self.infer()

    def train(self):
        image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        def caption_transform(c):
            return torch.rand(self.args.semantic_dims)

        dataset = PreprocessedFlickr8k(
            self.args.flickr_path, gan_encodings_path=self.args.gan_encodings_path,
            image_transform=image_transform,
            caption_transform=self.encode_document
        )

        params = itertools.chain(self.semantic_z_encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.args.lr)
        f_z_loss = nn.MSELoss()
        f_class_loss = nn.MSELoss()# nn.BCELoss()
        step = 0
        for epoch_i in range(self.args.epochs):
            data_loader = DataLoader(
                dataset, batch_size=self.args.batch_size,
                shuffle=True, drop_last=True
            )
            loss_history = deque([], 30)
            tqdm_iter = tqdm.tqdm(enumerate(data_loader))
            for batch_i, (img_arr, img_z, img_classes, caption_arr) in tqdm_iter:
                img_arr = img_arr.to(self.args.device)
                img_z = img_z.to(self.args.device)
                img_classes = img_classes.to(self.args.device)
                caption_arr = caption_arr.to(self.args.device)

                optimizer.zero_grad()
                p_z_caption, p_classes_caption = self.semantic_z_encoder(caption_arr)
                loss = f_z_loss(p_z_caption, img_z)
                # loss += f_class_loss(
                #     F.softmax(p_classes_caption, dim=-1),
                #     F.softmax(p_classes_img, dim=-1)
                # )
                loss += f_class_loss(p_classes_caption, img_classes)
                loss.backward()
                optimizer.step()

                loss_history.append(float(loss))
                step += 1
                mean_loss = np.mean(loss_history)
                tqdm_iter.set_postfix(dict(loss=float(mean_loss)))
                if step % self.args.report_freq == 0:
                    imgs = self.render_training_samples(
                        p_z_caption[:8], p_classes_caption[:8], img_arr[:8],
                        step=step
                    )

    def render_training_samples(self, z, classes, real, step=0):
        filename = f'search_{step}.png'
        print(z.shape, classes.shape)
        p_imgs = self.biggan(
            z, F.softmax(classes, dim=-1),
            self.args.truncation
        )
        p_imgs = (p_imgs + 1) / 2 # biggan is [-1, 1]
        imgs = torch.cat([real, p_imgs])
        torchvision.utils.save_image(
            imgs, os.path.join(self.args.output_path, filename),
            normalize=True, nrow=z.shape[0]
        )

    def infer(self):
        self.semantic_z_encoder.eval()
        with torch.no_grad():
            while True:
                text = input('Text to render:')
                docvec = self.encode_document(text)
                docvec = torch.from_numpy(docvec)
                z, classes = self.semantic_z_encoder(docvec[None, ...])
                if z is None:
                    print('No words matched.')
                    continue
                print(z.shape, classes.shape)
                # generate images
                output = self.biggan(
                    z, F.softmax(classes, dim=-1), self.args.truncation
                )
                filename = re.sub('\W+', '_', text.replace(' ', '_'))
                torchvision.utils.save_image(
                    output, os.path.join(self.args.output_path, f'{filename}.png'),
                    normalize=True
                )

    def build_models(self):
        print('Loading BigGAN...')
        self.biggan = BigGAN.from_pretrained(self.args.model).eval().to(self.args.device)
        print('Loading w2v...')
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(
            self.args.w2v_model_path, binary=True, limit=self.args.vocab_limit
        )
        print('Loading vgg...')
        self.vgg = torchvision.models.vgg.vgg16(pretrained=True).eval().to(self.args.device)
        print('Building other models...')
        bgc = self.biggan.config
        self.z_approximator = ZApproximator(self.biggan, vgg=self.vgg)#.to(self.args.device)
        self.semantic_z_encoder = SemanticZEncoder(
            self.args.semantic_dims, bgc.z_dim, bgc.num_classes
        ).to(self.args.device)
        print(self.z_approximator)
        print(self.semantic_z_encoder)

    def encode_document(self, text):
        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        vecs = []
        for word in text:
            if word in self.w2v:
                vecs.append(self.w2v[word])
        if not vecs:
            print(f'No encoding for `{text}`')
            return np.zeros(self.args.semantic_dims, dtype=np.float32)
        return np.mean(vecs, axis=0)

    @property
    def image_size(self):
        return (128, 128)

    @classmethod
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument('w2v_model_path')
        parser.add_argument('flickr_path')
        parser.add_argument('gan_encodings_path')
        parser.add_argument('--output-path', default='./generated/')
        parser.add_argument('--model', default='biggan-deep-128')
        parser.add_argument('--device', default='cpu')
        parser.add_argument('--num-samples', type=int, default=8)
        parser.add_argument('--encoder-dims', type=int, default=100)
        parser.add_argument('--semantic-dims', type=int, default=300)
        parser.add_argument('--truncation', type=float, default=1.)
        parser.add_argument('--epochs', type=int, default=1000000)
        parser.add_argument('--report-freq', type=int, default=1)
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--vocab-limit', type=int, default=200000)


if __name__ == '__main__':
    app = BigMouthApp.create_from_args()
    app.run()
