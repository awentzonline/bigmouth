bigmouth
=========
Render images from text.

Data
----
 * Download word vectors from https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
 * [Flickr8k](https://www.kaggle.com/ming666/flicker8k-dataset/data)

Use `bigmouth.preprocess` to find BigGAN input codes which approximate
the dataset images.
use `bigmouth.train_preprocessed` to learn a mapping from document vectors
to BigGAN inputs that produce corresponding images.
