import PIL.Image
import numpy
import paddle.io
import paddle.vision.transforms as transforms


class Dataset(paddle.io.Dataset):
    def __init__(self, mode='train'):
        self.size = 160, 160
        self.mode = mode

        self.image_paths = []
        self.label_paths = []

        with open('./' + mode + '.txt', 'r') as file:
            for line in file.readlines():
                image, label = line.strip().split('\t')
                self.image_paths.append(image)
                self.label_paths.append(label)

    def _get_image(self, path, grayscale=False, tf=None):
        with PIL.Image.open(path) as img:
            if grayscale:
                if img.mode not in ('L', 'I;16', 'I'):
                    img = img.convert('L')
            else:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            return transforms.Compose([transforms.Resize(self.size)] + (tf or []))(img)

    def __getitem__(self, item):
        image = self._get_image(self.image_paths[item], tf=[
            transforms.Transpose(),
            transforms.Normalize(mean=127.5, std=127.5)
        ])
        label = self._get_image(self.label_paths[item], grayscale=True, tf=[
            transforms.Grayscale()
        ])
        return numpy.array(image, dtype='float32'), numpy.array(label, dtype='int64')

    def __len__(self):
        return len(self.image_paths)
