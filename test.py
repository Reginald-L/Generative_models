import sys
sys.path.insert(0, '/home/liruijun/projects/Generative_models')

from data.data_utils import ImgDataset

img_dataset = ImgDataset('test', '/home/liruijun/datasets/flowers_dataset/train')
print(len(img_dataset))