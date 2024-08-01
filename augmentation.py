#system library
import sys
sys.path.append('../')


#python library
import random

#torch library
import torch

#third-party library
import albumentations as A
import cv2


#other project files
import config as cfg 


class Augmentation():

    def __init__(self):
        super(Augmentation, self).__init__()
        self.transform_structure = self._get_geometric_transformation()
        

    def  _get_geometric_transformation(self):
        rotation_limit = random.randint(5,10)
        transform = A.Compose(
            [
                A.CenterCrop (height = 384, width = 384, p=1.0),
                A.Rotate(limit=rotation_limit, p=0.5),
                A.VerticalFlip(p=0.5),              
                A.RandomRotate90(p=0.5),
            ],
            additional_targets = {'image0':'image', 'image1':'image'}
        )
        return transform

    def generation(self, img, target, img_8_bit):
        transformed = self.transform_structure(image=img, image0=target, image1=img_8_bit)
        aug_img, aug_label, aug_img_8_bit = transformed['image'], transformed['image0'], transformed['image1']
        #transformed_pixel = self.transform_pixel(image=aug_img)
        #aug_img = transformed_pixel['image']
        return aug_img, aug_label, aug_img_8_bit

