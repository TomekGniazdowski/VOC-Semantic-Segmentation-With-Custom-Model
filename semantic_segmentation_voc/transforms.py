import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize, Normalize, InterpolationMode
import albumentations as A
from kornia.augmentation import Denormalize
from PIL import Image
import numpy as np
import random
import cv2


SpacialTransforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(
        rotate=(-45, 45), 
        interpolation=cv2.INTER_NEAREST, 
        mask_interpolation=cv2.INTER_NEAREST, 
        mode=cv2.BORDER_CONSTANT, 
        cval=(255, 255, 255),
        cval_mask=255,
        p=0.3),
    A.Affine(
        translate_percent=(-0.15, 0.15), 
        interpolation=cv2.INTER_NEAREST, 
        mask_interpolation=cv2.INTER_NEAREST, 
        mode=cv2.BORDER_CONSTANT, 
        cval=(255, 255, 255),
        cval_mask=255,
        p=0.3),
    A.Affine(
        scale=(0.9, 1.1),
        interpolation=cv2.INTER_NEAREST, 
        mask_interpolation=cv2.INTER_NEAREST, 
        mode=cv2.BORDER_CONSTANT, 
        cval=(255, 255, 255),
        cval_mask=255,
        p=0.3),
    ], additional_targets={'image0': 'image'} # image0 is the target mask
)

ColorTransforms = A.Compose([
    A.AdvancedBlur((3, 3), (0.2, 1.0), p=0.15),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.15),
    A.RandomBrightnessContrast(p=0.15)
    ])

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
CONST_SIZE = 224 
ColorNormalization = Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
ColorDenormalization = Denormalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)


def pair_transforms_train(image: Image, target: Image):
    random.seed(np.random.randint(0, 1000))
    transformed = SpacialTransforms(image=np.asarray(image), image0=np.asarray(target))
    image, target = transformed['image'], transformed['image0']
    image = ColorTransforms(image=image)['image']
    image, target = Image.fromarray(image), Image.fromarray(target)
    
    image = Resize((CONST_SIZE, CONST_SIZE), interpolation=InterpolationMode.NEAREST)(image)
    target = Resize((CONST_SIZE, CONST_SIZE), interpolation=InterpolationMode.NEAREST)(target)
    
    out_image = TF.to_tensor(image)
    out_target = (TF.to_tensor(target) * 255).to(torch.long)
    
    out_image = ColorNormalization(out_image)
    
    assert out_image.size() == (3, CONST_SIZE, CONST_SIZE), out_image.size()
    assert out_target.size() == (1, CONST_SIZE, CONST_SIZE), out_target.size()

    return out_image, out_target.squeeze()


def pair_transforms_val(image: Image, target: Image):    
    image = Resize((CONST_SIZE, CONST_SIZE), interpolation=InterpolationMode.NEAREST)(image)
    target = Resize((CONST_SIZE, CONST_SIZE), interpolation=InterpolationMode.NEAREST)(target)

    out_image = TF.to_tensor(image)
    out_target = (TF.to_tensor(target) * 255).to(torch.long)
    
    out_image = ColorNormalization(out_image)
    
    assert out_image.size() == (3, CONST_SIZE, CONST_SIZE), out_image.size()
    assert out_target.size() == (1, CONST_SIZE, CONST_SIZE), out_target.size()
    
    return out_image, out_target.squeeze()