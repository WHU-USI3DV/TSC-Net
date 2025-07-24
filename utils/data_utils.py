import torch
import numpy as np
from PIL import Image
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def to_tensor(pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError(
            'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)

    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img



def process_image(img_path):
    image = Image.open(img_path).convert('RGB')
    w, h = image.size
    short_size = min(w, h)
    image = image.resize((short_size, short_size))
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = to_tensor(image)
    image = normalize(image)

    return image


def process_boxes(boxes, max_obj=50):
    position = np.zeros((max_obj, 4), dtype=np.float32)
    mask_2d = np.zeros((max_obj), dtype=bool)
    for i, box in enumerate(boxes):
        position[i] = box
        mask_2d[i] = 1

    return position, mask_2d

def process_input(img_path, boxes, device):
    image = process_image(img_path)
    position, mask_2d = process_boxes(boxes)
    position = torch.from_numpy(position)
    mask_2d = torch.from_numpy(mask_2d)
    image = image.unsqueeze(0).to(device)
    position = position.unsqueeze(0).to(device)
    mask_2d = mask_2d.unsqueeze(0).to(device)

    return {'image': image, 'position': position, 'mask_2d': mask_2d}