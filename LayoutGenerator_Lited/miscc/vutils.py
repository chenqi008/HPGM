import torch
import math
irange = range

from miscc.config import cfg
import os
import cv2
import numpy as np

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, 
              scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), "range has to be a tuple (min, max) if specified. min and max are numbers"

        # def norm_ip(img, min, max):
        #     img.clamp_(min=min, max=max)
        #     img.add_(-min).div_(max - min + 1e-5)

        # def norm_range(t, range):
        #     if range is not None:
        #         norm_ip(t, range[0], range[1])
        #     else:
        #         # print('min', float(t.min()))
        #         # print('max', float(t.max()))
        #         # assert False
        #         norm_ip(t, float(t.min()), float(t.max()))

        def norm_ip(img):
            img.mul_(0.5).add_(0.5)

        def norm_range(t, range):
            norm_ip(t)

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)


    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def make_grid_bbox(tensor, box, nrow=8, padding=2,
              normalize=False, range=None, 
              scale_each=False, pad_value=0, draw_line=False):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """

    # make the mini-batch of images into a grid
    # nmaps = tensor.size(0)
    nmaps = len(box)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    # height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    height, width = int(256 + padding), int(256 + padding)
    tensor = torch.ones(())
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    # # add the white image into the grid
    # block = tensor.new_full((3, height - padding, width - padding), 9.0/13)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            # add the white image into the grid
            block = tensor.new_full((3, height - padding, width - padding), 9.0/13)
            # print(box[0].size())
            # print(box[1].size())
            # assert False
            # num_curr_box = box[0][k].size(0)
            num_curr_box = box[k][0].size(0)
            for z in irange(num_curr_box):
                # label = box[1][k][z].item()
                try:
                    label = box[k][1][z].item()
                except:
                    print(box)
                    print(k)
                    assert False
                
                if label != -1:
                    block = draw_box(block, box[k][0][z], label, draw_line)
                    # print(k, z)
                else:
                    break
            # copy to the grid
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(block)
            k = k + 1
    return grid


def draw_box(image, curr_box, label, draw_line=False):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    # y1, x1, y2, x2 = box
    # print(curr_box)
    # assert False
    x1, y1, x2, y2 = curr_box[0], curr_box[1], curr_box[2], curr_box[3]
    _, h, w = image.size()
    x1 = int(x1.item() * w)
    y1 = int(y1.item() * h)
    x2 = int(x2.item() * w)
    y2 = int(y2.item() * h)
    if draw_line:
        if x1 > x2:
           x1, x2 = x2, x1
        if y1 > y2:
           y1, y2 = y2, y1
        image[:, y1:y1 + 3, x1:x2] = label/13.0
        image[:, y2:y2 + 3, x1:x2] = label/13.0
        image[:, y1:y2, x1:x1 + 3] = label/13.0
        image[:, y1:y2, x2:x2 + 3] = label/13.0
    else:
        image[:, y1:y1 + 3, x1:x2] = label/13.0
        image[:, y2:y2 + 3, x1:x2] = label/13.0
        image[:, y1:y2, x1:x1 + 3] = label/13.0
        image[:, y1:y2, x2:x2 + 3] = label/13.0
    return image


def make_grid_floor_plan(tensor, box, nrow=8, padding=2,
              normalize=False, range=None, 
              scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    # make the mini-batch of images into a grid
    # nmaps = tensor.size(0)
    nmaps = len(box)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    # height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    height, width = int(256 + padding), int(256 + padding)
    tensor = torch.ones(())
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    # # add the white image into the grid
    # block = tensor.new_full((3, height - padding, width - padding), 9.0/13)

    wall_thickness = 2
    wall_symbol = 2.0

    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            # add the white image into the grid
            block = tensor.new_full((3, height - padding, width - padding), 9.0/13)
            num_curr_box = box[k][0].size(0)
            
            # sorted the box according to their size
            sorted_box = {}
            for z in irange(num_curr_box):
                curr_box = box[k][0][z]
                x1, y1, x2, y2 = curr_box[0], curr_box[1], curr_box[2], curr_box[3]
                sorted_box[z] = (x2-x1)*(y2-y1)
            # to get sorted id
            sorted_box = sorted(sorted_box.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

            # obtain the sorted box and corresponding label
            for m in irange(num_curr_box):
                # get sorted id
                z = sorted_box[m][0]
                # label = box[1][k][z].item()
                try:
                    label = box[k][1][z].item()
                except:
                    assert False
                # draw box in the current image
                if label != -1:
                    block = draw_floor_plan(block, box[k][0][z], label)
                    # print(k, z)
                else:
                    break

            # copy the current image to the grid
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(block)
            k = k + 1
    return grid

def draw_floor_plan(image, curr_box, label):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    wall_thickness = 2
    wall_symbol = 2.0
    x1, y1, x2, y2 = curr_box[0], curr_box[1], curr_box[2], curr_box[3]
    _, h, w = image.size()
    x1 = int(x1.item() * w)
    y1 = int(y1.item() * h)
    x2 = int(x2.item() * w)
    y2 = int(y2.item() * h)
    image[:, y1:y2, x1:x2] = label/13.0
    image[:, y1-wall_thickness:y1+wall_thickness, x1:x2] = wall_symbol
    image[:, y2-wall_thickness:y2+wall_thickness, x1:x2] = wall_symbol
    image[:, y1:y2, x1-wall_thickness:x1+wall_thickness] = wall_symbol
    image[:, y1:y2, x2-wall_thickness:x2+wall_thickness] = wall_symbol
    return image


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    # ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    ndarr = grid.mul_(13).add_(0.5).clamp_(0, 13).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # colorize the gray images
    # ndarr = cv2.applyColorMap(cv2.convertScaleAbs(ndarr, alpha=20.0), cv2.COLORMAP_JET)
    palette=[]
    for i in xrange(256):
        palette.extend((255,255,255))
    palette[:3*14]=np.array([[84, 139, 84],
                            [0, 100, 0],
                            [0, 0, 128],
                            [85, 26, 139],
                            [255, 0, 255],
                            [165, 42, 42],
                            [139, 134, 130],
                            [205, 198, 115],
                            [139, 58, 58],
                            [255, 255, 255],
                            [0, 0, 0],
                            [30, 144, 255],
                            [135, 206, 235],
                            [255, 255, 0]], dtype='uint8').flatten()

    im = Image.fromarray(ndarr).convert('L')
    # colorize
    im.putpalette(palette)
    im.save(filename)


def save_bbox(tensor, box, filename, nrow=8, padding=0,
               normalize=False, scale_each=False, pad_value=0, draw_line=False, save=True):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    
    # print(box)
    # assert False

    grid = make_grid_bbox(tensor, box, nrow=nrow, padding=padding, pad_value=pad_value,
                        normalize=normalize, scale_each=scale_each, draw_line=draw_line)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    # ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    ndarr = grid.mul_(13).add_(0.5).clamp_(0, 13).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # colorize the gray images
    # ndarr = cv2.applyColorMap(cv2.convertScaleAbs(ndarr, alpha=20.0), cv2.COLORMAP_JET)
    palette = []
    for i in range(256):
        palette.extend((255, 255, 255))
    palette[:3*14] = np.array([[84, 139, 84],
                            [0, 100, 0],
                            [0, 0, 128],
                            [85, 26, 139],
                            [255, 0, 255],
                            [165, 42, 42],
                            [139, 134, 130],
                            [205, 198, 115],
                            [139, 58, 58],
                            [255, 255, 255],
                            [0, 0, 0],
                            [30, 144, 255],
                            [135, 206, 235],
                            [255, 255, 0]], dtype='uint8').flatten()

    # # draw box
    # ndarr = draw_box(ndarr, box, palette)

    im = Image.fromarray(ndarr).convert('L')
    # colorize
    im.putpalette(palette)
    if save:
        im.save(filename)
    else:
        return im


def save_floor_plan(tensor, box, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image


    grid = make_grid_floor_plan(tensor, box, nrow=nrow, padding=padding, pad_value=pad_value,
                            normalize=normalize, range=range, scale_each=scale_each)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    # ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # ndarr = grid.mul_(13).add_(0.5).clamp_(0, 13).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    ndarr = grid.mul_(13).add_(0.5).clamp_(0, 14).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # colorize the gray images
    # ndarr = cv2.applyColorMap(cv2.convertScaleAbs(ndarr, alpha=20.0), cv2.COLORMAP_JET)
    palette = []
    for i in xrange(256):
        palette.extend((255, 255, 255))
    palette[:3*15] = np.array([[84, 139, 84],
                            [0, 100, 0],
                            [0, 0, 128],
                            [85, 26, 139],
                            [255, 0, 255],
                            [165, 42, 42],
                            [139, 134, 130],
                            [205, 198, 115],
                            [139, 58, 58],
                            [255, 255, 255],
                            [0, 0, 0],
                            [30, 144, 255],
                            [135, 206, 235],
                            [255, 255, 0],
                            [0, 0, 0]], dtype='uint8').flatten()#

    # # draw box
    # ndarr = draw_box(ndarr, box, palette)

    im = Image.fromarray(ndarr).convert('L')
    # colorize
    im.putpalette(palette)
    im.save(filename)



def save_image_for_fid(tensor_real, tensor_fake, save_path_real, 
                save_path_fake, step_test, batch_size,
                nrow=1, padding=0, normalize=False, 
                range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image

    num_imgs = tensor_real.size(0)
    for i in xrange(num_imgs):
        # save real images
        grid_real = make_grid(tensor_real[i], nrow=nrow, padding=padding, pad_value=pad_value,
                         normalize=normalize, range=range, scale_each=scale_each)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        # ndarr_real = grid_real.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        ndarr_real = grid_real.mul_(13).add_(0.5).clamp_(0, 13).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im_real = Image.fromarray(ndarr_real)
        filename_real = os.path.join(save_path_real, \
            '{:0>4}.png'.format(step_test*batch_size+i))
        im_real.save(filename_real)

        # save fake images
        grid_fake = make_grid(tensor_fake[i], nrow=nrow, padding=padding, pad_value=pad_value,
                         normalize=normalize, range=range, scale_each=scale_each)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        # ndarr_fake = grid_fake.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        ndarr_fake = grid_fake.mul_(13).add_(0.5).clamp_(0, 13).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im_fake = Image.fromarray(ndarr_fake)
        filename_fake = os.path.join(save_path_fake, \
            '{:0>4}.png'.format(step_test*batch_size+i))
        im_fake.save(filename_fake)



