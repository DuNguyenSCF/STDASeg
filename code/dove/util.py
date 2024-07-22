import itertools, imageio, torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
# from scipy.misc import imresize
import cv2
import torch


# def show_result(G, x_, y_, num_epoch, show = False, save = False, path = 'result.png'):
#     # G.eval()
#     with torch.no_grad():
#         test_images = G(x_)

#     size_figure_grid = 3
#     fig, ax = plt.subplots(x_.size()[0], size_figure_grid, figsize=(6.5, 11), gridspec_kw={
#                            'width_ratios': [2, 2, 2],
#                            'height_ratios': [2, 2, 2, 2, 2],
#                        'wspace': 0.2,
#                        'hspace': 0.2})
#     for i, j in itertools.product(range(x_.size()[0]), range(size_figure_grid)):
#         ax[i, j].get_xaxis().set_visible(False)
#         ax[i, j].get_yaxis().set_visible(False)

#     for i in range(x_.size()[0]):
#         ax[i, 0].cla()
#         ax[i, 0].imshow((x_[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
#         ax[i, 1].cla()
#         ax[i, 1].imshow((test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
#         ax[i, 2].cla()
#         ax[i, 2].imshow((y_[i].numpy().transpose(1, 2, 0) + 1) / 2)

#     label = 'Epoch {0}'.format(num_epoch)
#     fig.text(0.5, 0.04, label, ha='center')

#     if save:
#         plt.savefig(path)

#     if show:
#         plt.show()
#     else:
#         plt.close()

def preprocess_input(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x
        
def show_result(G, paired_x_, x_, y_, num_epoch, show = False, save = False, path = 'result.png'):
    # G.eval()
    with torch.no_grad():
        test_images = G(paired_x_)

    size_figure_grid = 3
    fig, ax = plt.subplots(x_.size()[0], size_figure_grid, figsize=(9, 40), gridspec_kw={
                           'width_ratios': [2, 2, 2],
                           'height_ratios': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       'wspace': 0.2,
                       'hspace': 0.2})
    for i, j in itertools.product(range(x_.size()[0]), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for i in range(x_.size()[0]):
        ax[i, 0].cla()
        ax[i, 0].imshow((x_[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 1].cla()
        ax[i, 1].imshow((test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 2].cla()
        ax[i, 2].imshow((y_[i].numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
        
def show_result_mod(G, paired_x_, x_, x0_, y_, mask, num_epoch, show = False, save = False, path = 'result.png'):
    # G.eval()
    with torch.no_grad():
        test_images = G(paired_x_)
        # G_result_ = G(paired_x_)
        # test_images = G_result_*mask + x0_*(1-mask)

    size_figure_grid = 3
    fig, ax = plt.subplots(x_.size()[0], size_figure_grid, figsize=(9, 40), gridspec_kw={
                           'width_ratios': [2, 2, 2],
                           'height_ratios': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       'wspace': 0.2,
                       'hspace': 0.2})
    for i, j in itertools.product(range(x_.size()[0]), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for i in range(x_.size()[0]):
        ax[i, 0].cla()
        ax[i, 0].imshow((x_[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 1].cla()
        ax[i, 1].imshow((test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 2].cla()
        ax[i, 2].imshow((y_[i].numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
        
def show_result_3c(G, x_, y_, num_epoch, show = False, save = False, path = 'result.png'):
    # G.eval()
    with torch.no_grad():
        test_images = G(x_)

    size_figure_grid = 3
    fig, ax = plt.subplots(x_.size()[0], size_figure_grid, figsize=(6.5, 22), gridspec_kw={
                           'width_ratios': [2, 2, 2],
                           'height_ratios': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       'wspace': 0.2,
                       'hspace': 0.2})
    for i, j in itertools.product(range(x_.size()[0]), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for i in range(x_.size()[0]):
        ax[i, 0].cla()
        ax[i, 0].imshow((x_[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 1].cla()
        ax[i, 1].imshow((test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 2].cla()
        ax[i, 2].imshow((y_[i].numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def generate_animation(root, model, train_epoch):
    images = []
    for e in range(train_epoch):
        img_name = root + 'fixed_results/' + model + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(root + model + 'generate_animation.gif', images, fps=2)

def data_load(path, subfolder, transform, batch_size, shuffle=True):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]

    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1

        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)

# def imgs_resize(imgs, resize_scale = 286):
#     outputs = torch.FloatTensor(imgs.size()[0], imgs.size()[1], resize_scale, resize_scale)
#     for i in range(imgs.size()[0]):
#         img = imresize(imgs[i].numpy(), [resize_scale, resize_scale])
#         outputs[i] = torch.FloatTensor((img.transpose(2, 0, 1).astype(np.float32).reshape(-1, imgs.size()[1], resize_scale, resize_scale) - 127.5) / 127.5)

#     return outputs

# def imgs_resize(imgs, resize_scale = 286):
#     outputs = torch.FloatTensor(imgs.size()[0], imgs.size()[1], resize_scale, resize_scale)
#     for i in range(imgs.size()[0]):
#         img = cv2.resize(torch.permute(imgs[i],(1,2,0)).numpy(), (resize_scale, resize_scale))
#         outputs[i] = torch.FloatTensor((img.transpose(2, 0, 1).astype(np.float32).reshape(-1, imgs.size()[1], resize_scale, resize_scale) - 127.5) / 127.5)

#     return outputs

def imgs_resize(imgs, resize_scale = 286):
    outputs = torch.FloatTensor(imgs.size()[0], imgs.size()[1], resize_scale, resize_scale)
    for i in range(imgs.size()[0]):
        img = cv2.resize(torch.permute(imgs[i],(1,2,0)).numpy(), (resize_scale, resize_scale), interpolation = cv2.INTER_NEAREST)
        outputs[i] = torch.FloatTensor((img.transpose(2, 0, 1).astype(np.float32).reshape(-1, imgs.size()[1], resize_scale, resize_scale)))

    return outputs

# def random_crop(imgs1, imgs2, crop_size = 256):
#     outputs1 = torch.FloatTensor(imgs1.size()[0], imgs1.size()[1], crop_size, crop_size)
#     outputs2 = torch.FloatTensor(imgs2.size()[0], imgs2.size()[1], crop_size, crop_size)
#     for i in range(imgs1.size()[0]):
#         img1 = imgs1[i]
#         img2 = imgs2[i]
#         rand1 = np.random.randint(0, imgs1.size()[2] - crop_size)
#         rand2 = np.random.randint(0, imgs2.size()[2] - crop_size)
#         outputs1[i] = img1[:, rand1: crop_size + rand1, rand2: crop_size + rand2]
#         outputs2[i] = img2[:, rand1: crop_size + rand1, rand2: crop_size + rand2]

#     return outputs1, outputs2

# def random_fliplr(imgs1, imgs2):
#     outputs1 = torch.FloatTensor(imgs1.size())
#     outputs2 = torch.FloatTensor(imgs2.size())
#     for i in range(imgs1.size()[0]):
#         if torch.rand(1)[0] < 0.5:
#             img1 = torch.FloatTensor(
#                 (np.fliplr(imgs1[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs1.size()[1], imgs1.size()[2], imgs1.size()[3]) + 1) / 2)
#             outputs1[i] = (img1 - 0.5) / 0.5
#             img2 = torch.FloatTensor(
#                 (np.fliplr(imgs2[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs2.size()[1], imgs2.size()[2], imgs2.size()[3]) + 1) / 2)
#             outputs2[i] = (img2 - 0.5) / 0.5
#         else:
#             outputs1[i] = imgs1[i]
#             outputs2[i] = imgs2[i]

#     return outputs1, outputs2


def random_crop(imgs1, imgs2, imgs3, crop_size = 256):
    outputs1 = torch.FloatTensor(imgs1.size()[0], imgs1.size()[1], crop_size, crop_size)
    outputs2 = torch.FloatTensor(imgs2.size()[0], imgs2.size()[1], crop_size, crop_size)
    outputs3 = torch.FloatTensor(imgs3.size()[0], imgs3.size()[1], crop_size, crop_size)
    for i in range(imgs1.size()[0]):
        img1 = imgs1[i]
        img2 = imgs2[i]
        img3 = imgs3[i]
        rand1 = np.random.randint(0, imgs1.size()[2] - crop_size)
        rand2 = np.random.randint(0, imgs2.size()[2] - crop_size)
        outputs1[i] = img1[:, rand1: crop_size + rand1, rand2: crop_size + rand2]
        outputs2[i] = img2[:, rand1: crop_size + rand1, rand2: crop_size + rand2]
        outputs3[i] = img3[:, rand1: crop_size + rand1, rand2: crop_size + rand2]

    return outputs1, outputs2, outputs3

def random_fliplr(imgs1, imgs2, imgs3):
    outputs1 = torch.FloatTensor(imgs1.size())
    outputs2 = torch.FloatTensor(imgs2.size())
    outputs3 = torch.FloatTensor(imgs3.size())
    for i in range(imgs1.size()[0]):
        if torch.rand(1)[0] < 0.5:
            img1 = torch.FloatTensor(
                (np.fliplr(imgs1[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs1.size()[1], imgs1.size()[2], imgs1.size()[3]) + 1) / 2)
            outputs1[i] = (img1 - 0.5) / 0.5
            img2 = torch.FloatTensor(
                (np.fliplr(imgs2[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs2.size()[1], imgs2.size()[2], imgs2.size()[3]) + 1) / 2)
            outputs2[i] = (img2 - 0.5) / 0.5
            img3 = torch.FloatTensor(
                (np.fliplr(imgs3[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs3.size()[1], imgs3.size()[2], imgs3.size()[3]) + 1) / 2)
            outputs3[i] = (img3 - 0.5) / 0.5
        else:
            outputs1[i] = imgs1[i]
            outputs2[i] = imgs2[i]
            outputs3[i] = imgs3[i]

    return outputs1, outputs2, outputs3
