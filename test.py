import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
# from models.networks import UNetEncoder, UNetDecoder, Classifier
from networkDemo import Classifier, UNetEncoder, UNetDecoder

import os
from PIL import Image
from dataset.dataset import *
from tqdm import tqdm
import random
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from collections import defaultdict
import click
from metrics.uniqm import uiqm

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x

def var_to_img(img):
    return (img * 255).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)

def test(fE, fI, dataloader, model_name, which_epoch):
    mse_scores = []
    ssim_scores = []
    psnr_scores = []
    uiqm_scores = []
    criterion_MSE = nn.MSELoss().cuda()

    for idx, data in tqdm(enumerate(dataloader)):
        uw_img, cl_img, water_type, name = data
        uw_img = Variable(uw_img).cuda()
        cl_img = Variable(cl_img, requires_grad=False).cuda()
        
        fE_out, enc_outs = fE(uw_img)
        #单
        # fI_out = to_img(fI(fE_out, enc_outs).detach())

        #双
        fI_out1, fI_out2, fI_out3 = fI(fE_out, enc_outs)
        fI_out1, fI_out2, fI_out3 = to_img(fI_out1.detach()), to_img(fI_out2.detach()), to_img(fI_out3.detach())
        fI_out = fI_out3
        enc_outs = None

        # save_image(torch.stack([uw_img.squeeze().cpu().data, fI_out.squeeze().cpu().data, cl_img.squeeze().cpu().data]), './results/{}/{}/{}_{}.jpg'.format(model_name, which_epoch, name[0], 'out'))
        # save_image(fI_out.cpu().data, 'C:/Libin/Dataset/compare_data/EUVP/Oursim/{}.jpg'.format(name[0]))
        # save_image(torch.stack([uw_img.squeeze().cpu().data, fI_out.squeeze().cpu().data, cl_img.squeeze().cpu().data]), 'C:/Libin/Dataset/compare_data/EUVP/Oursim_compare/{}.jpg'.format(name[0]))

        mse = criterion_MSE(fI_out, cl_img).item()
        mse_scores.append(mse)

        fI_out = (fI_out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        cl_img = (cl_img * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)

        ssim = ssim_fn(fI_out, cl_img, multichannel=True)
        psnr = psnr_fn(cl_img, fI_out)

        # uiqm_d = uiqm(fI_out)

        ssim_scores.append(ssim)
        psnr_scores.append(psnr)
        # uiqm_scores.append(uiqm_d)

    return ssim_scores, psnr_scores, mse_scores#, uiqm_scores

def write_to_log(log_file_path, status):
	"""
		Write to the log file
	"""

	with open(log_file_path, "a") as log_file:
		log_file.write(status+'\n')

@click.command()
@click.argument('name', default='realword-EUVP_imagenet2')
@click.option('--num_channels', default=3, help='Number of input image channels')
@click.option('--test_dataset', default='nyu', help='Name of the test dataset (nyu)')
# @click.option('--data_path', default='C:\\Libin\\Code\\Domain-Adversarial\\dataset\\raw-890\\', help='Path of training input data')
# @click.option('--label_path', default='C:\\Libin\\Dataset\\EUVP\\underwater_all\\trainB\\', help='Path of training label data')# @click.option('--data_path', default='C:\\Libin\\Code\\Dataset\\raw-890\\', help='Path of training input data')
# @click.option('--label_path', default='C:\\Libin\\Code\\Domain-Adversarial\\dataset\\reference-890\\', help='Path of training label data')
@click.option('--data_path', default='C:\\Libin\\Dataset\\EUVP\\underwater_imagenet\\trainA\\', help='Path of training input data')
@click.option('--label_path', default='C:\\Libin\\Dataset\\EUVP\\underwater_imagenet\\trainB\\', help='Path of training label data')
@click.option('--which_epoch', default=695, help='Test for this epoch')
@click.option('--test_size', default=100, help='Lambda for N loss')        #3000
@click.option('--fe_load_path', default="train\\fE_latest.pth", help='Load path for pretrained fN')
@click.option('--fi_load_path', default="train\\fI_latest.pth", help='Load path for pretrained fE')
def main(name, num_channels, test_dataset, data_path, label_path, which_epoch, test_size, fe_load_path, fi_load_path):

    if not os.path.exists('./results'):
        os.mkdir('./results')

    if not os.path.exists('./results/{}'.format(name)):
        os.mkdir('./results/{}'.format(name))

    if not os.path.exists('./results/{}/{}'.format(name, which_epoch)):
        os.mkdir('./results/{}/{}'.format(name, which_epoch))

    fE_load_path = fe_load_path
    fI_load_path = fi_load_path

    fE = UNetEncoder(num_channels).cuda()
    fI = UNetDecoder(num_channels).cuda()

    if which_epoch:
    #    fE.load_state_dict(torch.load(os.path.join('./checkpoints', name, 'fE_{}.pth'.format(which_epoch))))
    #    fI.load_state_dict(torch.load(os.path.join('./checkpoints', name, 'fI_{}.pth'.format(which_epoch))))
        fE.load_state_dict(torch.load(os.path.join('checkpoints', name, 'fE.pth'.format())))
        fI.load_state_dict(torch.load(os.path.join('checkpoints', name, 'fI.pth'.format())))
    else:
        fE.load_state_dict(torch.load(fE_load_path))
        fI.load_state_dict(torch.load(fI_load_path))

    fE.eval()
    fI.eval()

    if test_dataset=='nyu':
        test_dataset = NYUUWDataset(data_path, 
            label_path,
            size=test_size,      #3000
            test_start= 3300,     #33000
            mode='test')
    else:
        # Add more datasets
        test_dataset = NYUUWDataset(data_path, 
            label_path,
            size=3000,
            test_start=33000,
            mode='test')

    batch_size = 1
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ssim_scores, psnr_scores, mse_scores = test(fE, fI, dataloader, name, which_epoch)



    ssim_log = "Average SSIM: {}".format(sum(ssim_scores) / len(ssim_scores))
    psne_log = "Average PSNR: {}".format(sum(psnr_scores) / len(psnr_scores))
    mse_log = "Average MSE: {}".format(sum(mse_scores) / len(mse_scores))
    # uiqm_log = "Average uiqm: {}".format(sum(uiqm_scores) / len(uiqm_scores))
    print(ssim_log)
    print(psne_log)
    print(mse_log)
    # print(uiqm_log)
    # print ("Average SSIM: {}".format(sum(ssim_scores)/len(ssim_scores)))
    # print ("Average PSNR: {}".format(sum(psnr_scores)/len(psnr_scores)))
    # print ("Average MSE: {}".format(sum(mse_scores)/len(mse_scores)))

    log_path = 'logs.txt'
    # write_to_log(log_path, name)
    # write_to_log(log_path, ssim_log)
    # write_to_log(log_path, psne_log)
    # write_to_log(log_path, mse_log)
    # write_to_log(log_path, '\n')

if __name__== "__main__":
    main()