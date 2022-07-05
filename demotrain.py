import torch
import torchvision
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
# from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
# from skimage.measure import compare_psnr as psnr
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from PIL import Image
from dataset.dataset import NYUUWDataset
from tqdm import tqdm
import random
from torchvision import models
import numpy as np
# from models.networks import Classifier, UNetEncoder, UNetDecoder
from networkDemo import Classifier, UNetEncoder, UNetDecoder
import click
import datetime


def to_img(x):
    """
        Convert the tanh (-1 to 1) ranged tensor to image (0 to 1) tensor
        将tanh (-1 to 1)范围张量转换为image (0 to 1)张量
    """

    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)  # 将随机变化的数值限定在一个给定的区间区间内
    x = x.view(x.size(0), 3, 256, 256)  # x.size(0)=4 返回和原tensor数据个数相同，但size不同的tensor

    return x


# 返回False，
def set_requires_grad(nets, requires_grad=False):
    """
        Make parameters of the given network not trainable
    """

    if not isinstance(nets, list):  # 返回一个对象是类的实例还是其子类的实例。
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    return requires_grad


# 计算验证集的SSIM、PSNR
def compute_val_metrics(fE, fI, fN, dataloader, no_adv_loss):
    """
        Compute SSIM, PSNR scores for the validation set
    """
    """	如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
        其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
        而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接
        model.train() ：启用 BatchNormalization 和 Dropout
        model.eval() ：不启用 BatchNormalization 和 Dropout
        """
    fE.eval()
    fI.eval()
    fN.eval()

    mse_scores = []
    ssim_scores = []
    psnr_scores = []
    corr = 0

    criterion_MSE = nn.MSELoss().cuda()
    # criterion_MSE = nn.MSELoss().cpu()  # mse损失

    # tqdm模块是python进度条库, 基于迭代对象运行: tqdm(iterator)
    for idx, data in tqdm(enumerate(dataloader)):
        uw_img, cl_img, water_type, _ = data
        uw_img = Variable(uw_img).cuda()
        # uw_img = Variable(uw_img)
        cl_img = Variable(cl_img, requires_grad=False).cuda()
        # cl_img = Variable(cl_img, requires_grad=False)
        """torch.autograd.Variable是Autograd的核心类，它封装了Tensor，并整合了反向传播的相关实现
            Variable和tensor的区别和联系
            Variable是篮子，而tensor是鸡蛋，鸡蛋应该放在篮子里才能方便拿走（定义variable时一个参数就是tensor）
            Variable这个篮子里除了装了tensor外还有requires_grad参数，表示是否需要对其求导，默认为False
            Variable这个篮子呢，自身有一些属性
            比如grad，梯度variable.grad是d(y) / d(variable)
            保存的是变量y对variable变量的梯度值，如果requires_grad参数为False，所以variable.grad返回值为None，
        如果为True，返回值就为对variable的梯度值
            比如grad_fn，对于用户自己创建的变量（Variable()）grad_fn是为none的，也就是不能调用backward函数，
        但对于由计算生成的变量，如果存在一个生成中间变量的requires_grad为true，那其的grad_fn不为none，反则为none
"""
        fE_out, enc_outs = fE(uw_img)  # encoder：return x5, (x1, x2, x3, x4)
        # fI_out = to_img(fI(fE_out, enc_outs))	#decoder：return nn.Tanh()(x)		3channel
        # fN_out = F.softmax(fN(fE_out), dim=1)	#fN(fE_out)=FN(x5) (水类型)
        fN_out = fN(fE_out)
        fN_out = F.softmax(fN_out, dim=1)
        fI_out = to_img(fI(fE_out, enc_outs))

        if int(fN_out.max(1)[1].item()) == int(water_type.item()):  # 计算水类型正确的次数
            corr += 1

        mse_scores.append(criterion_MSE(fI_out, cl_img).item())
        """
            .squeeze(0) 			压缩维度，将第一维度，即第一个[]去掉
            .data.numpy() 			把tensor数据转换为numpy
            .transpose() 			是改变序列
            .astype(np.uint8)		转换数据类型
        """
        fI_out = (fI_out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        cl_img = (cl_img * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        #	print(fI_out.shape, cl_img.shape)

        """multichannel: 如果为True，则将数组的最后一维视为通道。相似对每个通道独立进行计算，然后求平均值。"""
        ssim_scores.append(ssim(fI_out, cl_img, multichannel=True))
        psnr_scores.append(psnr(cl_img, fI_out))

    fE.train()
    fI.train()
    if not no_adv_loss:  # no_adv_loss调用输入
        fN.train()

    return sum(ssim_scores) / len(dataloader), sum(psnr_scores) / len(dataloader), sum(mse_scores) / len(
        dataloader), corr / len(dataloader)


def backward_I_loss(fI, fE_out, enc_outs, cl_img, criterion_MSE, optimizer_fI, lambda_I_loss, retain_graph):
    """
        Backpropagate the reconstruction loss
        reconstruction loss 是由encoder生成得到图像与给定输入图像X的清晰图像真值之间的均方误差
        encoder和decoder之间的梯度
    """

    fI_out = to_img(fI(fE_out, enc_outs))		#将tanh (-1 to 1)范围张量转换为image (0 to 1)张量
    # fI_out = to_img(fI(inp.detach(), enc_outs))  # 将tanh (-1 to 1)范围张量转换为image (0 to 1)张量
    I_loss = criterion_MSE(fI_out, cl_img) * lambda_I_loss
    optimizer_fI.zero_grad()
    I_loss.backward(retain_graph=retain_graph)
    optimizer_fI.step()

    return fI_out, I_loss


def backward_N_loss(fN, fE_out, actual_target, criterion_CE, optimizer_fN, lambda_N_loss):
    """
        Backpropagate the nuisance loss
        Nuisance loss：预测的水类型分布和目标水类型分布的交叉熵，Nuisance loss被反向传播用来更新classifier
        classfiler水类型梯度
    """

    fN_out = fN(fE_out.detach())
    N_loss = criterion_CE(fN_out, actual_target) * lambda_N_loss
    optimizer_fN.zero_grad()
    N_loss.backward()
    optimizer_fN.step()

    return N_loss


def backward_adv_loss(fN, fE_out, lambda_adv_loss, num_classes, neg_entropy):
    """
        Backpropagate the adversarial loss
        Adversarial loss：减少分类器预测的确定性或负熵，来增加分类器的不确定性或熵。
        所以，对于输入图像X的潜在向量Z，计算对抗损失LA，这是来自分类器D的预测谁类型分布的负熵。
        该loss被反向传播用于更新encoder E：
        encoder和classfiler梯度
    """

    fN_out = fN(fE_out)
    adv_loss = calc_adv_loss(fN_out, num_classes, neg_entropy) * lambda_adv_loss
    adv_loss.backward()

    return adv_loss


def write_to_log(log_file_path, status):
    """
        Write to the log file
    """

    with open(log_file_path, "a") as log_file:
        log_file.write(status + '\n')


def calc_adv_loss(fN_out, num_classes, neg_entropy):
    """
        Calculate the adversarial loss (negative entropy or cross entropy with uniform distribution)
    """

    if neg_entropy:
        fN_out_softmax = F.softmax(fN_out, dim=1)
        return torch.mean(torch.sum(fN_out_softmax * torch.log(torch.clamp(fN_out_softmax, min=1e-10, max=1.0)), 1))
    else:
        fN_out_log_softmax = F.log_softmax(fN_out, dim=1)
        return -torch.mean(torch.div(torch.sum(fN_out_log_softmax, 1), num_classes))


@click.command()
@click.argument('name', default='train')
@click.option('--data_path', default='dataset\\raw-890\\', help='Path of training input data')
@click.option('--label_path', default='dataset\\reference-890\\', help='Path of training label data')
@click.option('--learning_rate', default=1e-3, help='Learning rate')
@click.option('--batch_size', default=4, help='Batch size')
@click.option('--save_interval', default=5, help='Save models after this many epochs')
@click.option('--start_epoch', default=1, help='Start training from this epoch')
@click.option('--end_epoch', default=200, help='Train till this epoch')
@click.option('--num_classes', default=6, help='Number of water types')
@click.option('--num_channels', default=3, help='Number of input image channels')
@click.option('--train_size', default=50, help='Size of the training dataset')
@click.option('--test_size', default=20, help='Size of the testing dataset')
@click.option('--val_size', default=20, help='Size of the validation dataset')
@click.option('--fe_load_path', default=None, help='Load path for pretrained fN')
@click.option('--fi_load_path', default=None, help='Load path for pretrained fE')
@click.option('--fn_load_path', default=None, help='Load path for pretrained fN')
@click.option('--lambda_i_loss', default=100.0, help='Lambda for I loss')
@click.option('--lambda_n_loss', default=1.0, help='Lambda for N loss')
@click.option('--lambda_adv_loss', default=1.0, help='Lambda for adv loss')
@click.option('--fi_threshold', default=0.9, help='Train fI till this threshold')
@click.option('--fn_threshold', default=0.85, help='Train fN till this threshold')
@click.option('--continue_train', is_flag=True, help='Continue training from start_epoch')
@click.option('--neg_entropy', default=True,
              help='Use KL divergence instead of cross entropy with uniform distribution')
@click.option('--no_adv_loss', is_flag=True, help='Use adversarial loss during training or not')
def main(name, data_path, label_path, learning_rate, batch_size, save_interval, start_epoch, end_epoch, num_classes,
         num_channels,
         train_size, test_size, val_size, fe_load_path, fi_load_path, fn_load_path, lambda_i_loss, lambda_n_loss,
         lambda_adv_loss,
         fi_threshold, fn_threshold, continue_train, neg_entropy, no_adv_loss):
    fE_load_path = fe_load_path  # class
    fI_load_path = fi_load_path  # encoder
    fN_load_path = fn_load_path  # decoder

    lambda_I_loss = lambda_i_loss
    lambda_N_loss = lambda_n_loss

    fI_threshold = fi_threshold
    fN_threshold = fn_threshold

    # Define datasets and dataloaders
    train_dataset = NYUUWDataset(data_path,
                                 label_path,
                                 size=train_size,
                                 train_start=0,
                                 mode='train')

    val_dataset = NYUUWDataset(data_path,
                               label_path,
                               size=val_size,
                               val_start=51,
                               mode='val')

    test_dataset = NYUUWDataset(data_path,
                                label_path,
                                size=test_size,
                                test_start=71,
                                mode='test')
    # DataLoader本质上就是一个iterable（跟python的内置类型list等一样），并利用多进程来加速batch data的处理，使用yield来使用有限的内存
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # 对抗损失
    if not no_adv_loss:  # True
        """
            Define the nuisance classifier to include the adversarial loss in the model
        """

        fN = Classifier(num_classes).cuda()
        # fN = Classifier(num_classes).cpu()  # FN: classfier 输出type
        fN_req_grad = True
        fN.train()
        criterion_CE = nn.CrossEntropyLoss().cuda()
        # criterion_CE = nn.CrossEntropyLoss()
        optimizer_fN = torch.optim.Adam(fN.parameters(), lr=learning_rate,
                                        weight_decay=1e-5)

    # Define models, criterion and optimizers
    fE = UNetEncoder(num_channels).cuda()
    # fE = UNetEncoder(num_channels).cpu()  # FE: encoder
    fI = UNetDecoder(num_channels).cuda()
    # fI = UNetDecoder(num_channels).cpu()  # FI: decoder

    criterion_MSE = nn.MSELoss().cuda()
    # criterion_MSE = nn.MSELoss().cpu()
    # 优化
    optimizer_fE = torch.optim.Adam(fE.parameters(), lr=learning_rate,
                                    weight_decay=1e-5)
    optimizer_fI = torch.optim.Adam(fI.parameters(), lr=learning_rate,
                                    weight_decay=1e-5)

    fE.train()
    fI.train()

    if continue_train:
        """
            Load pretrained models to continue training
        """
        #
        if fE_load_path:
            fE.load_state_dict(torch.load(fE_load_path))
            print('Loaded fE from {}'.format(fE_load_path))
        if fI_load_path:
            fI.load_state_dict(torch.load(fI_load_path))
            print('Loaded fI from {}'.format(fI_load_path))
        if not no_adv_loss and fN_load_path:
            fN.load_state_dict(torch.load(fN_load_path))
            print('Loaded fN from {}'.format(fN_load_path))

    if not os.path.exists('./checkpoints/{}'.format(name)):
        os.mkdir('./checkpoints/{}'.format(name))

    log_file_path = './checkpoints/{}/log_file.txt'.format(name)

    # 训练开始时间
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    status = '\nTRAINING SESSION STARTED ON {}\n'.format(now)
    write_to_log(log_file_path, status)

    # error
    # Compute the initial cross validation scores
    if continue_train and not no_adv_loss:  # 计算验证集的ssim、psnr，fN_val_acc水类型的正确率
        fI_val_ssim, _, _, fN_val_acc = compute_val_metrics(fE, fI, fN, val_dataloader, no_adv_loss)
    else:
        fI_val_ssim = -1
        fN_val_acc = -1

    # Train only the encoder-decoder upto a certain threshold
    while fI_val_ssim < fI_threshold and not continue_train:  # ssim小于初始值

        epoch = start_epoch

        # print('fI_val_ssim < fI_threshold and not continue_train')

        status = 'Avg fI val SSIM: {}, Avg fN val acc: {}\nCurrent fI threshold: {}, Current fN threshold: {}'.format(
            fI_val_ssim, fN_val_acc, fI_threshold, fN_threshold)
        print(status)  # 输出fI_val_ssim, fN_val_acc, fI_threshold, fN_threshold
        write_to_log(log_file_path, status)
        # tqdm	装饰一个可迭代对象，返回一个精确操作的迭代器
        for idx, data in tqdm(enumerate(train_dataloader)):
            uw_img, cl_img, water_type, _ = data
            uw_img = Variable(uw_img).cuda()
            # uw_img = Variable(uw_img)
            cl_img = Variable(cl_img, requires_grad=False).cuda()
            # cl_img = Variable(cl_img, requires_grad=False)

            fE_out, enc_outs = fE(uw_img)  # encoder  return x5, (x1, x2, x3, x4)
            optimizer_fE.zero_grad()  # 设置梯度为none
            '''反向传播	I_loss = criterion_MSE(fI_out, cl_img) * lambda_I_loss
            fI_out,  fI_out = to_img(fI(fE_out, enc_outs))
            I_loss'''
            fI_out, I_loss = backward_I_loss(fI, fE_out, enc_outs, cl_img, criterion_MSE, optimizer_fI,
                                             lambda_I_loss, retain_graph=not no_adv_loss)

            progress = "\tEpoch: {}\tIter: {}\tI_loss: {}".format(epoch, idx, I_loss.item())

            # 单个优化，更新参数
            optimizer_fE.step()

            if idx % 50 == 0:
                save_image(uw_img.cpu().data, 'results/uw_img.png')
                save_image(fI_out.cpu().data, 'results/fI_out.png')

                print(progress)
                write_to_log(log_file_path, progress)

            break

        """保存一个序列化（serialized）的目标到磁盘。函数使用了Python的pickle程序用于序列化。
        模型（models），张量（tensors）和文件夹（dictionaries）都是可以用这个函数保存的目标类型。"""
        torch.save(fE.state_dict(), './checkpoints/{}/fE_latest.pth'.format(name))
        torch.save(fI.state_dict(), './checkpoints/{}/fI_latest.pth'.format(name))
        if epoch % save_interval == 0:
            torch.save(fE.state_dict(), './checkpoints/{}/fE_{}.pth'.format(name, epoch))
            torch.save(fI.state_dict(), './checkpoints/{}/fI_{}.pth'.format(name, epoch))

        status = 'End of epoch. Models saved.'
        print(status)
        write_to_log(log_file_path, status)
        # sum(ssim_scores)/len(dataloader), sum(psnr_scores)/len(dataloader), sum(mse_scores)/len(dataloader), corr/len(dataloader)
        # 计算正确率
        fI_val_ssim, _, _, fN_val_acc = compute_val_metrics(fE, fI, fN, val_dataloader, no_adv_loss)

        start_epoch += 1

        break

    demo = 1

    for epoch in range(start_epoch, end_epoch):
        """
            Main training loop
        """
        if not no_adv_loss:  # true
            """
                Print the current cross-validation scores
            """
            print('if not no_adv_loss:')
            status = 'Avg fI val SSIM: {}, Avg fN val acc: {}\nCurrent fI threshold: {}, Current fN threshold: {}'.format(
                fI_val_ssim, fN_val_acc, fI_threshold, fN_threshold)
            print(status)
            write_to_log(log_file_path, status)

        for idx, data in tqdm(enumerate(train_dataloader)):
            uw_img, cl_img, water_type, _ = data
            uw_img = Variable(uw_img).cuda()
            cl_img = Variable(cl_img, requires_grad=False).cuda()
            actual_target = Variable(water_type, requires_grad=False).cuda()
            # uw_img = Variable(uw_img)
            # cl_img = Variable(cl_img, requires_grad=False)
            # actual_target = Variable(water_type, requires_grad=False)

            fE_out, enc_outs = fE(uw_img)

            #	N_loss = backward_N_loss(fN, fE_out, actual_target, criterion_CE, optimizer_fN, lambda_N_loss)

            # 训练encoder-decoder
            # fI_val_ssim = -1 		fI_threshold = 0.9
            if demo == 1:
                # if fI_val_ssim < fI_threshold:
                """
                    Train the encoder-decoder only
                    encoder和decoder的重构损失，fI_out是decoder的输出，I_loss是MSE loss
                    优化encoder
                """
                print('demo1')
                optimizer_fE.zero_grad()
                fI_out, I_loss = backward_I_loss(fI, fE_out, enc_outs, cl_img, criterion_MSE, optimizer_fI,
                                                 lambda_I_loss, retain_graph=not no_adv_loss)

                if not no_adv_loss:  # True
                    if fN_req_grad:
                        fN_req_grad = set_requires_grad(fN, requires_grad=False)
                    adv_loss = backward_adv_loss(fN, fE_out, lambda_adv_loss, num_classes, neg_entropy)
                    progress = "\tEpoch: {}\tIter: {}\tI_loss: {}\tadv_loss: {}".format(epoch, idx, I_loss.item(),
                                                                                        adv_loss.item())
                else:
                    progress = "\tEpoch: {}\tIter: {}\tI_loss: {}".format(epoch, idx, I_loss.item())

                optimizer_fE.step()

                if idx % 50 == 0:
                    save_image(uw_img.cpu().data, 'results/uw_img.png')
                    save_image(fI_out.cpu().data, 'results/fI_out.png')

                print(fN_req_grad)
            # 训练classifier
            if demo == 1:
                # elif fN_val_acc < fN_threshold:
                """
                    Train the nusiance classifier only
                """
                print('demo2')
                if not fN_req_grad:  # fN_req_grad=false
                    fN_req_grad = set_requires_grad(fN, requires_grad=True)
                """
                    计算由encoder的输出产生的向量，classfiler判别后与真实type的差
                """
                N_loss = backward_N_loss(fN, fE_out, actual_target, criterion_CE, optimizer_fN, lambda_N_loss)
                progress = "\tEpoch: {}\tIter: {}\tN_loss: {}".format(epoch, idx, N_loss.item())

            if demo == 1:
                # else:
                # 训练encoder
                """
                    Train the encoder-decoder only
                """
                print('demo1')
                optimizer_fE.zero_grad()
                fI_out, I_loss = backward_I_loss(fI, fE_out, enc_outs, cl_img, criterion_MSE, optimizer_fI,
                                                 lambda_I_loss, retain_graph=not no_adv_loss)

                if not no_adv_loss:
                    if fN_req_grad:
                        fN_req_grad = set_requires_grad(fN, requires_grad=False)
                    adv_loss = backward_adv_loss(fN, fE_out, lambda_adv_loss, num_classes, neg_entropy)

                    progress = "\tEpoch: {}\tIter: {}\tI_loss: {}\tadv_loss: {}".format(epoch, idx, I_loss.item(),
                                                                                        adv_loss.item())

                else:
                    progress = "\tEpoch: {}\tIter: {}\tI_loss: {}".format(epoch, idx, I_loss.item())

                optimizer_fE.step()

                if idx % 50 == 0:
                    save_image(uw_img.cpu().data, 'results/uw_img.png')
                    save_image(fI_out.cpu().data, 'results/fI_out.png')

            if idx % 50 == 0:
                print(progress)
                write_to_log(log_file_path, progress)

            break

        # Save models
        torch.save(fE.state_dict(), './checkpoints/{}/fE_latest.pth'.format(name))
        torch.save(fI.state_dict(), './checkpoints/{}/fI_latest.pth'.format(name))
        if not no_adv_loss:
            torch.save(fN.state_dict(), './checkpoints/{}/fN_latest.pth'.format(name))

        if epoch % save_interval == 0:
            torch.save(fE.state_dict(), './checkpoints/{}/fE_{}.pth'.format(name, epoch))
            torch.save(fI.state_dict(), './checkpoints/{}/fI_{}.pth'.format(name, epoch))
            if not no_adv_loss:
                torch.save(fN.state_dict(), './checkpoints/{}/fN_{}.pth'.format(name, epoch))

        status = 'End of epoch. Models saved.'
        print(status)
        write_to_log(log_file_path, status)

        if not no_adv_loss:
            """
                Compute the cross validation scores after the epoch
            """
            fI_val_ssim, _, _, fN_val_acc = compute_val_metrics(fE, fN, val_dataloader, no_adv_loss)


if __name__ == "__main__":
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    main()