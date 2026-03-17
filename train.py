import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import os
from utils import train_model, eval_model, SmoothedDataset
from attack_lib import attack_setting
from torch.utils.tensorboard import SummaryWriter
from utils import get_sigma
import argparse
import matplotlib.pyplot as plt
import PIL


parser = argparse.ArgumentParser()
# Dataset Setting
parser.add_argument('--dataset', type=str, default='cifar')
parser.add_argument('--pair_id', type=int, default=0)

# Trojan Attack Setting
parser.add_argument('--atk_method', type=str, default='fourpixel')
parser.add_argument('--poison_r', type=float, default=0.02)
parser.add_argument('--delta', type=float, default=0.1)

# Smoothing Setting
parser.add_argument('--N_m', type=int, default=1000)
parser.add_argument('--sigma', type=float, default=0.5)
parser.add_argument('--dldp_sigma', type=float, default=0.0)
parser.add_argument('--dldp_gnorm', type=float, default=5.0)
parser.add_argument('--iter_sig_tr', type=int, default=1)
parser.add_argument('--iter_sig_ts', type=int, default=1)
parser.add_argument('--iter_sig_after', type=int, default=100)
parser.add_argument('--num_noise_vec', type=int, default=1)
parser.add_argument('--epoch_switch', type=int, default=0)


def optimize_sigma(model, loader, sigma_0, lr_sigma, iters_sig, flag='train', radius=None):
    model = model.eval()
    total = 0
    test_loss, test_loss_corrupted = 0, 0
    correct, correct_corrupted = 0, 0
    device = "cuda:0"
    for epoch in range(iters_sig):
        for _, (batch, targets, idx) in enumerate(loader):
            batch, targets = batch.to(device), targets.to(device)  # Here I will put iters to 1 as the outer loop contains the number of iterations
            sigma, batch_corrupted, rad = get_sigma(model, batch, lr_sigma, sigma_0[idx], 1, device, ret_radius=True)
            sigma_0, radius = sigma_0.to(device), radius.to(device)
            sigma_0[idx], radius[idx] = sigma, rad

            with torch.no_grad():
                outputs_softmax = model(batch).squeeze(1)
                outputs_corrputed_softmax = model(batch_corrupted).squeeze(1)

            targets = targets.float()
            loss = compute_loss(outputs_softmax, targets)
            loss_corrupted = compute_loss(outputs_corrputed_softmax, targets)

            test_loss += loss.item() * len(batch)
            test_loss_corrupted += loss_corrupted.item() * len(batch)

            targets = targets.to("cpu")

            correct += ((outputs_softmax > 0).cpu().long().eq(targets)).sum().item()
            correct_corrupted += ((outputs_corrputed_softmax > 0).cpu().long().eq(targets)).sum().item()
            total += targets.size(0)


        # Saving the sigmas
    return sigma_0

def compute_loss(outputs_softmax, targets):
    #outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
    return F.binary_cross_entropy_with_logits(outputs_softmax, targets)



if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    print (args)
    sigma = args['sigma']
    epoch_switch = args['epoch_switch']
    N_m = args['N_m']
    atk_method = args['atk_method']


    use_gpu = True
    dataset = args['dataset']
    if not os.path.isdir('./model'):
        os.mkdir('./model')
    if not os.path.isdir(f'./model/{args.dataset}'):
        os.mkdir(f'./model/{args.dataset}')
    if not os.path.isdir(f'./model/{args.dataset}/{args.wm_shape}'):
        os.mkdir(f'./model/{args.dataset}/{args.wm_shape}')
    if not os.path.isdir(f'./model/{args.dataset}/{args.wm_shape}/{args.sigma}'):
        os.mkdir(f'./model/{args.dataset}/{args.wm_shape}/{args.sigma}')

    PREFIX = './saved_model/_%s _%s sigma%s_epoch_switch%s' % (dataset, atk_method, sigma, epoch_switch)

    #PREFIX = './saved_model/_%s _%s sigma%s' % (dataset, atk_method, sigma)

    if not os.path.isdir(PREFIX):
        os.makedirs(PREFIX)

    base_bath_save = './base_bath_save'

    poisoned_train, testloader_benign, testloader_poison, BATCH_SIZE, N_EPOCH, LR, model = attack_setting(args)
    model = model(use_gpu)
    print(len(poisoned_train))
    print(len(testloader_poison))

    print('______________________________________________')
    ################ cifar_train 10000 test 2000    imagenet_train 20000 test 5000  minst 60000 and 2115
    radius_test = torch.zeros(2115)
    sigma_train = torch.ones(60000) * sigma
    sigma_testb = torch.ones(2115) * sigma
    sigma_test = torch.ones(2115) * sigma
    best_acc = 0.0


    if not os.path.isdir(PREFIX):
        os.makedirs(PREFIX)

    for epoch in range(N_m):
        # trainset = SmoothedDataset(poisoned_train, args['sigma'])
        trainloader = torch.utils.data.DataLoader(poisoned_train, batch_size=BATCH_SIZE, shuffle=True)
        save_path = PREFIX + '/smoothed_%d.model' % epoch

        if epoch >= epoch_switch:
            print('Switched to DRAB training, sigma train is {} and sigma test is {}'.format(sigma_train.mean().item(),
                                                                                             sigma_test.mean().item()))

            sigma_train = train_model(model, trainloader, lr=LR, sigma_0=sigma_train, epoch=epoch,
                                      iters_sig=args['iter_sig_tr'],
                                      dldp_setting=(args['dldp_sigma'], args['dldp_gnorm']), verbose=False)
            torch.save(model.state_dict(), save_path)
            #acc_benign, sigma_testb = eval_model(model, testloader_benign, sigma_testb, iters_sig=args['iter_sig_ts'])
            acc_poison, sigma_test = eval_model(model, testloader_poison, sigma_test, iters_sig=args['iter_sig_ts'])
            #print("Benign/Poison ACC %.4f/%.4fs" % (acc_benign, acc_poison))  # 打印出模型在良性和有毒数据上的准确率，并指出模型参数保存的位置和当前时间。
            print("Benign/Poison %.4fs" % (acc_poison))
        else:
            print('Training with RAB')

            sigma_train = train_model(model, trainloader, lr=LR, sigma_0=sigma_train, epoch=epoch, iters_sig=0,
                                      dldp_setting=(args['dldp_sigma'], args['dldp_gnorm']), verbose=False)
            torch.save(model.state_dict(), save_path)
            #acc_benign, sigma_testb = eval_model(model, testloader_benign, sigma_testb, 0)
            acc_poison, sigma_test = eval_model(model, testloader_poison, sigma_test, 0)
            print("Benign/Poison ACC %.4fs" % ( acc_poison))  # 打印出模型在良性和有毒数据上的准确率，并指出模型参数保存的位置和当前时间。

    print('Training has finished now we optimize for the sigmas. So far, sigma train is {} and sigma test is {}'.format(
        sigma_train.mean().item(), sigma_test.mean().item()))

    '''iters_sig = [200, 300, 400]
    # sigma_train = optimize_sigma(model, train_loader, writer, sigma_train, args.lr_sig, args.iter_sig_after, flag='train', radius = radius_train)
    for iter_sig_after in iters_sig:
        sigma_test = optimize_sigma(model, testloader_poison, sigma_test, 0.0001, iters_sig=iter_sig_after, flag='test',
                                    radius=radius_test)
        # sigma_testb = optimize_sigma(model, testloader_benign, sigma_testb, 0.0001, iters_sig=args['iter_sig_after'],flag='test',radius=radius_test)
        print('optimize_sigma, over')
        print(' sigma test is {}'.format(sigma_test.mean().item()))
        # print(' sigma testb is {}'.format(sigma_testb.mean().item()))

        # #Saving the sigmas
        # torch.save(sigma_train, base_bath_save + '/sigma_train.pth')
        torch.save(sigma_test, base_bath_save + '/_%s_%s_sigma%s_iter_sig_after%s.pth' % (dataset, atk_method, sigma, iter_sig_after))'''


    #torch.save(sigma_testb, base_bath_save + '/sigma_testb%.4f.pth'%sigma)
    sigma_test = optimize_sigma(model, testloader_poison, sigma_test, 0.0001,  iters_sig=args['iter_sig_after'], flag='test',
                                radius=radius_test)
    # sigma_testb = optimize_sigma(model, testloader_benign, sigma_testb, 0.0001, iters_sig=args['iter_sig_after'],flag='test',radius=radius_test)
    print('optimize_sigma, over')
    print(' sigma test is {}'.format(sigma_test.mean().item()))
    # print(' sigma testb is {}'.format(sigma_testb.mean().item()))

    # #Saving the sigmas
    # torch.save(sigma_train, base_bath_save + '/sigma_train.pth')
    torch.save(sigma_test,
               base_bath_save + '/_%s_%s_sigma%s_iter.pth' % (dataset, atk_method, sigma))