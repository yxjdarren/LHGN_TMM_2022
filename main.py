import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from torchsummary import summary

import numpy as np
from numpy.random import shuffle

import util
from models.net import LHGNets

from dataset import MtvDataset
from metric import accuracy
from visualize import *

def init_env(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='latent heterogeneous graph network for incomplete multi-view learning')

    # experiment set
    parser.add_argument('--data', type=str, required=True, help='Dataset feed to model')
    parser.add_argument('--gpu', type=str, default='0', help='GPU index for cuda used')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent representation dim')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat n times experiment')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--missing_rate', type=float, default=0.5, help='Missing rate for view-specfic data')
    parser.add_argument('--split_rate', type=float, default=0.8, help='Train and test data split ratio')
    parser.add_argument('--log_path', type=str, default='./log.txt', help='Log file save path')
    parser.add_argument('--normalize', type=bool, default=True, help='Normalization for dataset')
    parser.add_argument('--dropout_rate', type=float, default=0.6, help='Dropout rate(1-keep probability). ')
    parser.add_argument('--nheads', type=int, default=3, help='Head number for self-attention')
    args = parser.parse_args()
    args.logger = util.get_logger(args.log_path)
    # init environment: gpu, cuda
    init_env(args)
    # args.logger.info(args)
    return args


def train(args, model, data, label, train_idx, feature_mask, optimizer, epoch):
    model.train()
    optimizer.zero_grad()
    # criterion = nn.CrossEntropyLoss()
    rec_vec, output, semantic,= model(feature_mask)
    label = label.long().view(-1, )

    # classification loss
    cls_loss = F.nll_loss(output[train_idx], label[train_idx])
    # args.logger.warning("Classfication loss " + str(cls_loss.item()))

    # reconstruction loss
    rec_loss = 0.0
    for v in range(args.view_num):
        sum = torch.sum(
            torch.pow(torch.sub(rec_vec[v], data[v]), 2.0), 1)
        fea = feature_mask[:, v].double()
        loss = sum * fea
        loss = torch.sum(loss)
        args.logger.warning("View " + str(v) + " loss " + str(loss.item()))
        rec_loss += loss
    loss =10*cls_loss + rec_loss

    # summary loss
    # if epoch < 100:
    #     loss = rec_loss+ rec_loss
    # else:
    #     loss = cls_loss + rec_loss
    # args.logger.warning("Total loss " + str(loss.item()))

    loss.backward()
    optimizer.step()
    acc_train = accuracy(output[train_idx], label[train_idx]).item()
    args.logger.error("Epoch : " + str(epoch) + ' train accuracy : ' + str(acc_train))
    return loss,acc_train

def test(args, model, data, label, test_idx, feature_mask, epoch=0):
    model.eval()
    with torch.no_grad():
        _, output,semantic, = model(feature_mask)
        loss_test = F.nll_loss(output[test_idx], label[test_idx])
        acc_test = accuracy(output[test_idx], label[test_idx]).item()
        args.logger.error("Epoch : " + str(epoch) + ' test accuracy : ' + str(acc_test))
    return acc_test


def main(args):
    # load data
    mtv_data = MtvDataset(args)
    train_data, train_label = mtv_data.get_data('train')
    test_data, test_label = mtv_data.get_data('test')
    view_num = mtv_data.view_number
    view_dim = [train_data[i].shape[1] for i in range(view_num)]
    args.view_num = view_num

    train_sample_number = train_data[0].shape[0]
    test_sample_number = test_data[0].shape[0]
    all_sample_number = train_sample_number + test_sample_number

    # get incomplete mask
    feature_mask = mtv_data.get_missing_mask()

    # transductive learning setting
    train_idx = np.arange(0, train_sample_number)
    test_idx = np.arange(train_sample_number, all_sample_number)
    shuffle(train_idx)
    label = torch.cat([train_label, test_label])
    label = label - 1
    label = label.long().view(-1, )
    cls_num = torch.max(label).item() + 1
    data = {}
    for v in range(view_num):
        data[v] = torch.cat([train_data[v], test_data[v]])

    # build model
    model = LHGNets(view_num, all_sample_number, view_dim, cls_num, args.dropout_rate, args.nheads, args.gpu,
                    latent_dim=args.latent_dim).double()

    # transfer cuda
    if args.gpu != '-1':
        label = label.cuda()
        model = model.cuda()
        feature_mask = feature_mask.cuda()
        for v in range(view_num):
            data[v] = data[v].cuda()

    # args.logger.info("Model Parameters : ")
    for name, param in model.named_parameters():
        if param.requires_grad:
            args.logger.info(str(name) + str(param.data.shape))

    # build optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loss = []
    accuracy = []
    for i in range(args.epochs):
        loss,train_acc= train(args, model, data, label, train_idx, feature_mask, optimizer, i)
        # print(train_acc)
        test_acc = test(args, model, data, label, test_idx, feature_mask, i)
        loss1 = loss.cpu().detach().numpy()
        train_loss.append(loss1)
        test1 = test_acc
        accuracy.append(test1)
    # plt.plot(train_loss, label='loss')
    # plt.plot(test_acc, label='accuracy')
    # plt.xlabel('loss')
    # plt.ylabel('accuracy')
    # plt.legend(loc='best')
    # plt.savefig(".\curve.png")

        # tsne_visualization
        # if i in [1,105,110,150,230]:
        #     fea_train = semantic_train
        #     fea_train = fea_train.cpu().detach().numpy()
        #     la = label
        #     la=la.cpu().detach().numpy()
        #     visualize_data_tsne(fea_train, la, 10, './figures/'+ args.data +'_tsne'+str(i)+'.svg')
    # print(accuracy)
    return accuracy,train_loss

def run():
    # init env
    args = parse_args()
    train_acc = []
    train_loss = []
    acc = []
    loss=[]
    # repeat n times experiment for average
    ret = []
    for i in range(args.repeat):
        accuracy,train_loss = main(args)
        acc=accuracy
        loss = train_loss
        print(acc)
        print(loss)
        ret.append(acc)
    avg = round(np.mean(ret), 3)
    std = round(np.std(ret), 3)
    args.logger.info(args)
    msg = str(args.repeat) + " Times Average result: " + str(avg) + ' / ' + str(std)
    args.logger.info(msg)


if __name__ == '__main__':
    run()
