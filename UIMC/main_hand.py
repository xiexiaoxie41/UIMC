import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import UGC
from util import mv_dataset, read_mymat, build_ad_dataset, process_data, get_validation_set, mv_tabular_collate, get_validation_set_Sn, partial_mv_dataset, partial_mv_tabular_collate
import warnings
from EarlyStopping_hand import EarlyStopping
from collections import Counter
from select_k_neighbors import get_samples

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--pretrain_epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing-epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--patience', type=int, default=30, metavar='LR',
                        help='parameter of Earlystopping [default: 30]')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training [default: False]')
    parser.add_argument('--missing-rate', type=float, default=0, metavar='LR',
                        help='missingrate [default: 0]')
    parser.add_argument('--n-sample', type=int, default=30, metavar='LR',
                        help='times of sampling [default: 10]')
    parser.add_argument('--k', type=int, default=10, metavar='LR',
                        help='number of neighbors [default: 0]')
    parser.add_argument('--k_test', type=int, default=10, metavar='LR',
                        help='number of neighbors [default: 0]')
    parser.add_argument('--if-mean', type=int, default=0, metavar='LR',
                        help='if mean [default: True]')
    parser.add_argument('--latent-dim', type=int, default=64, metavar='LR',
                        help='latent layer dimension [default: True]')
    args = parser.parse_args()

    args.decoder_dims = [[240], [76], [216], [47], [64], [6]]
    args.encoder_dims = [[240], [76], [216], [47], [64], [6]]
    args.classifier_dims = [[240], [76], [216], [47], [64], [6]]
    view_num=6

    dataset_name = 'handwritten0.mat'
    missing_rate = args.missing_rate
    X, Y, Sn = read_mymat('./data/', dataset_name, ['X', 'Y'], missing_rate)
    partition = build_ad_dataset(Y, p=0.8, seed=999)

    X = process_data(X, view_num)
    X_train, Y_train, X_test, Y_test, Sn_train = get_samples(x=X, y=Y, sn=Sn,
                                                   train_index=partition['train'],
                                                   test_index=partition['test'],
                                                   n_sample=args.n_sample,
                                                   k=args.k)

    train_loader = DataLoader(dataset=partial_mv_dataset(X_train, Sn_train, Y_train), batch_size=args.batch_size,
                              shuffle=True, num_workers=1,
                              collate_fn=partial_mv_tabular_collate)
    test_batch_size = args.n_sample
    test_loader = DataLoader(dataset=mv_dataset(X_test, Y_test), batch_size=test_batch_size,
                             shuffle=False, num_workers=1,
                             collate_fn=mv_tabular_collate)



    model = UGC(10, view_num, args.classifier_dims, args.annealing_epochs)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.cuda:
        model.cuda()


    def pretrain(epoch):
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (data, sn, target) in enumerate(train_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].float().cuda())
            target = Variable(target.long().cuda())
            sn = Variable(sn.long().cuda())
            # refresh the optimizer
            optimizer.zero_grad()
            loss = model.classify(data, target, epoch, sn)
            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
        print('Pretrain Epoch: {} \tLoss: {:.6f}'.format(epoch, loss_meter.avg))


    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (data, sn, target) in enumerate(train_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].float().cuda())
            target = Variable(target.long().cuda())
            sn = Variable(sn.long().cuda())
            # refresh the optimizer
            optimizer.zero_grad()
            evidences, evidence_a, loss = model(data, target, epoch, batch_idx, sn)
            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data[0]), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss_meter.avg))
        return loss_meter.avg


    def test(epoch, dataloader):
        model.eval()
        data_num, correct_num = 0, 0
        for batch_idx, (data, target) in enumerate(dataloader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].float().cuda())
            target = Variable(target.long().cuda())
            data_num += target.size(0)
            with torch.no_grad():
                evidence, evidences = model(data, target, epoch, batch_idx, data, 1)
                _, predicted = torch.max(evidences.data, 1)
                list_ = predicted.detach().cpu().numpy().tolist()
                most_ = Counter(list_).most_common(1)[0][0]
                if most_ == target[0]:
                    correct_num += 1
        data_num = int(data_num/args.n_sample)
        acc = correct_num / data_num
        print("total_num：",data_num)
        print('====> accuracy：', acc)
        return acc

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        early_stopping(loss*(-1), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('checkpoint_hand.pt'), False)
    acc = test(epoch, test_loader)



    with open("./test-hand.txt", "a") as f:
        text = "\tmissing_rate:" + str(missing_rate) + "\taccuracy:" + str(acc) +"\n"
        f.write(text)
    f.close()