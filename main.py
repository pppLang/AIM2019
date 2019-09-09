import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data as udata
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter
import time
from dataset import MyDataset
from train import train_epoch, test
from model.SRResNet_Recu import MSRResNet as Model
# from model.SRResNet import MSRResNet as Model
from test_demo import main as test_main
from evalution import get_parameters, get_runtime
from utils.L1_Focal_loss import L1_Focal_Loss

parser = argparse.ArgumentParser(description="BinarySISR")
parser.add_argument("--batchSize_per_gpu", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=90, help="number of epochs")
parser.add_argument("--lr", type=float, default=3e-4, help="initial learning rate")
parser.add_argument("--gpus", type=str, default="5", help='path log files')
opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

scaling_factor = 4
device_ids = [0]
batch_size = opt.batchSize_per_gpu * len(device_ids)
wd = 0
interval = 20
times = 2
input_large = False
selected_num = 800
num_feas = 64 
num_blocks = 3 

outf = 'logs{}_2/SRResnet_Recuv9_{}wd_{}_{}block_{}_{}_{}_{}_{}_{}Int'.format(selected_num, wd, num_feas, num_blocks, opt.lr, \
    opt.epochs, opt.batchSize_per_gpu, len(device_ids), interval, times)

print('this is motherfucker out path {}'.format(outf))
print('using gpu id : {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))


def main():
    print("loading dataset ...")
    trainDataset = MyDataset(mode='train', scaling_factor=scaling_factor, input_large=input_large, selected_num=selected_num)
    print('total train sample num {}'.format(len(trainDataset)))
    trainLoader = udata.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # criterion = nn.L1Loss()
    criterion = L1_Focal_Loss()
    criterion.cuda()

    model = Model(in_nc=3, out_nc=3, nf=num_feas, nb=num_blocks, upscale=4)
    model.cuda()

    parameters = get_parameters(model)
    cost_time = get_runtime(model)
    print('\n\n===========')
    print(parameters, cost_time)
    print('===========\n\n')

    load = False 
    if load:
        print('Reloading the checkpoint.')
        checkpoint = torch.load(os.path.join(outf, 'checkpoint_49.pth'), map_location=torch.device('cuda:0'))
        epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('successfully load checkpoint')
    else:
        epoch = 0
        beta1 = 0.9
        beta2 = 0.999
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=wd, betas=(beta1, beta2))

    writer = SummaryWriter(outf)
    test(model, epoch=epoch, writer=writer)

    lr = opt.lr
    lr_line = [1e-5, 4e-5, 7e-5, 1e-4, 3e-4]
    lr_line = lr_line + [lr]*20+[lr/times]*15+[lr/(times**2)]*15 + \
        [lr/times**3]*15 +[lr/times**4]*10 + [lr/times**5]*5 + [1e-5]*5
    print(len(lr_line), lr_line[-1])

    while epoch < opt.epochs:
        start = time.time()
        #current_lr = opt.lr / (times**int(epoch / interval))
        current_lr = lr_line[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print("epoch {} learning rate {}".format(epoch, current_lr))

        train_epoch(model, optimizer, trainLoader, criterion, epoch, writer=writer)

        if epoch % 1 == 0:
            test(model, epoch=epoch, writer=writer)
        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(outf, 'checkpoint_{}.pth'.format(epoch)))
            test_main(model, outf)

        end = time.time()
        print('epoch {} cost {} hour '.format(
            epoch, str((end - start) / (60 * 60))))
        epoch += 1
        
    torch.save(model.state_dict(), os.path.join(outf, 'model.pth'))
    
    if epoch % 10 != 0:
        test_main(model, outf)


if __name__ == "__main__":
    main()
