import argparse
import os
import random
import shutil
import time
import warnings
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import util.folder_data_io as folder_data_io
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx as torch_onnx
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch EXperiments for EasyDL mini-testset')
parser.add_argument('--data', type=str,
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--n_class', default=4, type=int,
                    help='n_class')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--min_scale', default=0.08, type=float,
                    help='min_scale')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--decay_epoch', default=30, type=int,
                    help='decay_epoch.')

parser.add_argument('--log_name', default='0', type=str,
                    help='log_name')

#best_acc1 = 0

args = parser.parse_args()

dirnum = 0
path = args.data
if os.path.exists(os.environ['JOB_OUTPUT'] + "labels.txt"):
    os.remove(os.environ['JOB_OUTPUT'] + "labels.txt")
file_write_obj = open(os.environ['JOB_OUTPUT'] + "labels.txt", 'w')
for lists in os.listdir(path):
    sub_path = os.path.join(path, lists)
    print(sub_path)
    if os.path.isdir(sub_path):
        dirnum = dirnum+1
        file_write_obj.writelines(lists)
        file_write_obj.write('\n')
args.n_class = dirnum
file_write_obj.close()

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler(os.environ['JOB_OUTPUT'] + "log_%s.txt"%( args.log_name ), mode = 'w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)

logger.info("Start print log -> :")

dict_model_acc = {}
dict_test_acc = {}
dict_model_time = {}
dict_model_top = {}



def make_test_dataset(dir):
    images = []
    dir = os.path.abspath(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        len_file = len(os.listdir(d))
        list_test = random_filename(os.listdir(d),(int(len_file*0.05)+1))
        images.extend(list_test)
    return images

def random_filename(filename,length):
    random.seed(args.seed)
    result = random.sample(filename,length)
    return result

def main(arch_name,arch_model,arch_lr,testlist_temp):
    t0 = time.time()
    global dict_model_acc,dict_model_time,dict_test_acc,dict_model_top

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    best_acc1 = 0

    # create model
    model = models.__dict__[arch_name](num_classes = args.n_class)
    if arch_model:
        logger.info("=> using pre-trained model '{}'".format(arch_name))
        pretrained_dict = torch.load(arch_model)
        for i in range(2):
            pretrained_dict.popitem()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        

    torch.cuda.set_device(args.gpu)
  
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), arch_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    model.cuda(args.gpu)
    # Data loading code
    traindir = args.data
    testdir = args.data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale = (args.min_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness = 0.5,contrast = 0.5,hue = 0.5),
                #transforms.RandomGrayscale(p =0.5),
                transforms.ToTensor(),
				normalize,
            ])
    test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    train_dataset, val_dataset = folder_data_io.get_trainval_dataset(
        traindir,testlist_temp, train_transform, None, test_transform, None
        )


    # batch_size tricks cyliu7
    train_batch_size = args.batch_size
    while (1 + len(train_dataset) / train_batch_size) < 5:
        train_batch_size = train_batch_size / 2
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(train_batch_size), shuffle = True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle = False,
        num_workers=min(args.workers / 4, 4), pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        folder_data_io.DatasetFolder(testdir,testlist_temp, test_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.workers / 4, 4), pin_memory=True)

    # if args.evaluate:
    #     validate(test_loader, model, criterion, args)
    #     return

    logger.info('len(train_dataset) = %d, train_batch_size = %d, len(train_loader) = %d.'%(len(train_dataset), train_batch_size, len(train_loader)))
    choosed_test_acc = 0
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args,arch_lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1,train_top_dict= validate(val_loader, model, criterion, args, 'VAL')

        # remember best acc@1 and save checkpoint
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)

        if epoch == 0:
            dict_model_acc[arch_name] = float('{:.3f}'.format(acc1))
        else:
            if float('{:.3f}'.format(acc1)) > dict_model_acc[arch_name]:
                dict_model_acc[arch_name] = float('{:.3f}'.format(acc1))

        test_acc,test_top_dict = validate(test_loader, model, criterion, args, 'TEST')

        if epoch == 0:
            dict_test_acc[arch_name] = float('{:.3f}'.format(test_acc))
        else:
            if float('{:.3f}'.format(test_acc)) > dict_test_acc[arch_name]:
                dict_test_acc[arch_name] = float('{:.3f}'.format(test_acc))

        if is_best:
            choosed_test_acc = test_acc
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'test_acc': test_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best,train_top_dict, model_name = arch_name)

        logger.info('Choosed Test ACC = %.3f' % (1.0 * choosed_test_acc))
    t1 = time.time() - t0
    dict_model_time[arch_name] = t1




def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = ()
    if args.n_class < 5:
        for i in range(1, args.n_class + 1):
            topk = topk + (i,)
    else:
        topk = (1, 2, 3, 4, 5)

    # switch to train mode
    model.train()

    end = time.time()
    for i, _data in enumerate(train_loader):

        input, target = _data[0], _data[1]
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
   
        #logger.info('-----', target.size(0), target)
        # compute output

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, args, strr = 'Test'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()
    top4 = AverageMeter()
    top5 = AverageMeter()
    topk = ()
    if args.n_class < 5:
        for i in range(1, args.n_class + 1):
            topk = topk + (i,)
    else:
        topk = (1, 2, 3, 4, 5)

    # switch to evaluate model
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, _data in enumerate(val_loader):

            input, target = _data[0], _data[1]

            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            #acc1= accuracy(output, target, topk=(1,))[0]
            #losses.update(loss.item(), input.size(0))
            #top1.update(acc1[0], input.size(0))


            acc1, acc2, acc3, acc4, acc5 = accuracy(output, target, topk)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top2.update(acc2[0], input.size(0))
            top3.update(acc3[0], input.size(0))
            top4.update(acc4[0], input.size(0))
            top5.update(acc5[0], input.size(0))
           

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('{0}: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       strr, i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))

        logger.info('{0} * Acc1@1 {top1.avg:.3f}\t'
               'Acc2@1 {top2.avg:.3f}\t'
               'Acc3@1 {top3.avg:.3f}\t'
               'Acc4@1 {top4.avg:.3f}\t'
               'Acc5@1 {top5.avg:.3f}\t'
              .format(strr, top1=top1, top2=top2, top3=top3, top4=top4, top5=top5))

        top1_to_5_dict = {'top1':top1.avg,'top2':top2.avg,'top3':top3.avg,'top4':top4.avg,'top5':top5.avg}

    return top1.avg,top1_to_5_dict


def save_checkpoint(state, is_best,dict_temp,model_name):
    global dict_model_top
    filename = os.environ['JOB_OUTPUT'] + model_name + '_checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.environ['JOB_OUTPUT'] + model_name + '_model_best.pth.tar')
        dict_model_top[model_name] = dict_temp



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


def adjust_learning_rate(optimizer, epoch, args,arch_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = arch_lr * (0.1 ** (epoch // args.decay_epoch))
    logger.info('Current LR = %.6f.'%(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        for top in range(maxk+1,6):
            res_top = res[-1]
            res.append(res_top)
        return res


if __name__ == '__main__':
    t2 = time.time()
    dict_model = {'alexnet':'alexnet-owt-4df8aa71.pth','resnet18':'resnet18-5c106cde.pth','resnet34':'resnet34-333f7ec4.pth','squeezenet1_0':'squeezenet1_0-a815701f.pth','vgg16':'vgg16-397923af.pth'}
    # dict_model = {'alexnet':'alexnet-owt-4df8aa71.pth','resnet18':'resnet18-5c106cde.pth','resnet34':'resnet34-333f7ec4.pth','squeezenet1_0':'squeezenet1_0-a815701f.pth','vgg16':'vgg16-397923af.pth','shufflenetv2_x1_0':'shufflenetv2_x1-5666bf0f80.pth'}
    #dict_model = {'resnet34':'resnet34-333f7ec4.pth','vgg16':'vgg16-397923af.pth','shufflenetv2_x1_0':'shufflenetv2_x1-5666bf0f80.pth'}
    
    traindir_temp = args.data
    test_img_all = make_test_dataset(traindir_temp)
    for key in dict_model.keys():
        if key == 'shufflenetv2_x1_0':
            main(key,dict_model[key],0.1,test_img_all)
        else:
            main(key,dict_model[key],0.004,test_img_all)
    max_model_val = max(dict_model_acc,key = dict_model_acc.get)
    max_model_test = max(dict_test_acc,key = dict_test_acc.get)
    t3 = time.time() - t2
    
    input_shape = (3, 224, 224)
    dummy_input = Variable(torch.randn(1, *input_shape))
    
	
    time.sleep(2)
    print('-----------------------val_acc------------------------')
    print(dict_model_acc)
    print(max_model_val)
    print(dict_model_acc[max_model_val])
    model_onnx_path = os.environ['JOB_OUTPUT'] + max_model_val + "_torch_Train_model.onnx"
    model = models.__dict__[max_model_val](num_classes = args.n_class)
    model.load_state_dict(torch.load(os.environ['JOB_OUTPUT'] + max_model_val + '_model_best.pth.tar')['state_dict'])
    output = torch_onnx.export(model, dummy_input, model_onnx_path, verbose=False)
    print("Export of torch_model.onnx complete!")
    time.sleep(2)
    print('-----------------------test_acc------------------------')
    print(dict_test_acc)
    print(max_model_test)
    print(dict_test_acc[max_model_test])
    model_onnx_path = os.environ['JOB_OUTPUT'] + max_model_test + "_torch_Test_model.onnx"
    model = models.__dict__[max_model_test](num_classes = args.n_class)
    model.load_state_dict(torch.load(os.environ['JOB_OUTPUT'] + max_model_test + '_model_best.pth.tar')['state_dict'])
    output = torch_onnx.export(model, dummy_input, model_onnx_path, verbose=False)
    print("Export of torch_model.onnx complete!")

    f = open(os.environ['JOB_OUTPUT'] + '/top.txt', 'w')
    for i in range(1,6):
        f.write('top%d:%.3f'%(i,dict_model_top[max_model_val]['top'+str(i)]))
        f.write('\n')
    f.close()

