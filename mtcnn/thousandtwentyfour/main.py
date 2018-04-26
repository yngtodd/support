from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MTCNN
from data import Deidentified
from data import load_wv_matrix
from utils import load_results
from objective import objective

import time
import logging
from logging.config import fileConfig
from logger import print_progress
from logger import print_accuracy
from parser import parse_args

import numpy as np

from torch.cuda.profiler import start, stop


experiment = Experiment(api_key="1gZw4BPQhKSQ63qn9buShJCcs", project_name="support")


def print_shapes(predicted_name, predicted, actual_name, actual):
    print('{} is {} and has shape {}'.format(predicted_name, type(predicted), predicted.size()))
    print('{} is {} and has shape {}'.format(actual_name, type(predicted), actual.size()))


def train(epoch, train_loader, optimizer, criterion, train_size, args):
    """
    Train the model.

    Parameters:
    ----------
    * `epoch`: [int]
        Epoch number.

    * `train_loader` [torch.utils.data.Dataloader]
        Data loader to load the test set.

    * `model`: [Pytorch model class]
        Instantiated model.

    * `optimizer`: [torch.optim optimizer]
        Optimizer for learning the model.

    * `criterion`: [torch loss function]
        Loss function to measure learning.

    * `train_size`: [int]
        Size of the training set (for logging).

    * `args`: [argparse object]
        Parsed arguments.
    """
    model.train()
#    wv_matrix = torch.randn(2882, 300) 
    for batch_idx, sample in enumerate(train_loader):
        sentence = sample['sentence']
        subsite = sample['subsite']
        laterality = sample['laterality']
        behavior = sample['behavior']
        grade = sample['grade']

        if args.cuda:
#            wv_matrix = wv_matrix.cuda()
            sentence = sentence.cuda()
            subsite = subsite.cuda()
            laterality = laterality.cuda()
            behavior = behavior.cuda()
            grade = grade.cuda()
            if args.half_precision:
#                wv_matrix = wv_matrix.half()
                sentence = sentence.half()
                subsite = subsite.half()
                laterality = laterality.half()
                behavior = behavior.half()
                grade = grade.half()

#        print('grade is of type {} with shape {}'.format(type(grade), grade.size()))
#        wv_matrix = Variable(wv_matrix)
        sentence = Variable(sentence)
        subsite = Variable(subsite)
        laterality = Variable(laterality)
        behavior = Variable(behavior)
        grade = Variable(grade)

        optimizer.zero_grad()
        out_subsite, out_laterality, out_behavior, out_grade = model(sentence)
        loss_subsite = criterion(out_subsite, subsite)
        loss_laterality = criterion(out_laterality, laterality)
        loss_behavior = criterion(out_behavior, behavior)
        loss_grade = criterion(out_grade, grade)
        loss = loss_subsite + loss_laterality + loss_behavior + loss_grade
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print_progress(epoch, batch_idx, args.batch_size, train_size, loss.data[0], logger=logger)

        return loss.data[0]

def test(epoch, test_loader, args):
    """
    Test the model.
 
    Parameters:
    ----------
    * `epoch`: [int]
        Epoch number.

    * `test_loader`: [torch.utils.data.Dataloader]
        Data loader for the test set.

    * `args`: [argparse object]
        Parsed arguments.
    """
    model.eval()
    subsite_correct = 0
    laterality_correct = 0
    behavior_correct = 0
    grade_correct = 0
    total = 0
    
    subsite_loss = []
    laterality_loss = []
    behavior_loss = []
    grade_loss = []
    
    for _, sample in enumerate(test_loader):
        sentence = sample['sentence']
        subsite = sample['subsite']
        laterality = sample['laterality']
        behavior = sample['behavior']
        grade = sample['grade']

        if args.cuda:
            sentence = sentence.cuda()
            subsite = subsite.cuda()
            laterality = laterality.cuda()
            behavior = behavior.cuda()
            grade = grade.cuda()

        sentence = Variable(sentence)
        subsite = Variable(subsite)
        laterality = Variable(laterality)
        behavior = Variable(behavior)
        grade = Variable(grade)

        out_subsite, out_laterality, out_behavior, out_grade = model(sentence)
        _, subsite_predicted = torch.max(out_subsite.data, 1)
        _, laterality_predicted = torch.max(out_laterality.data, 1)
        _, behavior_predicted = torch.max(out_behavior.data, 1)
        _, grade_predicted = torch.max(out_grade.data, 1)

        #print_shapes('subsite predicted', subsite_predicted, 'actual', subsite)

        total += subsite.size(0)
        subsite_correct += (subsite_predicted == subsite.data).sum()
        laterality_correct += (laterality_predicted == laterality.data).sum()
        behavior_correct += (behavior_predicted == behavior.data).sum()
        grade_correct += (grade_predicted == grade.data).cpu().sum()
    
    subsite_acc = 100 * subsite_correct / total
    laterality_acc = 100 * laterality_correct / total
    behavior_acc = 100 * behavior_correct / total
    grade_acc = 100 * grade_correct / total
    
    metrics = {'subsite_accuracy': subsite_acc,
               'laterality_accuracy': laterality_acc,
               'behavior_accuracy': behavior_acc,
               'grade_accuracy': grade_acc}

    experiment.log_multiple_metrics(metrics)

    print_accuracy(
        epoch, subsite_correct, laterality_correct,
        behavior_correct, grade_correct, total, logger=logger
    )
   
    return subsite_acc, laterality_acc, behavior_acc, grade_acc


def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    global fileConfig
    fileConfig('logging_config.ini')

    global logger
    logger = logging.getLogger()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_data = Deidentified(
        data_path=args.data_dir + '/data/train',
        label_path=args.data_dir + '/labels/train'
    )

    train_size = len(train_data)

    test_data = Deidentified(
        data_path=args.data_dir + '/data/test',
        label_path=args.data_dir + '/labels/test'
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    wv_matrix = load_wv_matrix(args.data_dir + '/wv_matrix/wv50.npy')

    results = load_results(args.stored_results, sort=True)
    space = results[args.num_result]
    domain = space.x
    kernel1 = int(domain[0])
    kernel2 = int(domain[1])
    kernel3 = int(domain[2])
    num_filters1 = int(domain[3])
    num_filters2 = int(domain[4])
    num_filters3 = int(domain[5])
    dropout1 = float(domain[6])
    dropout2 = float(domain[7])
    dropout3 = float(domain[8])

    parameters = {'kernel1': kernel1,
                  'kernel2': kernel2,
                  'kernel3': kernel3,
                  'num_filters1': num_filters1,
                  'num_filters2': num_filters2,
                  'num_filters3': num_filters3,
                  'dropout1': dropout1,
                  'dropout2': dropout2,
                  'dropout3': dropout3}

    experiment.log_multiple_params(parameters)
    
    global model
    model = MTCNN(wv_matrix, kernel1=kernel1, kernel2=kernel2, kernel3=kernel3,
                  num_filters1=num_filters1, num_filters2=num_filters2, num_filters3=num_filters3,
                  dropout1=dropout1, dropout2=dropout2, dropout3=dropout3)

    if args.cuda and torch.cuda.device_count() > 1:
#        model = nn.DataParallel(model)
        model = model.cuda()
        if args.fp16:
            model = network_to_half(model)

    global param_copy
    if args.fp16:
        param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model.parameters()]
        for param in param_copy:
            param.requires_grad = True
    else:
        param_copy = list(model.parameters())

    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    loss_arry = []    
    subsite_arry = []
    laterality_arry = []
    behavior_arry = []
    grade_arry = []
    epoch_time = []
    model.count_parameters()
    start_time = time.time()
    start()
    for epoch in range(1, args.num_epochs + 1):
            epoch_start = time.time()
            loss = train(epoch, train_loader, optimizer, criterion, train_size, args)
            loss_arry.append(loss)
            subsite_acc, laterality_acc, behavior_acc, grade_acc = test(epoch, test_loader, args)
            epoch_time.append(time.time() - epoch_start)
            subsite_arry.append(subsite_acc)
            laterality_arry.append(laterality_acc)
            behavior_arry.append(behavior_acc)
            grade_arry.append(grade_acc)
    stop()
   
    logger.info('Total runtime: {:.4f} seconds'.format(time.time() - start_time)) 
    print('Total runtime: {:.4f} seconds'.format(time.time() - start_time)) 

#    loss_arry = np.asarray(loss_arry)
#    subsite_arry = np.asarray(subsite_arry)
#    laterality_arry = np.asarray(laterality_arry)
#    behavior_arry = np.asarray(behavior_arry)
#    grade_arry = np.asarray(grade_arry)
#    epoch_time = np.asarray(epoch_time)

#    np.save('loss_time', loss_arry)
#    np.save('subsite_acc', subsite_arry)
#    np.save('laterality_acc', laterality_arry)
#    np.save('behavior_acc', behavior_arry)
#    np.save('grade_acc', grade_arry)
#    np.save('epoch_times', epoch_time)


if __name__=='__main__':
    main()
