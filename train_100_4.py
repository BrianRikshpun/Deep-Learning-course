from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
import os
#import cv2
import time
# from model.residual_attention_network_pre import ResidualAttentionModel
# based https://github.com/liudaizong/Residual-Attention-Network
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel

model_file = r'C:\Users\Brian\PycharmProjects\ResidualAttentionNetwork-pytorch\Residual-Attention-Network\model_92_sgd.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale',
'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
'bottles', 'bowls', 'cans', 'cups', 'plates',
'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
'bed', 'chair', 'couch', 'table', 'wardrobe',
'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
'bear', 'leopard', 'lion', 'tiger', 'wolf',
'bridge', 'castle', 'house', 'road', 'skyscraper',
'cloud', 'forest', 'mountain', 'plain', 'sea',
'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
'crab', 'lobster', 'snail', 'spider', 'worm',
'baby', 'boy', 'girl', 'man', 'woman',
'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
'maple', 'oak', 'palm', 'pine', 'willow',
'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
'lawn-mower', 'rocket', 'streetcar', 'tank','tractor')



# for test
def test(model, test_loader, btrain=False, model_file=model_file):
    # Test
    if not btrain:
        model.load_state_dict(torch.load(model_file))
    model.eval()

    correct = 0
    total = 0
    #
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))

    for images, labels in test_loader:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        #
        c = (predicted == labels.data).squeeze()
        for i in range(20):
            label = labels.data[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
    print('Accuracy of the model on the test images:', float(correct) / total)
    for i in range(100):
        if(class_total[i] == 0):
            print('Accuracy of %5s : None' % (
                classes[i]))
        else :
            print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    return correct / total


def main():
    # Image Preprocessing
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((32, 32), padding=4),  # left, top, right, bottom
        # transforms.Scale(224),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # when image is rgb, totensor do the division 255
    # CIFAR-10 Dataset


    train_dataset = datasets.CIFAR100(root='./data/',
                                     train=True,
                                     transform=transform,
                                     download=True)

    test_dataset = datasets.CIFAR100(root='./data/',
                                    train=False,
                                    transform=test_transform)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=5,  # 64
                                               shuffle=True, num_workers=6)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=20,
                                              shuffle=False)

    model = ResidualAttentionModel().cuda()
    print(model)

    lr = 0.1  # 0.1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    is_train = True
    is_pretrain = False
    acc_best = 0
    total_epoch = 300
    if is_train is True:
        if is_pretrain == True:
            model.load_state_dict((torch.load(model_file)))
        # Training
        for epoch in range(total_epoch):
            model.train()
            tims = time.time()
            for i, (images, labels) in enumerate(train_loader):
                images = Variable(images.cuda())
                # print(images.data)
                labels = Variable(labels.cuda())

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print("hello")
                if (i + 1) % 100 == 0:
                    print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (
                    epoch + 1, total_epoch, i + 1, len(train_loader), loss.item()))
            print('the epoch takes time:', time.time() - tims)
            print('evaluate test set:')
            acc = test(model, test_loader, btrain=True)
            if acc > acc_best:
                acc_best = acc
                print('current best acc,', acc_best)
                torch.save(model.state_dict(), model_file)
            # Decaying Learning Rate
            if (epoch + 1) / float(total_epoch) == 0.3 or (epoch + 1) / float(total_epoch) == 0.6 or (
                    epoch + 1) / float(total_epoch) == 0.9:
                lr /= 10
                print('reset learning rate to:', lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    print(param_group['lr'])
                # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                # optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
        # Save the Model
        torch.save(model.state_dict(), 'last_model_92_sgd_cfar100.pkl')

    else:
        test(model, test_loader, btrain=False)


if __name__ == '__main__':
    main()
