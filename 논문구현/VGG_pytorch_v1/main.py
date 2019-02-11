import torch
import torch.nn as nn
import VGG
from torch.autograd import Variable


if __name__=="__main__":
    print('Select VGG')
    layer_num = int(input('1.VGG-A\n2.VGG-A_LRN\n3.VGG-B\n4.VGG-C\n5.VGG-D\n6.VGG-E'))
    if layer_num < 1:
        layer_num=1
    elif layer_num >6 :
        layer_num=6

    model = VGG.VGG_Net(layer_num=layer_num)


    x = Variable(torch.randn(1,3,224,224).float())
    '''
    input: input tensor (minibatch x in_channels x iH x iW)
    '''
    y = model(x)

    print(model)
    print('output shape: ', y.shape)

