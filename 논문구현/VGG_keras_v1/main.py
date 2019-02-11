import VGG

if __name__=="__main__":
    print('Select VGG')
    layer_num = int(input('1.VGG-A\n2.VGG-A_LRN\n3.VGG-B\n4.VGG-C\n5.VGG-D\n6.VGG-E'))
    if layer_num < 1:
        layer_num=1
    elif layer_num >6 :
        layer_num=6

    model = VGG.VGG_Net(layer_num=layer_num)
    print(model.feature_output_shape())
    print(model.summary())
    print(model.output_shape())



