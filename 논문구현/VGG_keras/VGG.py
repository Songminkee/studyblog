import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
import LRN

class VGG_Net:
    def __init__(self,layer_num=1, numclass=1000):
        super(VGG_Net,self).__init__()

        if layer_num== 1 :
            self.feature = self.VGG_A()
        elif layer_num == 2:
            self.feature = self.VGG_A()
        elif layer_num == 3:
            self.feature = self.VGG_B()
        elif layer_num == 4:
            self.feature = self.VGG_C()
        elif layer_num == 5:
            self.feature = self.VGG_D()
        elif layer_num == 6:
            self.feature = self.VGG_E()

        self.feature_outshape=self.feature.output_shape

        self.classifier = self.VGG_Classifier(self.feature)

    def summary(self):
        return self.classifier.summary()

    def feature_output_shape(self):
        return self.feature_outshape

    def output_shape(self):
        return self.classifier.output_shape


    def VGG_Classifier(self,feature):
        model = feature
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(4096, input_dim=25088, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, input_dim=4096, activation='relu'))
        model.add(Dense(1000, input_dim=4096, activation='softmax'))

        return model



    def VGG_A(self):
        model = keras.models.Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(128, 128, 64), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 128), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 256), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 256), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 512), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))

        return model

    def VGG_ALRN(self):
        model = keras.models.Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu', padding='same'))
        model.add(LRN.LRN())
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(128, 128, 64), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 128), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 256), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 256), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 512), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))


    def VGG_B(self):
        model = keras.models.Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 64), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(128, 128, 64), activation='relu', padding='same'))
        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(128, 128, 128), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 128), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 256), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 256), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 512), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        return model

    def VGG_C(self):
        model = keras.models.Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 64), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(128, 128, 64), activation='relu', padding='same'))
        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(128, 128, 128), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 128), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 256), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(1, 1), input_shape=(64, 64, 256), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 256), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(1, 1), input_shape=(32, 32, 512), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(1, 1), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        return model

    def VGG_D(self):
        model = keras.models.Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 64), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(128, 128, 64), activation='relu', padding='same'))
        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(128, 128, 128), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 128), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 256), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 256), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 256), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 512), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        return model

    def VGG_E(self):
        model = keras.models.Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(224, 224, 64), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(128, 128, 64), activation='relu', padding='same'))
        model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(128, 128, 128), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 128), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 256), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 256), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(64, 64, 256), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 256), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(32, 32, 512), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), input_shape=(16, 16, 512), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2, strides=2))
        return model


