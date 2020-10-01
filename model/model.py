from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

def emo_Model(input_shape=(48,48,1)):
    # first input model
    visible = Input(shape=input_shape, name='input')
    num_classes = 7
    #the 1-st block
    conv1_1 = Conv2D(32, kernel_size=3, activation='relu', padding='same', name = 'conv1_1')(visible)
    conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_2')(conv1_1)
    pool1_1 = MaxPooling2D(pool_size=(2,2), name = 'pool1_1')(conv1_2)
    drop1_1 = Dropout(0.25, name = 'drop1_1')(pool1_1)

    #the 2-nd block
    conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_1')(drop1_1)
    pool2_1 = MaxPooling2D(pool_size=(2,2), name = 'pool2_1')(conv2_1)
    conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_2')(pool2_1)
    pool2_2 = MaxPooling2D(pool_size=(2,2), name = 'pool2_2')(conv2_2)
    drop2_1 = Dropout(0.25, name = 'drop2_1')(pool2_2)

    #Flatten and output
    flatten = Flatten(name = 'flatten')(drop2_1)
    dense3_1 = Dense(1024, activation='relu', name='dense3_1')(flatten)
    drop3_1 = Dropout(0.5, name='drop3_1')(dense3_1)
    ouput = Dense(num_classes, activation='softmax', name = 'output')(drop3_1)

    # create model 
    model = Model(inputs =visible, outputs = ouput)
    # summary layers
    print(model.summary())
    
    return model