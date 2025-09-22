#
import os
from keras import Input, Model
from keras.layers import  Bidirectional, Concatenate, GRU, Dense, Flatten,Conv2D,Reshape
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.optimizers import Adam

def Crispr_SGRU():
    #inputs = Input(shape=(24, 7))
    inputs_1 = Input(shape=(1, 24, 7), name='main_input')
    conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(inputs_1)
    conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(inputs_1)
    conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(inputs_1)
    conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(inputs_1)
    conv_output = tf.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])
    conv_output = Reshape((24, 40))(conv_output)
    x0 = Bidirectional(GRU(30, return_sequences=True))(conv_output)
    inputs_2=Reshape((24,7))(inputs_1)
    x = Concatenate(axis=2)([inputs_2, x0])

    x1 = Bidirectional(GRU(20, return_sequences=True))(x)
    x = Concatenate(axis=2)([x0, x1])
    
    x2 = Bidirectional(GRU(10, return_sequences=True))(x)
    x = Concatenate(axis=2)([x1, x2])

    x = Concatenate(axis=-1)([x0, x1, x2])
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)#其实好处就是自动获取维度，你看这个GRU的维度就给了一个
    x = Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.35)(x)
    x = Dense(2, activation="sigmoid")(x) 

    model = Model(inputs=inputs_1, outputs=x)
    print(model.summary())
    opt = Adam(learning_rate=0.0001)
    # opt = SGD(learning_rate=0.0001)
    # model.compile(loss=lg.dice_loss(), optimizer=opt, metrics=['accuracy'])
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
    # model.compile(loss=lg.asymmetric_focal_loss(), optimizer=opt, metrics=['accuracy'])
    return model
