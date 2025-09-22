


from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras,argmax
from keras.layers import Reshape, Conv1D, Input, Dense, Flatten, Concatenate, MaxPooling1D, Reshape, Embedding, GRU

## No random state is initialized, np.random.seed(42) is used in the model runner file if needed
def get_cnn(sequence_length, bp_presenation, only_seq_info, if_bp,if_seperate_epi, num_of_additional_features, epigenetic_window_size, epigenetic_number, task = None):
        return create_convolution_model(sequence_length, bp_presenation, only_seq_info, if_bp,if_seperate_epi, num_of_additional_features, epigenetic_window_size, epigenetic_number, task)   
def get_xgboost_cw(scale_pos_weight, random_state, if_data_reproducibility):
    sub_sample = 1
    if if_data_reproducibility:
        sub_sample = 0.5

    return XGBClassifier(random_state=random_state,subsample=sub_sample,scale_pos_weight=scale_pos_weight, objective='binary:logistic',n_jobs=-1) 
def get_xgboost(random_state):
        return XGBClassifier(random_state=random_state, objective='binary:logistic',n_jobs=-1)
def get_logreg(random_state,if_data_reproducibility):
    if if_data_reproducibility:
        return LogisticRegression(random_state=random_state,solver='sag',n_jobs=-1)
    return LogisticRegression(random_state=random_state,n_jobs=-1)
def create_conv_seq_layers(seq_input,sequence_length,bp_presenation):
    seq_input_reshaped = Reshape((sequence_length, bp_presenation)) (seq_input)

    seq_conv_1 = Conv1D(32, 3, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_input_reshaped)
    seq_acti_1 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_1)
    seq_drop_1 = keras.layers.Dropout(0.1)(seq_acti_1)
    
    seq_conv_2 = Conv1D(64, 3, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_drop_1)
    seq_acti_2 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_2)
    seq_max_pooling_1 = MaxPooling1D(pool_size=3, padding="same")(seq_acti_2)

    seq_conv_3 = Conv1D(128, 3, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_max_pooling_1)
    seq_acti_3 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_3)

    seq_conv_4 = Conv1D(256, 2, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_acti_3)
    seq_acti_4 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_4)
    seq_max_pooling_2 = MaxPooling1D(pool_size=3, padding="same")(seq_acti_4)

    seq_conv_5 = Conv1D(512, 2, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_max_pooling_2)
    seq_acti_5 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_5)

    seq_flatten = Flatten()(seq_acti_5) 
    return seq_flatten
def create_conv_epi_layer(epi_input,kernal_size,strides,epigenetic_window_size,epigenetic_number):
    epi_input_reshaped = Reshape((epigenetic_window_size,epigenetic_number))(epi_input)
    epi_conv_6 = Conv1D(32,kernel_size=kernal_size,kernel_initializer='random_uniform',strides=strides,padding='valid')(epi_input_reshaped)
    epi_acti_6 = keras.layers.LeakyReLU(alpha=0.2)(epi_conv_6)
    epi_max_pool_3 = MaxPooling1D(pool_size=2, padding='same')(epi_acti_6) 
    epi_conv_7 = Conv1D(64,kernel_size=3,kernel_initializer='random_uniform',strides=strides,padding='valid')(epi_max_pool_3)
    epi_acti_7 = keras.layers.LeakyReLU(alpha=0.2)(epi_conv_7)
    epi_seq_flatten = Flatten()(epi_acti_7)
    return epi_seq_flatten

def create_convolution_model(sequence_length, bp_presenation,only_seq_info,if_bp,if_seperate_epi,num_of_additional_features,epigenetic_window_size,epigenetic_number, task=None):
    # set seq conv layers
    seq_input = Input(shape=(sequence_length * bp_presenation))
    seq_flatten = create_conv_seq_layers(seq_input=seq_input,sequence_length=sequence_length,bp_presenation=bp_presenation)

    if (only_seq_info or if_bp): # only seq information given
        combined = seq_flatten
    elif if_seperate_epi: # epigenetics in diffrenet conv
        epi_feature = Input(shape=(epigenetic_window_size * epigenetic_number))
        epi_seq_flatten = create_conv_epi_layer(epi_input=epi_feature,kernal_size=(int(epigenetic_window_size/10)),strides=5,epigenetic_window_size=epigenetic_window_size,epigenetic_number=epigenetic_number)
        combined = Concatenate()([seq_flatten, epi_seq_flatten])
        
    else:
        feature_input = Input(shape=(num_of_additional_features))
        combined = Concatenate()([seq_flatten, feature_input])

    seq_dense_1 = Dense(256, activation='relu')(combined)
    seq_drop_2 = keras.layers.Dropout(0.3)(seq_dense_1)
    seq_dense_2 = Dense(128, activation='relu')(seq_drop_2)
    seq_drop_3 = keras.layers.Dropout(0.2)(seq_dense_2)
    seq_dense_3 = Dense(64, activation='relu')(seq_drop_3)
    seq_drop_4 = keras.layers.Dropout(0.2)(seq_dense_3)
    seq_dense_4 = Dense(40, activation='relu')(seq_drop_4)
    seq_drop_5 = keras.layers.Dropout(0.2)(seq_dense_4)
    
    # Set loss and last neuron for the task
    loss,metrics,output_activation = task_model_parameters(task)
    output = Dense(1, activation=output_activation)(seq_drop_5)
    
    ## Set inputs and outputs sizes for the model to accepet.
    if (only_seq_info or if_bp):
        model = keras.Model(inputs=seq_input, outputs=output)
    elif if_seperate_epi:
        model = keras.Model(inputs=[seq_input,epi_feature], outputs=output)

    else:
        model = keras.Model(inputs=[seq_input, feature_input], outputs=output)
    
    
    model.compile(loss=loss, optimizer= keras.optimizers.Adam(learning_rate=1e-3), metrics=metrics)
    print(model.summary())
    return model



def get_gru_emd(task, input_shape=(24, 25), embed_dim=44,
            dense_layers=(128, 64), activation_funs=("relu", "relu"), optimizer="adam", num_of_additional_features=0,if_flatten = False):
    
        """
        Initializes a C_2 model instance.

        Args:
            model_task (Model_task): Specifies whether the task is classification or regression.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            input_shape (tuple or list, optional): Shape of the primary input. If a list,
                it is treated as [primary_shape, additional_input_shape]. Defaults to (24, 25).
            embed_dim (int, optional): Dimensionality of the embedding layer. Defaults to 44.
            embed_dropout (float, optional): Dropout rate for the embedding layer. Defaults to 0.2.
            dense_layers (tuple, optional): Sizes of the hidden dense layers. Defaults to (128, 64).
            activation_funs (tuple, optional): Activation functions for the dense layers. Defaults to ("relu", "relu").
            optimizer (str, optional): Name of the optimizer to use. Defaults to "adam".
        """

        sequence_length = input_shape[0]    
        vocab_size = input_shape[1]
        if if_flatten:
            inputs = Input(shape=(sequence_length * vocab_size,))  # Input shape: (600,)
            reshaped_inputs = Reshape((sequence_length, vocab_size))(inputs)  # Reshape to (24, 25)
        else : reshaped_inputs = Input(shape=(sequence_length, vocab_size))  # Input shape: (24, 25)
        
        # Reduce one-hot rows to integers
        reduced_input = keras.layers.Lambda(argmax_layer)(reshaped_inputs)
        
        print("Reduced input shape:", reduced_input.shape)  # (None, 24)

        # Embedding layer
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=sequence_length)
        embedded_output = embedding_layer(reduced_input)
        print("After embedding:", embedded_output.shape)  # (None, 24, 44)

        # GRU layer
        gru = GRU(64, return_sequences=True)
        gru_output = gru(embedded_output)
        print("After GRU:", gru_output.shape)  # (None, 24, 64)
        x = keras.layers.Flatten()(gru_output)
        if num_of_additional_features > 0 :
            feature_input = Input(shape=(num_of_additional_features,))
            x = Concatenate()([x, feature_input])
        for dense_size,activation_func in zip(dense_layers,activation_funs):
            x = keras.layers.Dense(dense_size,activation=activation_func)(x)
        loss,metrics,output_activation = task_model_parameters(task)
        output = Dense(1, activation=output_activation)(x)
        
        
        if num_of_additional_features > 0 :
            model = keras.Model(inputs=[inputs, feature_input], outputs=output)
        else:
            model = keras.Model(inputs=inputs, outputs=output)
        
        
        model.compile(loss=loss, optimizer= keras.optimizers.Adam(learning_rate=1e-3), metrics=metrics)
        print(model.summary())
        return model


def argmax_layer(x):
    return argmax(x, axis=-1)


def task_model_parameters(task):
    if task.lower() == "classification":
        return keras.losses.BinaryCrossentropy(), ['binary_accuracy'], 'sigmoid'
    elif task.lower() == "regression":
        return keras.losses.MeanSquaredError(), ['mean_absolute_error'], 'linear'
    else:
        raise ValueError("Task must be set to 'Classification' or 'Regression'")


def argmax_with_ste(x):
    indices = tf.argmax(x, axis=-1)
    one_hot = tf.one_hot(indices, depth=tf.shape(x)[-1])
    return x + tf.stop_gradient(one_hot - x)  # STE approximation    

def replace_argmax_layer(old_model, replacement_function = argmax_with_ste):
    sequence_length, vocab_size = 24,25
    inputs = Input(shape=(24, 25))
    
    # Replace argmax layer
    reduced_input = keras.layers.Lambda(argmax_with_ste)(inputs)
    
    # Rebuild remaining layers
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=44, input_length=sequence_length)
    embedded_output = embedding_layer(reduced_input)
    
    gru = GRU(64, return_sequences=True)
    gru_output = gru(embedded_output)
    
    x = keras.layers.Flatten()(gru_output)
    
    if hasattr(old_model, 'feature_input'):
        feature_input = Input(shape=(old_model.feature_input.shape[1],))
        x = Concatenate()([x, feature_input])
        inputs = [inputs, feature_input]
    
    for dense_layer in old_model.layers[3:]:
        x = Dense(dense_layer.units, activation=dense_layer.activation)(x)
    
    output = Dense(1, activation=old_model.layers[-1].activation)(x)
    
    new_model = keras.Model(inputs=inputs, outputs=output)
    
    # Copy weights from old model, except for replaced layers
    for old_layer, new_layer in zip(old_model.layers, new_model.layers):
        if not isinstance(old_layer, keras.layers.Lambda):
            new_layer.set_weights(old_layer.get_weights())
    
    return new_model
