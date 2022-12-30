"""Load and process the data."""
import numpy as np
import tensorflow as tf
import keras
import pywt
import lib.readData as readData

class Model(keras.Model):
    def __init__(self, input_shape, num_classes, batch_size, epochs, name=None, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_shape
        self.num_of_classes = num_classes
        self.size_of_batch = batch_size
        self.num_of_epochs = epochs

        self.history = keras.callbacks.History()
        # Creating the convolutional network model
        self.model = keras.models.Sequential()
        
        self.build_model()

    @tf.function
    def build_model(self):
        # A kernel of 5x5 sweeps the 127x127x9 into 32 feature layers. The
        # kernel moves by strides of 1x1. Each output layer has 122x122x9.
        self.model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                        activation='relu',
                        input_shape=self.input_dim))
        
        # From each feature layer, "Max pooling" selects the max element in a 
        # window of pool_size, which sweeps in strides of 2x2.
        # Output has a spatial shape: 
        # output_shape = math.floor((input_shape - pool_size)/ strides) + 1
        # = mat.floor(((122,122)-(2,2))/(2,2)) + 1 = (61,61)
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # 2nd set of conv. layer. Output size (56,56)
        self.model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))

        # 2nd set of max pooling: 
        # output_size = mat.floor(((56,56) - (2,2))/(1,1)) + 1 = (55,55)
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # Flat each of the (55,55) layer into one array.
        self.model.add(keras.layers.Flatten())

        # Fully connected layer with 1000 perceptrons.
        self.model.add(keras.layers.Dense(1000, activation='relu'))

        # Output layer with the number of classes as the output.
        self.model.add(keras.layers.Dense(self.num_of_classes, activation='softmax'))

        print("Compiling the model...")
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])

    @tf.function
    def train(self, x_train, y_train):        
        self.model.fit(x_train, y_train,
            batch_size=self.size_of_batch,
            epochs=self.num_of_epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[self.history])
    
    @tf.function
    def test(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)

if __name__ == "__main__":
    folder_ucihar = './data/UCI_HAR/' 
    # Read a data already separated in training and testing sets.
    train_signals_ucihar, train_labels_ucihar, \
    test_signals_ucihar, test_labels_ucihar = readData.load_ucihar_data(folder_ucihar)
    
    # The training set contains 7352 signals where each signal has 128
    # measurement samples and 9 components.
    print("Training set shape:", train_signals_ucihar.shape)
    print("Testing set shape:", test_signals_ucihar.shape)
    
    print("Sample of the test data")
    print(test_signals_ucihar[0, :, :])

    print("test labels shape: ", len(test_labels_ucihar))
    print("Sample of the first test data label")
    print(test_labels_ucihar[0])

    print("Unique labels:")
    list_set = list(set(test_labels_ucihar))
    for item in list_set:
        print(item)

    # Prepare the datasets.
    # Select a bunch from the training and test sets.
    scales = range(1,128)
    waveletname = 'morl'
    train_size = 5000
    test_size= 500

    # A 128 length signal is decomposed into 127 x 127 pixels with the 
    # cwt. As each individual experiment has 9 signals, the approach is
    # to stack 9 cwt images as a convolution network input.
    train_data_cwt = np.ndarray(shape=(train_size, 127, 127, 9))
    print("Processing the training set...")
    for ii in range(0,train_size):
        if ii % 1000 == 0:
            print(ii)
        for jj in range(0,9):
            signal = train_signals_ucihar[ii, :, jj]
            coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
            coeff_ = coeff[:,:127]
            train_data_cwt[ii, :, :, jj] = coeff_

    test_data_cwt = np.ndarray(shape=(test_size, 127, 127, 9))
    print("Processing the test set...")
    for ii in range(0,test_size):
        if ii % 100 == 0:
            print(ii)
        for jj in range(0,9):
            signal = test_signals_ucihar[ii, :, jj]
            coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
            coeff_ = coeff[:,:127]
            test_data_cwt[ii, :, :, jj] = coeff_

    # Rearrange the labels to match the previous processing.
    train_labels_ucihar = list(map(lambda x: int(x) - 1, train_labels_ucihar))
    test_labels_ucihar = list(map(lambda x: int(x) - 1, test_labels_ucihar))

    x_train = train_data_cwt
    y_train = list(train_labels_ucihar[:train_size])
    x_test = test_data_cwt
    y_test = list(test_labels_ucihar[:test_size])

    # Training Step
    print("Building the convolutional network...")
    
    img_x = 127
    img_y = 127
    img_z = 9
    input_shape = (img_x, img_y, img_z)

    num_classes = 6
    batch_size = 4
    epochs = 10
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    conv_model = Model(input_shape=(img_x, img_y, img_z), num_classes=6, 
        batch_size=4, epochs=10, name="Conv Model")

    conv_model.train(x_train, y_train)

    train_score = conv_model.test(x_train, y_train)
    print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
    test_score = conv_model.test(x_test, y_test)
    print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))
