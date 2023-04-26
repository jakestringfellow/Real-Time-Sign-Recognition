'''
Jake Stringfellow
Real-Time Sign Language Recognition
Building, training, analyzing, and modifying a deep network for real-time hand gesture
recognition based on the ASL (American Sign Language) alphabet.

modelTraining.py: Building, training, and analyzing a deep network on the Sign-MNIST dataset
                  with the intention of hand gesture recognition
'''

# Import statements
import string
# Helps to load dataframe in 2D array format
import pandas as pd
# Numpy arrays are fast and perform large computations quickly
import numpy as np
# Open-source library for Machine Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import save_model
# Library used to draw visualizations
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator

'''
    loadData
    Method that loads data and labels from csv files
    Params: 
        path: path to the csv file
    Returns: 
        x, y: datasets used for training/testing
'''
def loadData(path):
    df = pd.read_csv(path)
    y = np.array([label if label < 9
                  else label - 1 for label in df['label']])
    df = df.drop('label', axis=1)
    x = np.array([df.iloc[i].to_numpy().reshape((28, 28))
                  for i in range(len(df))]).astype(float)
    x = np.expand_dims(x, axis=3)
    y = pd.get_dummies(y).values

    return x, y

'''
    class NeuralNetwork
    Stores and creates the Sequential Model CNN 
'''
class NeuralNetwork():
    def __init__(self):
        self.model = tf.keras.models.Sequential([
                        tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3, 3),
                                            activation='relu',
                                            input_shape=(28, 28, 1)),
                        tf.keras.layers.MaxPooling2D(2, 2),

                        tf.keras.layers.Conv2D(filters=64,
                                kernel_size=(3, 3),
                                activation='relu'),
                        tf.keras.layers.MaxPooling2D(2, 2),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dense(256, activation='relu'),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Dense(24, activation='softmax')
                        ])

'''
    visualizeDataset
    Method that plots a sample of the dataset info
    Params: 
        class_names:
        X_train: Image data for training set
        Y_train: Image labels for training set
    Returns: 
        None
'''
def visualizeDataset(class_names, X_train, Y_train):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i].squeeze(), cmap=plt.cm.binary)
        plt.xlabel(class_names[np.argmax(Y_train, axis=1)[i]])
    plt.tight_layout()
    plt.show()

'''
    vizualizeTrainingData
    Method that plots the training history of the given model
    Params: 
        history: Data provided by the training of the mdoel
    Returns: 
        None
'''
def vizualizeTrainingData(history):
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot()
    history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
    plt.show()


'''
    Main Function
    Loads data from the Sign-MNIST dataset, builds and trains cnn model, plots dataset info and
    training data, saves model state.
'''
def main():

    # Obtain training and testing datasets form their respective csv files
    X_train, Y_train = loadData('support/signData/sign_mnist_train.csv')
    X_test, Y_test = loadData('support/signData/sign_mnist_test.csv')


    # Print the shape of the datasets (Number of images, height, width, color channel) (total labels, unique labels)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    # The class names are every letter of the alphabet except for j and z, which the sign-MNIST
    # does not account for due to them requiring motion to gesture
    class_names = list(string.ascii_lowercase[:26].replace(
        'j', '').replace('z', ''))

    # Print the list of sign-able letters
    print(class_names)

    # Write the letters to a file to be accessed in the main program
    with open('support/sign.names', 'w') as f:
        for line in class_names:
            f.write(f"{line}\n")

    # Create data for training evaluation
    visualizeDataset(class_names, X_train, Y_train)

    # Initialize our model
    model = NeuralNetwork().model

    # Print the summary of our model's architecture
    model.summary()

    # Compile the model
    model.compile(
        optimizer='adam',                   # Help optimize cost function
        loss='categorical_crossentropy',    # Loss function
        metrics=['accuracy']                # Evalutate model, predict training and validation
    )

    # Train the model
    history = model.fit(X_train, Y_train,
                        validation_data=(X_test, Y_test),
                        epochs=5,
                        verbose=1)

    # Plot training history
    vizualizeTrainingData(history)

    # Evaluate our model's performance, find out how accurate the model was
    model.evaluate(X_test,Y_test)

    # Save the model to be reused in the live recognition task
    model.save("ASL_model")

if __name__ == "__main__":
    main()