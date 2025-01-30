import os
import cv2 # load and process images
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def parse_data():
    mnist = tf.keras.datasets.mnist

    # X is handwritten data itself, Y is classification (digit)
    (x_train,y_train), (x_test,y_test) = mnist.load_data()

    # Normalize data (between 0 and 1)
    x_train = tf.keras.utils.normalize(x_train, axis = 1)
    x_test = tf.keras.utils.normalize(x_test, axis = 1)

    return (x_train,y_train),(x_test,y_test)


# # Initialize and train model itself (Basic Sequential NN)
def train_model(x_train,y_train):
    model = tf.keras.models.Sequential() 
    # Flattens input images that are 28x28 pixels to a 1x784 image
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    # Add dense layer (each neuron connected to each other layer neurons)
    model.add(tf.keras.layers.Dense(128,activation='relu')) # Activation function = relu
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    # Output layer (10 units represent individual digits)
    model.add(tf.keras.layers.Dense(10,activation='softmax')) # Softmax makes sure all outputs (all digit values) all add up to 1

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)

    model.save('handwritten.keras')

def print_metrics(x_test,y_test):
    model = tf.keras.models.load_model('handwritten.keras')

    loss, accuracy = model.evaluate(x_test,y_test)

    print(f"Loss: {loss * 100}%")
    print(f"Accuracy: {accuracy * 100}%")

# Use own numbers on model!
def test_own_numbers(model):
    image_number = 1
    while os.path.isfile(f"digits/digit{image_number}.png"):
        try:
            img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
            print("Got here")
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            # Prints the digit with the highest value (highest likelihood)
            print(f"This digit is probably a {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print("Error!")
        finally:
            image_number += 1

if __name__ == '__main__':

    (x_train,y_train),(x_test,y_test) = parse_data()

    train_model(x_train,y_train)
    
    print_metrics(x_test,y_test)

    nn_model = tf.keras.models.load_model('handwritten.keras')

    test_own_numbers(nn_model)

