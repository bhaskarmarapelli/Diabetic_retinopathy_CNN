import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
class Classifcation_model:
    def simple_model(self):
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                Flatten(),
                Dense(32, activation='relu'),
                Dense(64, activation='relu'),
                Dense(128, activation='relu'),
                Dense(5, activation='softmax'),
            ])
            return model
    def compile_model(self,model,opt):
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    def summary_model(self,history):
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    # Function to compare and plot the performance of different models
    def compare_models(self,models,train_batches,val_batches, epochs):
        histories = []
        for model in models:
            history = model.fit(train_batches, epochs=epochs, validation_data=val_batches)
            histories.append(history)

        # Plot the accuracy for each optimizer
        plt.figure(figsize=(10, 6))
        for i, history in enumerate(histories):
            plt.plot(history.history['accuracy'], label=f'Model {i + 1}')

        plt.title('Model Training Accuracy Comparison')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show(block=True)



