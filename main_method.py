import preprocess as ps
import classification_model as cm

data_path = r"C:\Users\Bhaskar Marapelli\Downloads\gaussian_filtered_images\gaussian_filtered_images"
da=ps.Dataprocess()
dataset_df = da.generate_images_dataset(data_path)
# Save the DataFrame to a CSV file
csv_filename = "diabetic_retinopathy_dataset.csv"
dataset_df.to_csv(csv_filename, index=False)
print(f"Dataset information saved to {csv_filename}")
data=da.Generate_new_feature_in_csv(csv_filename)
binary_csv="new_dataset.csv"
data.to_csv(binary_csv, index=False)
sklearn_class_weights=da.data_EDA(binary_csv)

train_batches, val_batches, test_batches=da.ImageDataGenerator_Data(binary_csv)

obj=cm.Classifcation_model()
model=obj.simple_model()
model=obj.compile_model(model,opt='Adam')
history=model.fit(train_batches, epochs=10, validation_data=val_batches)
obj.summary_model(history)

model.save("model_binary.h5")

loss, acc = model.evaluate(test_batches, verbose=1)
# print("Loss: ", loss)
print("Accuracy: ", acc)
"""
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


def predict_class(path):
    img = cv2.imread(path)

    RGBImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    RGBImg= cv2.resize(RGBImg,(224,224))
    plt.imshow(RGBImg)
    image = np.array(RGBImg) / 255.0
    new_model = tf.keras.models.load_model("model_binary.h5")
    predict=new_model.predict(np.array([image]))
    per=np.argmax(predict,axis=1)
    if per==1:
        print('No DR')
    else:
        print('DR')
predict_class('C:/Users/Bhaskar Marapelli/Downloads/gaussian_filtered_images/gaussian_filtered_images/Mild/0024cdab0c1e.png')

# creating instances of all the 3 different models
model_adam =obj.compile_model(model,'adam')
model_sgd = obj.compile_model(model,'sgd')
model_rmsprop = obj.compile_model(model,'rmsprop')

obj.compare_models([model_adam, model_sgd, model_rmsprop], train_batches,val_batches,3)

"""