#import prepareformodel as pm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random,os
import seaborn as sns
import shutil
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import accuracy_score
class Dataprocess:
    def generate_images_dataset(self, data_path):
        image_paths = []
        labels = []

        for class_label, class_name in enumerate(os.listdir(data_path)):
            class_path = os.path.join(data_path, class_name)
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                image_paths.append(file_path)
                labels.append(class_label)
            print('label{}]n'.format(class_path))
            print('Number of train images : {} \n'.format(len(image_paths)))
            print('Number of train images labels: {} \n'.format(len(labels)))
        print('Number of train images : {} \n'.format(len(image_paths)))
        print('Number of train images labels: {} \n'.format(len(labels)))
        df = pd.DataFrame({"Image_Path": image_paths, "Label": labels})
        print("Dataset Information:")
        print(df.head())
        return df

    def Generate_new_feature_in_csv(self, input):
        '''Generate new feature, Mapping the output feature'''

        data = pd.read_csv(input)
        Defect_binary = {
            0: 'DR',
            1: 'DR',
            2: 'NO_DR',
            3: 'DR',
            4: 'DR'}
        diagnosis_all_dict = {2: 'No_DR', 0: 'Mild', 1: 'Moderate', 4: 'Severe', 3: 'Proliferate_DR', }
        data['binary_type'] = data['Label'].map(Defect_binary.get)
        data['type'] = data['Label'].map(diagnosis_all_dict.get)
        return data

    def data_EDA(self,csv_path):
        df = pd.read_csv(csv_path)
        # Assuming the column names in your CSV file are 'image_path' and 'Label'
        image_path_column = 'Image_Path'
        label_column = 'type'
        # Create a count plot using seaborn
        plt.figure(figsize=(10, 6))
        sns.countplot(x=label_column, data=df)
        plt.title('Distribution of Image Labels')
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()

        #calculate class weights
        #sklearn_class_weights = class_weight.compute_class_weight('classes',np.unique(df['type']),df['type'])
        #print(sklearn_class_weights)
        #return sklearn_class_weights




    def data_separation(self,input):
        '''Train_Test_Separation'''
        data = self.Generate_new_feature_in_csv(input)
        train, val = train_test_split(data, test_size=0.2, stratify=data['type'])
        train, test = train_test_split(train, test_size=0.15 / (1 - 0.15), stratify=train['type'])
        return train, val, test

    # Create working directory
    def mkdir_separate_image(self,input):
        '''
        1. Initiate the data
        2. Create Train,Test,val directory
        3. Separate train,test,val copy the images
        '''

        train, val, test = self.data_separation(input)
        base_dir = ''
        train_dir = os.path.join(base_dir, 'train')
        val_dir = os.path.join(base_dir, 'val')
        test_dir = os.path.join(base_dir, 'test')
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir)
        if os.path.exists(val_dir):
            shutil.rmtree(val_dir)
        os.makedirs(val_dir)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir)
        # Copy images to respective working train,test,val directory
        src_dir = r'C:\Users\Bhaskar Marapelli\Downloads\diabetic_retinopathy'
        for index, row in train.iterrows():
            diagnosis = row['type']
            binary_diagnosis = row['binary_type']
            id_code = row['Image_Path']
            srcfile = os.path.join(src_dir, diagnosis, id_code)
            dstfile = os.path.join(train_dir, diagnosis)
            os.makedirs(dstfile, exist_ok=True)
            shutil.copy(srcfile, dstfile)
        for index, row in test.iterrows():
            diagnosis = row['type']
            binary_diagnosis = row['binary_type']
            id_code = row['Image_Path']
            srcfile = os.path.join(src_dir, diagnosis, id_code)
            dstfile = os.path.join(test_dir, diagnosis)
            os.makedirs(dstfile, exist_ok=True)
            shutil.copy(srcfile, dstfile)
        for index, row in val.iterrows():
            diagnosis = row['type']
            binary_diagnosis = row['binary_type']
            id_code = row['Image_Path']
            srcfile = os.path.join(src_dir, diagnosis, id_code)
            dstfile = os.path.join(val_dir, diagnosis)
            os.makedirs(dstfile, exist_ok=True)
            shutil.copy(srcfile, dstfile)
        return train, val, test

    def ImageDataGenerator_Data(self,input):
        train, val, test = self.mkdir_separate_image(input)
        train_path = 'train'
        val_path = 'val'
        test_path = 'test'
        train_batches = ImageDataGenerator(rescale=1 / 255.).flow_from_directory(train_path, target_size=(224, 224),
                                                                                 shuffle=True)
        val_batches = ImageDataGenerator(rescale=1 / 255.).flow_from_directory(val_path, target_size=(224, 224),
                                                                               shuffle=True)
        test_batches = ImageDataGenerator(rescale=1 / 255.).flow_from_directory(test_path, target_size=(224, 224),
                                                                                shuffle=True)
        return train_batches, val_batches, test_batches



