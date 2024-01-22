import preprocess as ps
import classification_model as cm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Set the path to your data
data_path = r"C:\Users\Bhaskar Marapelli\Downloads\gaussian_filtered_images\gaussian_filtered_images"

# Initialize DataProcessor
da = ps.Dataprocess()

# Generate and save the dataset information to a CSV file
dataset_df = da.generate_images_dataset(data_path)
csv_filename = "diabetic_retinopathy_dataset.csv"
dataset_df.to_csv(csv_filename, index=False)
print(f"Dataset information saved to {csv_filename}")

# Generate new features in the CSV file
binary_csv = "new_dataset.csv"
data = da.Generate_new_feature_in_csv(csv_filename)
data.to_csv(binary_csv, index=False)



# Load train, validation, and test batches using the augmented generator
train_batches, val_batches, test_batches = da.ImageDataGenerator_Data(binary_csv)

#train_batches=train_datagen

# Initialize ClassificationModel
obj = cm.Classifcation_model()

# Build and compile the model
model = obj.simple_model()
model = obj.compile_model(model, opt='Adam')

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Define the paths to your train, validation, and test data directories
train_data_dir = 'train'
validation_data_dir = 'val'
test_data_dir = 'test'

# Set the batch size and image dimensions
batch_size = 128
img_height, img_width = 224, 224  # Adjust these dimensions based on your requirements

# Create an ImageDataGenerator for data augmentation on the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

# Create an ImageDataGenerator for validation and test sets (no augmentation)
val_test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

# Create generators for train, validation, and test sets
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # or 'binary' depending on your problem
    shuffle=True
)

validation_generator = val_test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # or 'binary'
    shuffle=False  # No shuffling for validation
)

test_generator = val_test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # or 'binary'
    shuffle=False  # No shuffling for test
)

# Now, use these generators in your model.fit() call
history = model.fit(
    train_generator,
    epochs=3,
    validation_data=validation_generator
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


model.save("model_multi_class.h5")
