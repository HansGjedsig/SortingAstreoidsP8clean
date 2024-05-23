import os
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import io, transform
from sklearn.utils import shuffle
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import pandas as pd 

SaveModel = False
get_missclass = False
export_place = 'ml_source_screeningP8'

thumbnail_dir = 'training_dataNYNY/'  # Path to the directory containing thumbnail images
#thumbnail_dir = 'training_data_backup/'
model_filename = 'tensorflow_convneuralnet_model.keras' # name of file containing trained model

# Define classes for positive and negative examples
positive_class = 'good_thumbs'
negative_class = 'bad_thumbs'

X = []  # List to store image data
y = []  # List to store class labels
X_filenames = []

for class_dir in os.listdir(thumbnail_dir):
    if os.path.isdir(os.path.join(thumbnail_dir, class_dir)):
        class_label = class_dir

        # Determine if the class is positive or negative
        if class_label == positive_class:
            class_label = 1  # Positive class
        elif class_label == negative_class:
            class_label = 0  # Negative class
            print("class-label",class_label)
        else:
            continue 

        class_path = os.path.join(thumbnail_dir, class_dir)
        
        for image_file in os.listdir(class_path):
            if image_file.endswith('.png'):
                image_path = os.path.join(class_path, image_file)
                image = io.imread(image_path)
                if image.shape != (17, 17):
                    print(f"This image is the wrong size: {image.shape}",image_file)
                X.append(image)
                X_filenames.append(image_path)
                y.append(class_label)

# Convert lists to NumPy arrays
X = np.asarray(X)
y = np.asarray(y)

print ("\n"+"Splitting data into training and testing.")
random_state = 41
inputs_train, inputs_test, targets_train, targets_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
filenames_inputs_train, filenames_inputs_test, targets_train, targets_test = train_test_split(X_filenames, y, test_size=0.2, random_state=random_state)
#file_shuffle = shuffle(X_filenames, random_state=random_state)

file_shuffle = np.concatenate((filenames_inputs_train, filenames_inputs_test), axis=0)
inputs_tot = np.concatenate((inputs_train, inputs_test), axis=0)
targets_tot = np.concatenate((targets_train, targets_test), axis=0)

inputs_train, inputs_test = inputs_train.reshape(len(inputs_train),17,17,1), inputs_test.reshape(len(inputs_test),17,17,1)

inputs_cv = shuffle(inputs_tot, random_state= 42)
targets_cv = shuffle(targets_tot, random_state= 42)
epochs = 150
batch_size = 32

print ("Creating and training the MLP classifier.")

def get_optimizer():
    STEPS_PER_EPOCH = len(X)//batch_size
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps=STEPS_PER_EPOCH*1000,
        decay_rate=1,
        staircase=False)
    
    return tf.keras.optimizers.Adam(lr_schedule)

def get_callbacks():
    return [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)
    ]

def model():
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(17,17,1)),
        layers.Conv2D(16, (3,3), activation='relu', input_shape=(17, 17,1)),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.1), activation=tf.nn.relu),
        layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
        ])
    model.compile(optimizer = get_optimizer(),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model

mlp_classifier = model()   
history = mlp_classifier.fit(inputs_train, targets_train, validation_data=[inputs_test, targets_test], callbacks=get_callbacks(), epochs=epochs, batch_size=batch_size)

print ("\n"+"Defining the K-fold Cross Validator.")
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=41)

#Cross validation
acc_per_fold = []
loss_per_fold = []
for train, test in kfold.split(inputs_cv, targets_cv):
    model_cv = model()
    history_cv = model_cv.fit(inputs_cv[train], targets_cv[train], validation_data=[inputs_cv[test], targets_cv[test]], callbacks=get_callbacks(), epochs=epochs, batch_size=batch_size)   
    
    val_acc_cv = history_cv.history['val_accuracy']
    val_loss_cv = history_cv.history['val_loss']
    acc_per_fold.append(val_acc_cv[-1])
    loss_per_fold.append(val_loss_cv[-1])
    
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

#mlp_classifier.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
print('------------------------------------------------------------------------')
print(f'> Train_acc: {acc[-1]}')
print(f'> Validation_acc: {val_acc[-1]}')
print('------------------------------------------------------------------------')


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

if SaveModel:
    #joblib.dump(mlp_classifier, model_filename)
    mlp_classifier.save(model_filename)
    
    training_data = {'accuracy': acc,
            'val_accuracy': val_acc,
            'loss': loss,
            'val_loss': val_loss}
                                
    fold_data = {'fold_val_accuracy': acc_per_fold,
            'fold_val_loss': loss_per_fold
        }
    
    t_df = pd.DataFrame(training_data)
    f_df = pd.DataFrame(fold_data)
    
    t_df.to_csv('df_training_histroy_convneuralnet_model'+'.csv', index=False)
    f_df.to_csv('df_crossvalidation_convneuralnet_model'+'.csv', index=False)
    
    print(f'Model saved as {model_filename} and training history have been saved')
    


if get_missclass:
    #predictions = predict(inputs_tot)
    predictions = mlp_classifier.predict(inputs_tot)
    predictions = np.reshape(predictions, len(predictions))
    labels = (predictions > 0.50)*1
    
    missclass = (labels !=targets_tot) 
    
    data = {'path': file_shuffle[missclass],
            'prediction': labels[missclass],
            'assigned': targets_tot[missclass],
            'good_pred': predictions[missclass],
            'bad_pred': 1-predictions[missclass]}
                                    
    df = pd.DataFrame(data)
    
    print('\n'+'Exporting system data to CSV')
    df.to_csv(thumbnail_dir+'missclassifications_conv_NN'+'.csv', index=False)

