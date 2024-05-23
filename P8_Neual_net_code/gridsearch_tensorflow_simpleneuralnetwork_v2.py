import numpy as np
import pandas as pd
import tensorflow as tf
import scipy as sci
from sklearn.model_selection import KFold
import os
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

export_data = True
export_name = 'Tensorflow_GridSearch'

#thumbnail_dir = 'training_data/'  # Path to the directory containing thumbnail images
thumbnail_dir = 'training_data_NEWEST/'

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

print ("Creating grid and base model.")
u = np.arange(6, 10, 1)
HP_NUM_UNITS = 2**u

o = np.arange(0, 7, 1)
L2_REGULIZER = 1/(10**o)

HP_DROPOUT = np.arange(0.10, 0.70, 0.10)
HP_OPTIMIZER = np.array(['adam', 'sgd'])

#Setup GPU usage strategy
#mirrored_strategy = tf.distribute.MirroredStrategy(devices=['GPU'])

#number of images to consider at a time when running an epoch 2611//32 = 82
def train_test_model(num_units, l2_reg, dropout_rate, optimizer, num_layers):
    
    model = Sequential([
        layers.Flatten(),
        layers.Rescaling(1./255, input_shape=(1, 289))
    ])
    for i in range(0, num_layers):
        model.add(layers.Dense(num_units[i], kernel_regularizer=regularizers.l2(l2_reg), activation=tf.nn.relu))
        model.add(layers.Dropout(dropout_rate))      
    model.add(layers.Dense(1, activation='sigmoid'))
    
    
    model.compile(
        optimizer= get_optimizer(optimizer),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    #with mirrored_strategy.scope():
    history = model.fit(inputs[train], targets[train], validation_data=[inputs[test], targets[test]], callbacks=get_callbacks(), batch_size=batch_size, epochs=epochs, verbose=0) # Run with 1 epoch to speed things up for demo purposes
    
    return history

#Training data
inputs = X
targets = y

# K-fold Cross Validation model evaluation
batch_size = 32
epochs = 50

#Functions to use for decreasing the learning rate and early stopping 
STEPS_PER_EPOCH = len(X)//batch_size
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH*100,
    decay_rate=1,
    staircase=False)

def get_optimizer(name):
    if name == 'adam':
        return tf.keras.optimizers.Adam(lr_schedule)
    if name == 'sgd':
        return tf.keras.optimizers.experimental.SGD(lr_schedule)
    

def get_callbacks():
    return [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
    ]

num_folds = 5
max_layers = 3

print ("\n"+"Defining the K-fold Cross Validator.")
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=41)

layer_sessions = [] 
for k in range (1, max_layers+1):
    layer_sessions.append(len(HP_DROPOUT)*len(HP_OPTIMIZER)*len(L2_REGULIZER)*sci.special.binom(len(HP_NUM_UNITS)-1+k,k))

tot_sessions = sum(layer_sessions)

print('max sessions in memory: '+str(max(layer_sessions)))

session_num = 1
print ("\n"+"Starting gridsearch.")
for num_layers in range(2, max_layers):
    loop_session = 1
    parameters = []
    std_acc_fold = []
    avg_acc_fold = []
    avg_loss_fold = []
    if num_layers == 0:
        num_units = np.zeros(num_layers+1)
        for num_units[0] in HP_NUM_UNITS:
            for l2_reg in L2_REGULIZER:
                for dropout_rate in HP_DROPOUT:
                    for optimizer in HP_OPTIMIZER:
                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name +'/'+str(int(tot_sessions))+', run in current loop: '+str(loop_session)+'/'+str(int(layer_sessions[num_layers])))
                        print('num_units in layer: '+str(num_units)+', l2_reg: '+str(l2_reg)+', dropout_rate: '+str(dropout_rate)+', optimizer: '+str(optimizer)+', num_layers: '+str(num_layers+1))
                        
                        acc_per_fold = []
                        loss_per_fold = []
                        for train, test in kfold.split(inputs, targets):
                            history = train_test_model(num_units, l2_reg, dropout_rate, optimizer, num_layers)   
                            
                            val_acc_cv = history.history['val_accuracy']
                            val_loss_cv = history.history['val_loss']
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
                        
                        std_acc_fold.append(np.std(acc_per_fold))
                        avg_acc_fold.append(np.mean(acc_per_fold))
                        avg_loss_fold.append(np.mean(loss_per_fold))
                        
                        num_units_copy = num_units.copy()
                        parameters.append([num_units_copy[0], l2_reg, dropout_rate, str(optimizer), num_layers])
                        session_num += 1
                        loop_session += 1
        
        # getting best parameters
        parameters_best = parameters[np.argmax(avg_acc_fold)]
        print('\n'+'max avg. accuracy: '+str(np.round(max(avg_acc_fold),3))
              +'\n'+'best parameters: '
              +'num_units: '+str(parameters_best[0])
              +', l2_reg: '+str(parameters_best[1])
              +', dropout_rate: '+str(parameters_best[2])
              +', optimizer: '+str(parameters_best[3])
              +', num_layers: '+str(parameters_best[4]+1))   
        
        parameters = np.asanyarray(parameters)
        
        data = {'avg_acc_fold': avg_acc_fold,
                'std_acc_fold': std_acc_fold,
                'avg_loss_fold': avg_loss_fold,
                'num_units_in_layer1': parameters[:,0],
                'l2_regulizer': parameters[:,1],
                'dropout_rate': parameters[:,2],
                'optimizer': parameters[:,3],
                'num_layers': parameters[:,4]}
                                        
        df1 = pd.DataFrame(data)
        if export_data:
            print('\n'+'Exporting system data to CSV')
            df1.to_csv(export_name+'/'+'df'+str(num_layers+1)+'.csv', index=False)
            del data
            del df1
           
    if num_layers == 1:
        num_units = np.zeros(num_layers+1)
        for num_units[0] in HP_NUM_UNITS:
            for num_units[1] in HP_NUM_UNITS:
                if num_units[1] <= num_units[0]:
                    for l2_reg in L2_REGULIZER:
                        for dropout_rate in HP_DROPOUT:
                            for optimizer in HP_OPTIMIZER:
                                run_name = "run-%d" % session_num
                                print('--- Starting trial: %s' % run_name +'/'+str(int(tot_sessions))+', run in current loop: '+str(loop_session)+'/'+str(int(layer_sessions[num_layers])))
                                print('num_units in layer: '+str(num_units)+', l2_reg: '+str(l2_reg)+', dropout_rate: '+str(dropout_rate)+', optimizer: '+str(optimizer)+', num_layers: '+str(num_layers+1))
                                
                                acc_per_fold = []
                                loss_per_fold = []
                                for train, test in kfold.split(inputs, targets):
                                    history = train_test_model(num_units, l2_reg, dropout_rate, optimizer, num_layers)   
                                    
                                    val_acc_cv = history.history['val_accuracy']
                                    val_loss_cv = history.history['val_loss']
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
                                
                                std_acc_fold.append(np.std(acc_per_fold))
                                avg_acc_fold.append(np.mean(acc_per_fold))
                                avg_loss_fold.append(np.mean(loss_per_fold))
                                
                                num_units_copy = num_units.copy()
                                parameters.append([num_units_copy[0], num_units_copy[1], l2_reg, dropout_rate, str(optimizer), num_layers])
                                session_num += 1
                                loop_session += 1
                                
        # getting best parameters
        parameters_best = parameters[np.argmax(avg_acc_fold)]
        print('\n'+'max avg. accuracy: '+str(np.round(max(avg_acc_fold),3))
              +'\n'+'best parameters: '
              +'num_units: '+str(parameters_best[0])+','+str(parameters_best[1])
              +', l2_reg: '+str(parameters_best[2])
              +', dropout_rate: '+str(parameters_best[3])
              +', optimizer: '+str(parameters_best[4])
              +', num_layers: '+str(parameters_best[5]+1))
        
        parameters = np.asanyarray(parameters)
        
        data = {'avg_acc_fold': avg_acc_fold,
                'std_acc_fold': std_acc_fold,
                'avg_loss_fold': avg_loss_fold,
                'num_units_in_layer1': parameters[:,0],
                'num_units_in_layer2': parameters[:,1],
                'l2_regulizer': parameters[:,2],
                'dropout_rate': parameters[:,3],
                'optimizer': parameters[:,4],
                'num_layers': parameters[:,5]}
                                        
        df2 = pd.DataFrame(data)
        if export_data:
            print('\n'+'Exporting system data to CSV')
            df2.to_csv(export_name+'/'+'df'+str(num_layers+1)+'.csv', index=False)
            del data
            del df2
                
    if num_layers == 2:
        num_units = np.zeros(num_layers+1)
        for num_units[0] in HP_NUM_UNITS:
            for num_units[1] in HP_NUM_UNITS:
                if num_units[1] <= num_units[0]:
                    for num_units[2] in HP_NUM_UNITS:
                        if num_units[2] <= num_units[1]:
                            for l2_reg in L2_REGULIZER:
                                for dropout_rate in HP_DROPOUT:
                                    for optimizer in HP_OPTIMIZER:
                                        run_name = "run-%d" % session_num
                                        print('--- Starting trial: %s' % run_name +'/'+str(int(tot_sessions))+', run in current loop: '+str(loop_session)+'/'+str(int(layer_sessions[num_layers])))
                                        print('num_units in layer: '+str(num_units)+', l2_reg: '+str(l2_reg)+', dropout_rate: '+str(dropout_rate)+', optimizer: '+str(optimizer)+', num_layers: '+str(num_layers+1))
                                        acc_per_fold = []
                                        loss_per_fold = []
                                        for train, test in kfold.split(inputs, targets):
                                            history = train_test_model(num_units, l2_reg, dropout_rate, optimizer, num_layers)   
                                            
                                            val_acc_cv = history.history['val_accuracy']
                                            val_loss_cv = history.history['val_loss']
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
                                        
                                        std_acc_fold.append(np.std(acc_per_fold))
                                        avg_acc_fold.append(np.mean(acc_per_fold))
                                        avg_loss_fold.append(np.mean(loss_per_fold))
                                        
                                        num_units_copy = num_units.copy()
                                        parameters.append([num_units_copy[0], num_units_copy[1], num_units_copy[2], l2_reg, dropout_rate, str(optimizer), num_layers])
                                         
                                            
                                        session_num += 1
                                        loop_session += 1
                                        
        # getting best parameters
        parameters_best = parameters[np.argmax(avg_acc_fold)]
        print('\n'+'max avg. accuracy: '+str(np.round(max(avg_acc_fold),3))
              +'\n'+'best parameters: '
              +'num_units: '+str(parameters_best[0])+','+str(parameters_best[1])+','+str(parameters_best[2])
              +', l2_reg: '+str(parameters_best[3])
              +', dropout_rate: '+str(parameters_best[4])
              +', optimizer: '+str(parameters_best[5])
              +', num_layers: '+str(parameters_best[6]+1))
        
        parameters = np.asanyarray(parameters)
                        
        data = {'avg_acc_fold': avg_acc_fold,
                'std_acc_fold': std_acc_fold,
                'avg_loss_fold': avg_loss_fold,
                'num_units_in_layer1': parameters[:,0],
                'num_units_in_layer2': parameters[:,1],
                'num_units_in_layer3': parameters[:,2],
                'l2_regulizer': parameters[:,3],
                'dropout_rate': parameters[:,4],
                'optimizer': parameters[:,5],
                'num_layers': parameters[:,6]}
                                        
        df3 = pd.DataFrame(data)
        if export_data:
            print('\n'+'Exporting system data to CSV')
            df3.to_csv(export_name+'/'+'df'+str(num_layers+1)+'.csv', index=False)
            del data
            del df3
                                    
    if num_layers == 3:
        num_units = np.zeros(num_layers+1)
        for num_units[0] in HP_NUM_UNITS:
            for num_units[1] in HP_NUM_UNITS:
                if num_units[1] <= num_units[0]:
                    for num_units[2] in HP_NUM_UNITS:
                        if num_units[2] <= num_units[1]:
                            for num_units[3] in HP_NUM_UNITS: 
                                if num_units[3] <= num_units[2]:
                                    for l2_reg in L2_REGULIZER:
                                        for dropout_rate in HP_DROPOUT:
                                            for optimizer in HP_OPTIMIZER:
                                                run_name = "run-%d" % session_num
                                                print('--- Starting trial: %s' % run_name +'/'+str(int(tot_sessions))+', run in current loop: '+str(loop_session)+'/'+str(int(layer_sessions[num_layers])))
                                                print('num_units in layer: '+str(num_units)+', l2_reg: '+str(l2_reg)+', dropout_rate: '+str(dropout_rate)+', optimizer: '+str(optimizer)+', num_layers: '+str(num_layers+1))
                                                
                                                acc_per_fold = []
                                                loss_per_fold = []
                                                for train, test in kfold.split(inputs, targets):
                                                    history = train_test_model(num_units, l2_reg, dropout_rate, optimizer, num_layers)   
                                                    
                                                    val_acc_cv = history.history['val_accuracy']
                                                    val_loss_cv = history.history['val_loss']
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
                                                
                                                std_acc_fold.append(np.std(acc_per_fold))
                                                avg_acc_fold.append(np.mean(acc_per_fold))
                                                avg_loss_fold.append(np.mean(loss_per_fold))
                                                
                                                num_units_copy = num_units.copy()
                                                parameters.append([num_units_copy[0], num_units_copy[1], num_units_copy[2], num_units_copy[3], l2_reg, dropout_rate, str(optimizer), num_layers])
                                                session_num += 1
                                                loop_session += 1
                                                
        # getting best parameters
        parameters_best = parameters[np.argmax(avg_acc_fold)]
        print('\n'+'max avg. accuracy: '+str(np.round(max(avg_acc_fold),3))
              +'\n'+'best parameters: '
              +'num_units: '+str(parameters_best[0])+','+str(parameters_best[1])+','+str(parameters_best[2])+','+str(parameters_best[3])
              +', l2_reg: '+str(parameters_best[4])
              +', dropout_rate: '+str(parameters_best[5])
              +', optimizer: '+str(parameters_best[6])
              +', num_layers: '+str(parameters_best[7]+1))
        
        parameters = np.asanyarray(parameters)
        
        data = {'avg_acc_fold': avg_acc_fold,
                'std_acc_fold': std_acc_fold,
                'avg_loss_fold': avg_loss_fold,
                'num_units_in_layer1': parameters[:,0],
                'num_units_in_layer2': parameters[:,1],
                'num_units_in_layer3': parameters[:,2],
                'num_units_in_layer4': parameters[:,3],
                'l2_regulizer': parameters[:,4],
                'dropout_rate': parameters[:,5],
                'optimizer': parameters[:,6],
                'num_layers': parameters[:,7]}

        df4 = pd.DataFrame(data)
        
        if export_data:
            print('\n'+'Exporting system data to CSV')
            df4.to_csv(export_name+'/'+'df'+str(num_layers+1)+'.csv', index=False)
            del data
            del df4
            