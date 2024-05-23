import os
import numpy as np
from skimage import io
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import time as time


thumbnail_dir = 'training_data'  # Path to the directory containing thumbnail images
model_filename = 'mlp_model.joblib' # name of file containing trained model

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
        else:
            continue 

        class_path = os.path.join(thumbnail_dir, class_dir)
        
        for image_file in os.listdir(class_path):
            if image_file.endswith('.png'):
                image_path = os.path.join(class_path, image_file)
                image = io.imread(image_path)
                #print("IMAGE",image)
                if image.shape != (17, 17):
                    print(f"This image is the wrong size: {image.shape}",image_file)
                image_array=image.flatten()
                #image_array = image_array.reshape(1, -1)
                #print("appending",image_array)

                X.append(image_array)
                X_filenames.append(image_path)
                y.append(class_label)


# Convert lists to NumPy arrays and normalize it
X = np.array(X)/255
Y = np.array(y)

# removes repeats in the training data if remove_repeats=True
remove_repeats = False
if remove_repeats:
    X_t = X
    X, indices = np.unique(X,axis=0, return_index=True)
    Y = Y[indices]
    X_filenames = np.array(X_filenames)[indices]
    
#Shuffles and split the data (included the paths) according to random_state
random_state = 42
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=random_state)
X_train_path,X_test_path,Y_train,Y_test = train_test_split(X_filenames,Y,test_size=0.2, random_state=random_state)


# Creates a concatenated shuffled list of all the data set. usefull for cross validatation and ensures that good and bad thumbs does not show up in order.
X_shuff = np.concatenate([X_train,X_test])
Y_shuff = np.concatenate([Y_train,Y_test])
paths_shuff = np.concatenate([X_train_path,X_test_path])

#set up the pipeline NMF --> SVM and defines global parameters.
pipe = Pipeline([("nmf",NMF()),("svm",SVC())])
pipe.set_params(svm__probability=True,nmf__random_state=random_state,svm__kernel="rbf")

# Does a grid search on lists of parameters on the training data. Currently set to the best fitted parameters.
#param_grid = {'nmf__max_iter':[2000],'svm__C':10**np.linspace(-3,6,10),'svm__gamma':10**np.linspace(-4,5,10),'nmf__n_components':[10,11,12,13,14,15,16]}
param_grid = {'nmf__max_iter':[2000],'svm__C':[10],'svm__gamma':[1],'nmf__n_components':[14]}
t1 = time.time()
grid = GridSearchCV(pipe, param_grid=param_grid,cv=5,verbose=1)
grid.fit(X_train,Y_train)
print(str(time.time()-t1)+" sekunder!")
print("best cross-validated accuracy: {:.3f}".format(grid.best_score_))
print("Best params: {}".format(grid.best_params_))
print("-"*50)

#sets the best parameters from the grid seach as the parameters for the pipe.
pipe.set_params(nmf__max_iter=grid.best_params_.get("nmf__max_iter"),nmf__n_components=grid.best_params_.get("nmf__n_components"),nmf__random_state=random_state,svm__kernel="rbf",svm__C=grid.best_params_.get("svm__C"),svm__gamma=grid.best_params_.get("svm__gamma"))
pipe.fit(X_train,Y_train)

#cross validates on the entire data set and prints results.
res = cross_validate(pipe, X_shuff,Y_shuff,cv=5,return_train_score=True)
res_df = pd.DataFrame(res)
print("5-split cross validation means:\n"+str(res_df.mean()))
print("-"*50)
print("test score")

#prints accuracy on the training set and test set and saves the model to a joblib file
y_valid_pred = pipe.predict(X_test)
#print("yvaldpred",pipe.predict_proba(X_test))
y_valid_pred_train = pipe.predict(X_train)

#%%
def As(y_test,y_pred):   
    return np.sum(y_test==y_pred)/len(y_test)

def Rs(y_test,y_pred):   
   return np.sum(y_pred[y_test==y_pred]==1)/(np.sum(y_pred[y_test==y_pred]==1)+np.sum(y_pred[y_test!=y_pred]==1))

def Ps(y_test,y_pred):
   return np.sum(y_pred[y_test==y_pred]==1)/(np.sum(y_pred[y_test==y_pred]==1)+np.sum(y_pred[y_test!=y_pred]==0))

print(As(Y_test, y_valid_pred))
print(Rs(Y_test, y_valid_pred))
print(Ps(Y_test, y_valid_pred))


accuracy_valid = accuracy_score(Y_test, y_valid_pred)
print(f'Accuracy on the validation set: {accuracy_valid:.2f}')

accuracy_valid_train = accuracy_score(Y_train, y_valid_pred_train)
print(f'Accuracy on the training set: {accuracy_valid:.2f}')

joblib.dump(pipe, model_filename)
print(f'Model saved as {model_filename}')

#%% saves the CV results to a dataframe and saves all misclasifications to a csv (used in misclasification_checker.py)
grid_cv_results = pd.DataFrame(grid.cv_results_)
grid_cv_results.to_csv("gridCVresults6.csv")
prediction = pipe.predict(X_shuff)
predict_per = pipe.predict_proba(X_shuff)
prediction_per0 = []
prediction_per1 = []
path_mis_class = []
pred_mis_class = []
true_mis_class = []
for i in range(len(Y_shuff)):
    if Y_shuff[i] != prediction[i]:
        prediction_per0.append(predict_per[i,0])
        prediction_per1.append(predict_per[i,1])
        path_mis_class.append(paths_shuff[i])
        pred_mis_class.append(prediction[i])
        true_mis_class.append(Y_shuff[i])

data = {"path":path_mis_class,
        "prediction":pred_mis_class,
        "Assigned":true_mis_class,
        "Good_Prediction%":prediction_per1,
        "Bad_Prediction%":prediction_per0}
df = pd.DataFrame(data)
df.to_csv("misclasifications.csv")


#%% Shows the feature components if plot_components = True
weights = pipe.named_steps["nmf"].transform(X_test)
here = np.c_[weights,y_valid_pred,Y_test]

df = pd.DataFrame(here)

plot_components = True
if plot_components:
    fig, axes = plt.subplots(4%grid.best_params_.get("nmf__n_components"),4,figsize=(15,15))
    for i, (component,ax) in enumerate(zip(pipe.named_steps["nmf"].components_,axes.ravel())):
        ax.imshow(component.reshape(17,17),cmap="Greys_r")
        ax.set_title("{}.component".format(i),size=30)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    plt.show()
    fig.savefig('featues.svg',bbox_inches='tight',dpi=200)
#%% shows all miss-categoriced thumbs if show_mis_thumbs = True

show_mis_thumbs = False 
if show_mis_thumbs:
    transformed_test_thumbs = pipe.named_steps.nmf.transform(X_test)
    predictions_test = pipe.predict(X_test)
    features = pipe.named_steps.nmf.components_
    for i in range(len(Y_test)):
        var = i
        if Y_test[i] != predictions_test[i]:
            test = np.matmul(transformed_test_thumbs[var,:],features)
            fig, axes = plt.subplots(1,2,figsize=(15,12))
            axes[0].imshow(X_test[var,:].reshape(17,17),cmap="Greys_r",vmin=0, vmax=1)
            axes[0].set_title(str(Y_test[i]),size=30)
            axes[1].imshow(test.reshape(17,17),cmap="Greys_r",vmin=0, vmax=1)
            axes[1].set_title(str(predictions_test[i]),size=30)
            plt.suptitle("Test set Miss-classifications\n"+X_test_path[i],size=30)
            plt.show()
    
    transformed_train_thumbs = pipe.named_steps.nmf.transform(X_train)
    prediction_train = pipe.predict(X_train)
    for i in range(len(Y_train)):
        var = i
        if Y_train[i] != prediction_train[i]:
            test = np.matmul(transformed_train_thumbs[var,:],features)
            fig, axes = plt.subplots(1,2,figsize=(15,12))
            axes[0].imshow(X_train[var,:].reshape(17,17),cmap="Greys_r",vmin=0, vmax=1)
            axes[0].set_title(str(Y_train[i]),size=30)
            axes[1].imshow(test.reshape(17,17),cmap="Greys_r",vmin=0, vmax=1)
            axes[1].set_title(str(prediction_train[i]),size=30)
            plt.suptitle("Training set Miss-classifications\n"+X_train_path[i],size=30)
            plt.show()





