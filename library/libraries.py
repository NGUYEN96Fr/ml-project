import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold


def data_import(file,features = None ,target_label = None):
    """ Import data. Extract features, target_label's values. Create samples, target.
    
    Author: NGUYEN Van-Khoa
    
    The function selects data's attributes for creating samples and target.
    
    Parameters
    ----------
    - file: string
        file name of the data
   
    - features: list of strings (default = None)
        + the name of features that we want to include in the sample
        + use default defined features if the features aren't given.
    
    - target_label: string (default = None)
        + the target label for creating the target
        + use the default defined target if the target isn't given. 
        
      Returns:
    ----------
    - samples: DataFrame object
        
    - target: DataFrame object 
        
    """
    
    if file == 'HousingData.csv':
        data = pd.read_csv(file)
        
        if (features != None) and (data != None):
            
            samples = data[features]
            target = data[target_label].to_frame()
            
        else:
            defined_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']
            defined_target_label = ['MEDV']
            
            samples = data[defined_features]
            target = data[defined_target_label]
        
    elif  file == 'prostate.data':
        data = pd.read_csv(file, sep = '\t')
        
        if (features != None) and (data != None):
            
            samples = data[features]
            target = data[target_label].to_frame()
            
        else:
            defined_features = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp',
       'gleason', 'pgg45']
            defined_target_label = ['lpsa']
            
            samples = data[defined_features]
            target = data[defined_target_label]
            
    else:
        print('File name Error')
     
    
    return samples, target
    
    
def data_cl(samples,target):  
    """ Prepprocessing data (including the samples and the target).       
    
    Author: NGUYEN Van-Khoa
        
    The function handles some problems existing in the original data such as missing values,
    categorial types, standarlization.
        
    Parameters
    ----------
    - samples: DataFrame object
    
    - target: DataFrame object


    Returns:
    ----------
    - pre_samples: DataFrame object
        Preprocessed samples.
        
    - pre_target: DataFrame object 
        Preprocessed target.
    
    """
    #
    # Deep copy the samples and targets
    #
    samples_copy = copy.deepcopy(samples)
    target_copy = copy.deepcopy(target)
    
    #
    # Check and replace the missing values by the most frequent values.
    #
    filled_samples = samples_copy.apply(lambda x: x.fillna(x.value_counts().index[0]))
    
    #
    # Encode the categorical values for both samples, targets 
    # (https://pbpython.com/categorical-encoding.html)
    #
    # Retrieve the object type variables
    obj_var_samples = filled_samples.select_dtypes(include=['object'])
    for var in obj_var_samples:
        filled_samples[var] = filled_samples[var].astype('category').cat.codes
    
    obj_var_target = target_copy.select_dtypes(include=['object'])
    for var in obj_var_target:
        target_copy[var] = target_copy[var].astype('category').cat.codes
    
    #
    # Normalize the sample values
    #
    scaler = StandardScaler()
    scaled_sample_array = scaler.fit_transform(filled_samples)
    sample_variable_names = list(filled_samples.columns.values)
    scaled_samples = pd.DataFrame(scaled_sample_array)
    scaled_samples.columns = sample_variable_names
    
    #
    # Return Data Frame objects
    #
    pre_samples = scaled_samples
    pre_target = target_copy
    
    return pre_samples,pre_target


def data_split(samples,target,test_size = 0.2, k = 10, shuffle = True):
    """Dividing data into traing set and test set. Preparing the training set for the cross-validation.
    
    Author: NGUYEN Van-Khoa
    
    Parameters
    ----------
    - samples: DataFrame object
        Preprocessed samples. # fill missing values, encode categorical types, standarlization
        
    - target: DataFrame object  
        Preprocessed target.
        
    - k: int (default = 10)
        Number of folds used in the cross-validation.
        
    - test_size: float (default = 0.2)
        The percentage of data kept for the test.
        
    - shuffle: boolean (default = True)
    
    Returns:
    ----------
    - train_set: a dict of array with keywords 'X_train', 'y_train'
    
    - test_set: a dict of array with keywords 'X_test', 'y_test'
    
    - train_crs_val: list of tuple of arrays [([x_train_fold_1,y_train_fold_1]),...,([x_train_fold_n,y_train_fold_n])]
        Traing sets for implementing the cross-validation.
        
    - test_crs_val:  list of tuple of arrays [([x_test_fold_1,y_test_fold_1]),...,([x_test_fold_n,y_test_fold_n])]
        Test sets for implementing the cross-validation. 
           
    """
    #
    # convert samples, target to numpy array.
    #
    X = samples.values
    y = target.values
    
    #
    # train, test split.
    #
    X_train, X_test, Y_train, Y_test = train_test_split( X, y, test_size = test_size, shuffle = shuffle)    
    
    # train set
    train_set = {}
    train_set['feature_train'] = X_train
    train_set['target_train'] = Y_train    
    
    # test set
    test_set = {}
    test_set['feature_test'] = X_test
    test_set['target_test'] = Y_test
    
    #
    # K-fold cross-validation.
    # https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833
    #
    cv = KFold(n_splits = k, random_state = 42, shuffle = False)
    
    train_crs_val = []
    test_crs_val =[]
    
    for train_index, test_index in cv.split(X):
        x_train, x_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        # add a fold to the lists
        train_crs_val.append((x_train,y_train))
        test_crs_val.append((x_test,y_test))
    
    return train_set, test_set, train_crs_val, test_crs_val


def lr_model(X_train,y_train):
    """
    This function trains a Linear Regression model form the training data

    :param X_train:
    :param y_train:
    :return: trained Linear Regression model
    """
    from sklearn.linear_model import LinearRegression
    line = LinearRegression()
    line.fit(X_train, y_train)
    return line

def ridge_model(X_train,y_train):
    """
    This function trains a Ridge Regression model form the training data
    :param X_train:
    :param y_train:
    :return:  trained Ridge Regression model
    """
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train, y_train)
    return ridge


def svr_linear_model(X_train,y_train):
    """
    This function trains a SVR Linear model form the training data
    :param X_train:
    :param y_train:
    :return:  trained SVR Linear model
    """
    from sklearn.svm import SVR
    svr_linear = SVR(kernel='linear')
    svr_linear.fit(X_train, y_train)
    return svr_linear


def svr_rbf_model(X_train,y_train):
    """
    This function trains a SVR RBF model form the training data
    :param X_train:
    :param y_train:
    :return:  trained SVR RBF model
    """
    from sklearn.svm import SVR
    svr_rbf = SVR(kernel='rbf')
    svr_rbf.fit(X_train, y_train)
    return svr_rbf


def eval_lr(model,X_train,Y_train,X_test,Y_test):
    """evaluate the performance of a model for regression problem.
         Author: YU Boyang

        This function uses two metrics: root mean square error and R2 score
        to show the performance of model on both train data and test data,
        rmse closer to 0 and R2 closer to 1, better the model.

        Args:
            model: An trained model of regression problem.
            X_train,Y_train: train data and labels
            X_test, Y_test:  test data and labels
        Returns:null

      """

    # model evaluation for training set
    y_train_predict = model.predict(X_train)
    rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
    # r2 represents the area between our model and the mean model/ area between best model and the mean model, so 1 is the best
    # https://ragrawal.wordpress.com/2017/05/06/intuition-behind-r2-and-other-regression-evaluation-metrics/
    r2 = r2_score(Y_train, y_train_predict)

    print("The model performance for training set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")

    # model evaluation for testing set
    y_test_predict = model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    r2 = r2_score(Y_test, y_test_predict)

    print("The model performance for testing set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    return None
