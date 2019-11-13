import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler

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
    #r2 represents the area between our model and the mean model/ area between best model and the mean model, so 1 is the best 
    #https://ragrawal.wordpress.com/2017/05/06/intuition-behind-r2-and-other-regression-evaluation-metrics/
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

    
def data_pre(samples,target):  
    """ 
        Prepprocessing samples for preparing the training et testing steps.       
        Author: NGUYEN Van-Khoa
        
        The function handles some problems existing in the original data such as missing values,
        categorial types, standarlization.
        
        Args:
        - samples (DataFrame object)
        - target (DataFrame object)
        
        Output:
        - pre_samples (DataFrame object): preprocessed samples.
        - pre_target (DataFrame object): preprocessed target.
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
        