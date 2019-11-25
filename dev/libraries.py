import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
from sklearn.linear_model import Ridge,LinearRegression,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

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
    samples = pd.DataFrame()
    target = pd.DataFrame()

    if file == 'HousingData.csv':
        data = pd.read_csv('../data/HousingData.csv')
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
        data = pd.read_csv('../data/prostate.data', sep = '\t')
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
    # Deep copy the  and targets
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
    - samples: array
        Preprocessed samples. # fill missing values, encode categorical types, standarlization

    - target: array
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
    X = samples
    y = target

    #
    # train, test split.
    #


    X_train, X_test, Y_train, Y_test = train_test_split( X, y, test_size = test_size, shuffle = shuffle)

    # train set
    train_set = {}
    train_set['X_train'] = X_train
    train_set['y_train'] = Y_train

    # test set
    test_set = {}
    test_set['X_test'] = X_test
    test_set['y_test'] = Y_test

    #
    # K-fold cross-validation.
    # https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833
    #
    cv = KFold(n_splits = k, random_state = 42, shuffle = shuffle)

    train_crs_val = []
    test_crs_val =[]

    for train_index, test_index in cv.split(X):
        x_train, x_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        # add a fold to the lists
        train_crs_val.append((x_train,y_train))
        test_crs_val.append((x_test,y_test))

    return train_set, test_set, train_crs_val, test_crs_val


def Dimension_Reduction(samples, target, variance = 0.9, nb_max = 13, to_plot = False,):
    """
    This function does the dimension reduction on the samples
    Author: NGUYEN Van-Khoa

    Parameters
    ----------
    - samples: DataFrame
        (nb_samples,features)
    - target: DataFrame
        Preprocessed target.
    - variances: float (default = 0.90)
        The percentage of variances to keep
    - nb_max: int (default = 13)
        max number of components considered to plot
    - to_plot: boolean
        plot the analysis

    Returns
    -------
    - X_new: array
        the new X with reduced dimensions
    - y: array
        the target
    """
    # number of observations
    X = samples.values
    y = target.values

    n = X.shape[0]

    # instanciation
    acp = PCA(svd_solver='full')
    X_transform = acp.fit_transform(X)
    print("Number of acp components features= ", acp.n_components_)
    #variance explained
    eigval = acp.explained_variance_

    # variance of each component
    variances = acp.explained_variance_ratio_

    # percentage of variance explained
    cumsum_var_explained= np.cumsum(variances)
    print("cumsum variance explained= ",cumsum_var_explained[0:nb_max-1])

    #get the number of components satisfying the establised variance condition
    nb_component_features = np.where(cumsum_var_explained>variance)[0][0]
    print('nb_component_features: ', nb_component_features)
    acp_features = PCA(svd_solver='full',n_components =nb_component_features+1)
    X_new = acp_features.fit_transform(X)

    if to_plot:

        plt.figure(figsize=(10,5))
        plt.plot(np.arange(1,nb_max),variances[0:nb_max-1])
        plt.scatter(np.arange(1,nb_max),variances[0:nb_max-1])
        plt.title("Variance explained by each component")
        plt.ylabel("Variance values")
        plt.xlabel("Component")

        #scree plot
        plt.figure(figsize=(15,10))

        plt.subplot(221)
        plt.plot(np.arange(1,nb_max),eigval[0:nb_max-1])
        plt.scatter(np.arange(1,nb_max),eigval[0:nb_max-1])
        plt.title("Scree plot")
        plt.ylabel("Eigen values")
        plt.xlabel("Factor number")

        plt.subplot(222)
        plt.plot(np.arange(1,nb_max),cumsum_var_explained[0:nb_max-1])
        plt.scatter(np.arange(1,nb_max),cumsum_var_explained[0:nb_max-1])
        plt.title("Total Variance explained")
        plt.ylabel("Variance values")
        plt.xlabel("Factor number")

    return X_new,y


def decision_tree(criterion='mse', splitter='best', max_depth=None, min_samples_split=2,
min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False):
    """
    Regression with Decision Tree
    Author: NGUYEN Van-Khoa

    """
    tree = DecisionTreeRegressor(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, random_state=random_state,
    max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, presort=presort)
    return tree
    
def random_forest(n_estimators,max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    Regression with Decision Tree
    Author: YU Boyang
    """
    forest = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    return forest    


def lr_model():
    """
    This function trains a Linear Regression model form the training data
    Author: LIU Xi
    """
    line = LinearRegression()
    return line


def ridge_model(alpha=0.1):
    """
    This function trains a Ridge Regression model form the training data
    Author: LIU Xi
    """
    ridge = Ridge(alpha)
    return ridge


def lasso(alpha=0.0001):
    """
    This function trains a lasso model form the training data
    Author: LIU Xi
    """
    lasso = Lasso(alpha)
    return lasso


def knn(n_neighbors):
    """
    This function trains a KNN model form the training data
    Author: ZAN Lei
    
    Input: n_neighbors: int 
    
    """
    knn = KNeighborsRegressor(n_neighbors=int(n_neighbors))
    return knn

def svr_linear_model():
    """
    This function trains a SVR Linear model form the training data
    Author: LIU Xi
    """
    svr_linear = SVR(kernel='linear')
    return svr_linear

def svr_rbf_model():
    """
    This function trains a SVR RBF model form the training data
    Author: LIU Xi
    """
    svr_rbf = SVR(kernel='rbf')
    return svr_rbf


def feature_correlation(namedataset):
    """show the correlationship between different features of the dataset
        Author: Lei ZAN
    Parameters:

    namedataset:{'HousingData.csv', 'prostate.data'}

    Return:

    correlation map of different features

    """
    if namedataset== 'HousingData.csv':
        column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        #data_1 = pd.read_csv('../data/HousingData.csv', header=None, delimiter=r"\s+",names=column_names)
        data_1 = pd.read_csv('../data/HousingData.csv')
        data_1 = data_1[~(data_1['MEDV'] >= 50.0)]
        plt.figure(figsize=(20, 10))
        plt.title("The correlation between features of dataset HousingData")
        sns.heatmap(data_1.corr().abs(),  annot=True)
    elif namedataset== 'prostate.data':
        column_names = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45', 'lpsa', 'train/test indicator']
        data = pd.read_csv('../data/prostate.data', index_col=0, header=0, delimiter=r"\s+", names=column_names)
        column_sels = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45', 'lpsa']
        data = data.loc[:,column_sels]
        plt.figure(figsize=(20, 10))
        plt.title('The correlation between features of dataset prostate')
        sns.heatmap(data.corr().abs(),  annot=True)

def feature_distribution(namedataset):
    """show the distribution of different features of the dataset
        Author: Lei ZAN
    Parameters:

    namedataset:{'HousingData.csv', 'prostate.data'}

    Return:

    distribution map of different features

    """
    if namedataset== 'HousingData.csv':
        column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        #data_1 = pd.read_csv('../data/HousingData.csv', header=None, delimiter=r"\s+",names=column_names)
        data_1 = pd.read_csv('../data/HousingData.csv')
        data_1 = data_1[~(data_1['MEDV'] >= 50.0)]
        data_1 = data_1.apply(lambda x: x.fillna(x.value_counts().index[0]))
        #Voir les distributions de chaque features
        fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
        index = 0
        axs = axs.flatten()
        for k,v in data_1.items():
            sns.distplot(v, ax=axs[index])
            index += 1
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    elif namedataset== 'prostate.data':
        column_names = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45', 'lpsa', 'train/test indicator']
        data = pd.read_csv('../data/prostate.data', index_col=0, header=0, delimiter=r"\s+", names=column_names)
        column_sels = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45', 'lpsa']
        data = data.loc[:,column_sels]
        fig, axs = plt.subplots(ncols=9, nrows=1, figsize=(20, 10))
        index = 0
        axs = axs.flatten()
        for k,v in data.items():
            sns.distplot(v, ax=axs[index])
            index += 1
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

def dessin_heatmap(data):
    """show the correlation of different features chosen of the dataset
        Author: Lei ZAN
    Parameters:

    data: data of type dataframe

    Return:

    correlatons map of different features chosen

    """
    plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr().abs(),  annot=True)

    
def dessin_distribution(data, column_sels, rows, cols):
    """show the distribution of different features chosen of the dataset
        Author: Lei ZAN
    Parameters:

    data: dataframe like data
    column_sels: list of columns chosen 

    Return:

    distributions map of different features chosen

    """
    min_max_scaler = preprocessing.MinMaxScaler()
    #column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
    x = data.loc[:,column_sels]
    x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(20, 10))
    index = 0
    axs = axs.flatten()
    for col in x.columns:
        sns.distplot(x[col], ax=axs[index])
        index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

def tunning_knn(X_train, y_train, X_test, y_test):
    """find the best parameter n_neighbors for the model  
        Author: Lei ZAN
    Parameters:

    X_train: type array like
    y_train: type array like 
    X_test: type array like 
    y_test: type array like 

    Return:

    The best n_neighbours for this model 

    """
    start = 3
    stop = 50
    res = []
    best_n_neighbors = 0
    best_score = 0 
    for i in range(start, stop):
        model = knn(i)
        model.fit(X_train,y_train)
        score = model.score(X_test, y_test)
        res.append(score)
        if score > best_score:
            best_score = score
            best_n_neighbors = i
    plt.plot(range(start, stop), res)
    print("The best r2 score is ", best_score)
    print('The best n_neighbors is ', best_n_neighbors)
    return best_n_neighbors
    
    
def eval_lr(model,train_set,test_set,train_crs_val,test_crs_val):
    """
    Evaluate the performance a model for regression problem on:
    - train set, train cross validation set
    - test set, test cross validation set

    Co-Authors: YU Boyang, NGUYEN Van-Khoa

    This function uses two metrics: root mean square error and R2 score
    to show the performance of model on both train data and test data,
    rmse closer to 0 and R2 closer to 1, better the model.

    Parameters:
    ----------
    - model: A model of regression problem.
    - train_set: a dict of array with keywords 'X_train', 'y_train'
    - test_set: a dict of array with keywords 'X_test', 'y_test'
    - train_crs_val: list of tuple of arrays [([x_train_fold_1,y_train_fold_1]),...,([x_train_fold_n,y_train_fold_n])]
        Traing sets for implementing the cross-validation.
    - test_crs_val:  list of tuple of arrays [([x_test_fold_1,y_test_fold_1]),...,([x_test_fold_n,y_test_fold_n])]
        Test sets for implementing the cross-validation.
    Returns:
    --------
    null
      """
    X_train = train_set['X_train']
    y_train =  train_set['y_train']
    X_test =  test_set['X_test']
    y_test = test_set['y_test']
    # model evaluation for training set
    model.fit(X_train,y_train)
    y_train_predict = model.predict(X_train)
    rmse_train = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
    # r2 represents the area between our model and the mean model/ area between best model and the mean model, so 1 is the best
    # https://ragrawal.wordpress.com/2017/05/06/intuition-behind-r2-and-other-regression-evaluation-metrics/
    r2_train = r2_score(y_train, y_train_predict)
    print("----------------------------------------")
    print("|The model performance for training set|")
    print("---------------------------------------")
    print('RMSE is: {:0.4f}'.format(rmse_train))
    print('R2 score is: {:0.4f}'.format(r2_train))

    # model evaluation for testing set
    y_test_predict = model.predict(X_test)
    rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
    r2_test = r2_score(y_test, y_test_predict)

    print("------------------------------------")
    print("|The model performance for test set|")
    print("------------------------------------")
    print('RMSE is: {:0.4f}'.format(rmse_test))
    print('R2 score is: {:0.4f}'.format(r2_test))

    # number of folds
    k = len(train_crs_val)
    rmse_train_cros_val = []
    rmse_train_cros_val_total = 0
    rmse_test_cros_val = []
    rmse_test_cros_val_total = 0
    r2_train_cros_val = []
    r2_train_cros_val_total = 0
    r2_test_cros_val = []
    r2_test_cros_val_total = 0

    # Cross-validation
    for fold in range(k):
        model.fit(train_crs_val[fold][0],train_crs_val[fold][1])
        # for training fold
        y_train_predict = model.predict(train_crs_val[fold][0])
        #float("{0:.2f}".format(r2_score(train_crs_val[fold][1], y_train_predict)))
        
        rmse_train_cros_val.append(float("{0:.4f}".format(np.sqrt(mean_squared_error(train_crs_val[fold][1], y_train_predict)))))
        rmse_train_cros_val_total += np.sqrt(mean_squared_error(train_crs_val[fold][1], y_train_predict))
        r2_train_cros_val.append(float("{0:.4f}".format(r2_score(train_crs_val[fold][1], y_train_predict))))
        r2_train_cros_val_total += r2_score(train_crs_val[fold][1], y_train_predict)
        # for testing fold
        y_test_predict = model.predict(test_crs_val[fold][0])
        rmse_test_cros_val.append(float("{0:.4f}".format(np.sqrt(mean_squared_error(test_crs_val[fold][1], y_test_predict)))))
        rmse_test_cros_val_total += np.sqrt(mean_squared_error(test_crs_val[fold][1], y_test_predict))
        r2_test_cros_val.append(float("{0:.4f}".format(r2_score(test_crs_val[fold][1], y_test_predict))))
        r2_test_cros_val_total += r2_score(test_crs_val[fold][1], y_test_predict)
    print("----------------------------------------------------------------")
    print("|The cross validation of the model performance for training set|")
    print("----------------------------------------------------------------")
    print('Cross-validation RMSE is \n {}'.format(rmse_train_cros_val))
    print('Cross-validation R2 score is \n {}'.format(r2_train_cros_val))
    print('Average RMSE is: {:0.4f}'.format(rmse_train_cros_val_total/k))
    print('Average R2 score is: {:0.4f}'.format(r2_train_cros_val_total/k))

    print("------------------------------------------------------------")
    print("|The cross validation of the model performance for test set|")
    print("------------------------------------------------------------")
    print('Cross-validation RMSE is \n {}'.format(rmse_test_cros_val))
    print('Cross-validation R2 score is \n {}'.format(r2_test_cros_val))
    print('Average RMSE is: {:0.4f}'.format(rmse_test_cros_val_total/k))
    print('Average R2 score is: {:0.4f}'.format(r2_test_cros_val_total/k))
