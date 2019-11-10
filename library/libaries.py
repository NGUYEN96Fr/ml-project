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
