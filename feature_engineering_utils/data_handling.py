import pandas as pd
import numpy as np


def binary_based_class_split(class_names, dataframe, feature_col, target_col):
    """
	Splitting the value of certain feature based on the class (binary class)
	Args:
		class_names:		a list of clas names, ex:['macet', 'lancar']
		dataframe:		    The dataframe, contains feature and target or class columns
		feature_col:        The name of the feature column, a string
		target_col:         The name of target columns, a string
	Return:
		A dataframe with the splitted data based on the class value, the class names are used as feature names
	"""
    
    # create a dictionary of class names
    dict = {}
    for i in class_names:
        dict[i] = []

    # get the value of target columns
    # append it to the corresponding class name dictionary
    for index, row in dataframe.iterrows():
        val = row[target_col]
        if val == 1.0:
            dict[class_names[0]].append(row[feature_col])
        else:
            dict[class_names[1]].append(row[feature_col])
            
    print (len(dict[class_names[0]]), len(dict[class_names[1]]))

    # return a dictionary, with class names as column names
    df = pd.DataFrame({class_names[0]: dict[class_names[0]], class_names[1]: dict[class_names[1]]})
    return df



def agg_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.

    source : https://towardsdatascience.com/machine-learning-kaggle-competition-part-two-improving-e5b4d61ab4b8
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    
    # First calculate counts
    counts = pd.DataFrame(df.groupby(group_var, as_index = False)[df.columns[1]].count()).rename(columns = {df.columns[1]: '%s_counts' % df_name})
    
    # Group by the specified variable and calculate the statistics
    agg = df.groupby(group_var).agg(['mean', 'max', 'min', 'sum']).reset_index()
    
    # Need to create new column names
    columns = [group_var]
    
    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
              
    #  Rename the columns
    agg.columns = columns
    
    # Merge with the counts
    agg = agg.merge(counts, on = group_var, how = 'left')
    
    return agg


def identify_zero_importance_features(train, train_labels, iterations = 2):
    import lightgbm as lgb
    """
    Identify zero importance features in a training dataset based on the 
    feature importances from a gradient boosting model. 

    source : https://towardsdatascience.com/machine-learning-kaggle-competition-part-two-improving-e5b4d61ab4b8
    
    Parameters
    --------
    train : dataframe
        Training features
        
    train_labels : np.array
        Labels for training data
        
    iterations : integer, default = 2
        Number of cross validation splits to use for determining feature importances
    """
    
    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(train.shape[1])

    # Create the model with several hyperparameters
    model = lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', 
                               n_estimators = 10000, class_weight = 'balanced')
    
    # Fit the model multiple times to avoid overfitting
    for i in range(iterations):

        # Split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(train, train_labels, 
                                                                            test_size = 0.25, 
                                                                            random_state = i)

        # Train using early stopping
        model.fit(train_features, train_y, early_stopping_rounds=100, 
                  eval_set = [(valid_features, valid_y)], 
                  eval_metric = 'auc', verbose = 200)

        # Record the feature importances
        feature_importances += model.feature_importances_ / iterations
    
    feature_importances = pd.DataFrame({'feature': list(train.columns), 
                            'importance': feature_importances}).sort_values('importance', 
                                                                            ascending = False)
    
    # Find the features with zero importance
    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    print('\nThere are %d features with 0.0 importance' % len(zero_features))
    
    return zero_features, feature_importances



def feature_selection(feature, target, iterations, code):
    import xgboost
    from sklearn.model_selection import train_test_split
    
    importances = dict.fromkeys(feature.keys(), 1)
    importances =dict.fromkeys(importances, 0)
    
    if code == "regression":
        xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
        
    for i in range (iterations):
        train_features, valid_features, train_y, valid_y = train_test_split(feature, 
                                                                            target, test_size = 0.25, 
                                                                            random_state = i)
        xgb.fit(train_features, train_y)
        fscore = xgb.feature_importances_
        
        for idx, i in enumerate(importances):
            importances[i] += fscore[idx]
            
    return importances
