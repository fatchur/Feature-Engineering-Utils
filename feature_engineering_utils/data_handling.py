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


def get_feture_importance(feature, target, code):
    """
	A function to calculate the feature importance
	Args:
		Feature:		The pandas dataframe with columns as feature
		Target:		    The pandas dataframe with single column as a target
		The code:       The string of code ('regresion' or 'classification')     
	Return:
		A plot of feature importance
	"""
    import xgboost
    from sklearn.model_selection import train_test_split
    from xgboost import plot_importance,XGBClassifier
    
    if code == "regression":
        xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

    train_features, valid_features, train_y, valid_y = train_test_split(feature, 
                                                                        target, test_size = 0.25, 
                                                                        random_state = 2)
    xgb.fit(train_features, train_y)
    fscore = xgb.feature_importances_
    plot = plot_importance(xgb)
    
    return plot


def print_dataframe_full(x):
    """
	Print all the pandas dataframe value
	Args:
		x:		The dataframe
	"""
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')



