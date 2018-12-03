import pandas as pd
import numpy as np


def binary_based_class_split(class_names, dataframe, feature_col, target_col):
    """
	Splitting the value of certain feature based on the class (binary class)
	Args:
		class_names:		a list of clas names, ex:['macet', 'lancar']
		dataframe:		    a dataframe, contains feature and target or class columns
		feature_col:        a list of string, contains the name of the feature column
		target_col:         a string, the name of target columns (the value of this column is 0 or 1)
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


class TimeHandler():
    def __init__(self):
        self.day_dict = {'non_kabisat': {'1': 0, '2':31, '3':59, '4':90, '5':120, '6':151,\
                                         '7': 181, '8':212, '9':243, '10':273, '11':304, '12':334},
                         'kabisat': {'1': 0, '2':31, '3':60, '4':91, '5':121, '6':152, \
                                     '7': 182, '8':213, '9':244, '10':274, '11':305, '12':335}}
        
        self.indo_holiday_dict = {2016: {1: [1], 2: [8, 29], 3: [9, 25], 4: [], \
                                    5: [5], 6: [], 7: [1, 2, 3, 4, 5, 6, 7], 8: [17], \
                                    9: [12], 10: [], 11:[], 12: [12, 25, 26]},
                             2017: {1: [1, 28], 2: [], 3: [28], 4: [14, 24], \
                                    5: [1, 11, 23], 6: [1, 21, 22, 23, 24, 25, 26, 27], 7: [], 8: [17], \
                                    9: [1, 21], 10: [], 11:[], 12: [1, 25, 26]},
                             2018: {1: [1], 2: [16], 3:[18, 30], 4: [13], \
                                    5: [1, 10, 29], 6: [10, 11, 12, 13, 14, 15, 16], 7: [], 8: [17, 22], \
                                    9: [12], 10: [], 11: [20], 12: [1, 25, 26]},
                             2019: {1: [1], 2: [5], 3: [7], 4: [3, 19], \
                                    5: [1, 30, 31], 6: [1, 2, 3, 4, 5, 6], 7: [], 8: [17], \
                                    9: [12], 10: [], 11:[9], 12: [25, 26]}}
                         
        self.weekend_start = {2016:[2, 3], 2017:[1, 7], 2018:[6, 7], 2019:[5, 6]}

    def encode_oneyear_YYYYMMDD(self, pandas_column):
        pandas_column = pandas_column.str.replace("-", "")
        ## get the year data
        year = []
        month = []
        date = []
        encoded_day = []
        for idx, i in enumerate(pandas_column):
            tmp_year = int(i[:4])
            tmp_month = int(i[4:6])
            tmp_date = int(i[6:8])
            if tmp_year % 4 == 0:
                flag = 'kabisat'
            else: flag = 'non_kabisat'
            total_day = self.day_dict[flag][str(tmp_month)] + tmp_date
            #add the value
            year.append(tmp_year)
            month.append(tmp_month)
            date.append(tmp_date)
            encoded_day.append(total_day)
        return np.array(year), np.array(month), np.array(date), np.array(encoded_day)
    
    def encode_oneyear_date(self, year, pandas_month_column, pandas_date_column):
        encoded_day = []
        for j, k in zip(pandas_month_column, pandas_date_column):
            month = int(j)
            date = int(k)
            if year % 4 == 0:
                flag = 'kabisat'
            else: flag = 'non_kabisat'
            total_day = self.day_dict[flag][str(month)] + date 
            #add the value
            encoded_day.append(total_day)
        return encoded_day

    def encode_multiple_year_date(self, pandas_year_column, pandas_month_column, pandas_date_column):
        min_year = pandas_year_column.min()
        ## get the year data
        encoded_day = []
        for i, j, k in zip(pandas_year_column, pandas_month_column, pandas_date_column):
            year = int(i)
            month = int(j)
            date = int(k)
            if year % 4 == 0:
                flag = 'kabisat'
            else: flag = 'non_kabisat'
            total_day = self.day_dict[flag][str(month)] + date + 365 * (year - min_year)
            #add the value
            encoded_day.append(total_day)
        return encoded_day

    def is_holiday(self, pandas_year_column, pandas_month_column, pandas_date_column):
        holiday_list = []
        for i, j, k in zip(pandas_year_column, pandas_month_column, pandas_date_column):
            year = int(i)
            month = int(j)
            date = int(k)
            is_holiday = int(date in self.indo_holiday_dict[year][month])
            holiday_list.append(is_holiday)
        return holiday_list

    def is_weekend(self, pandas_year_column, pandas_encodedDay_oneYear_column):
        weekends = []
        for i, j in zip(pandas_year_column, pandas_encodedDay_oneYear_column):
            year = int(i)
            encoded_day = int(j)
            saturday, sunday = self.weekend_start[year]
            is_saturday = ((encoded_day - saturday)% 7 == 0)
            is_sunday = ((encoded_day - sunday)% 7 == 0)
            weekend = is_saturday or is_sunday
            weekends.append(int(weekend))
        return weekends



