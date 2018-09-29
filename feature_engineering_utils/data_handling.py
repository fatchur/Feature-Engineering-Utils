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
    
    dict = {}
    for i in class_names:
        dict[i] = []

    for index, row in dataframe.iterrows():
        val = row[target_col]
        if val == 1.0:
            dict[class_names[0]].append(row[feature_col])
        else:
            dict[class_names[1]].append(row[feature_col])
            
    print len(dict[class_names[0]]), len(dict[class_names[1]])

    df = pd.DataFrame({class_names[0]: dict[class_names[0]], class_names[1]: dict[class_names[1]]})
    return df
