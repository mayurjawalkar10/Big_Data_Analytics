"""
Author: Mayur Sunil Jawalkar (mj8628)
Big Data Analytics: Homework-03
Description: In this  assignment we are trying to predict the best feature to classify the data
             into sick or not sick categories. For this purpose we are trying to use cross
             correlation coefficient and one rule methods.
"""

# Import statement to handle command line input.
import sys
# Import pandas library for handling data.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_cross_correlation_coefficient(data):
    """
    This function returns the cross correlation coefficient for a given data.
    The input data is expected to have two attributes amongst which correlation is to be calculated.

    :param data: pandas data frame with 2 attributes.
    :return cross_corr_coef: returns the cross correlation coefficient for a given data.
    """
    attr_1, attr_2 = list(data.columns)  # Get the names of the attributes.
    attr_1_mean = data[attr_1].mean()   # Calculate the mean of attribute_1.
    attr_1_std = data[attr_1].std()   # Calculate the standard deviation of attribute_1.
    attr_2_mean = data[attr_2].mean()   # Calculate the mean of attribute_2.
    attr_2_std = data[attr_2].std()   # Calculate the standard deviation of attribute_1.

    cross_corr_coef = 0  # Initialize variable for cross correlation coefficient

    # Iterate over each record to calculate the sum for correlation.
    for index, row in data.iterrows():
        cross_corr_coef += ((row[attr_1]-attr_1_mean)/attr_1_std) * ((row[attr_2]-attr_2_mean)/attr_2_std)

    # Normalize the sum to get the cross correlation coefficient.
    cross_corr_coef /= len(data.index)

    return cross_corr_coef


def get_one_rule(data):
    """
    This function accepts the data input having 2 attributes to calculate the one rule for the same.
    It computes the mis-classifications to generate the one rule.

    :param data: pandas data frame with 2 attributes.
    :return [total_misclassification, total_misclassification/len(data.index)]:
                                list of total mis-classifications and mis-classification rate.
    :return one_rule: dictionary containing rule for decision making.
                      {key = levels of attr_1, value = levels of attr_2}
    """
    attr_1, attr_2 = list(data.columns)  # extract names of the attributes.
    attr_1_counts = data[attr_1].value_counts().to_dict()  # get the frequency of each level from attribute 1
    attr_2_counts = data[attr_2].value_counts().to_dict()  # get the frequency of each level from attribute 2

    # Generate and initialize the frequency table [row*col] = [#_attr_1_level * #_attr_2_levels]
    freq_table = pd.DataFrame(columns=sorted(attr_2_counts.keys()), index=sorted(attr_1_counts.keys()))
    freq_table = freq_table.fillna(0)  # Replace NA values with 0.

    total_misclassification = 0  # Initialize the mis-classification count to 0.

    # Iterate over each record to fill the frequency table.
    for index, row in data.iterrows():
        freq_table.loc[row[attr_1], row[attr_2]] += 1

    # print(freq_table)  # Print the frequency table for getting more insights about the data.

    one_rule = dict()  # Dictionary to store the one rule.
    # Iterate over each record in frequency table to generate the one_rule.
    for index, row in freq_table.iterrows():
        true_class = []  # list of true classes i.e., having maximum freq value.
        max_val = 0  # Maximum frequency value.
        # Iterate over each level/class of attribute 2 to check if it is a true class.
        for attr_2_level in attr_2_counts.keys():
            if max_val < row[attr_2_level]:  # Check if current class has max freq value.
                max_val = row[attr_2_level]  # Update the max value
                true_class = [attr_2_level]  # Update the true class
            elif max_val == row[attr_2_level]:  # Check if there is a tie in max value
                true_class.append(attr_2_level)  # append the class in true classes.

        # Check if there are more than one classes with same maximum value.
        if len(true_class) > 1:
            # Select the true class which has maximum frequency overall
            true_class = [level for level, count in attr_2_counts.items() if level in true_class
                          and count == max(attr_2_counts.values())]
        # Check if the above step doesn't solve the ambiguity in true classes.
        if len(true_class) > 1:
            # Based on the understanding of the data decide a default value for the data.
            # In our case by default no one is Sick hence default value is 0.
            true_class = [0]

        one_rule[index] = true_class[0]  # Save the one rule for current pair of attributes.

        # Compute the total mis-classifications.
        total_misclassification += sum([row[col] for col in attr_2_counts.keys() if col != true_class[0]])

    # print(attr_1_counts, attr_2_counts, len(attr_1_counts.keys()), len(attr_2_counts.keys()), total_misclassification,
    #       total_misclassification/len(data.index), one_rule)

    # Display the result in proper format
    print("{:12} vs {:10}: [Total Misclassifications : {}, Misclassification Rate : {:.5f}, one rule : {}]"
          .format(attr_1, attr_2, total_misclassification, total_misclassification/len(data.index), one_rule))

    # Return list of (total mis-classifications and mis-classification rate), and a one rule
    return [total_misclassification, total_misclassification/len(data.index)], one_rule


def write_trained_program(best_attribute, one_rule):
    """
    This function writes the trained program for us. The trained program is expected to perform the classification
    of validation data based on the one rule obtained from the best attribute.

    :param best_attribute: name of the best attribute.
    :param one_rule: dictionary for the one rule.
    """
    tab_ch = "    "
    with open("HW_03_Jawalkar_Mayur_Trained.py", "w+", encoding="utf-8") as train_file:
        train_file.write("import sys\n")
        train_file.write("import pandas as pd\n\n\n")
        train_file.write("def main():\n")
        train_file.write(tab_ch + "\"\"\"\n")
        train_file.write(tab_ch + "This function predicts if the person will get sick based on the given data.\n")
        train_file.write(tab_ch + "\"\"\"\n")
        train_file.write(tab_ch + "if len(sys.argv) < 2:\n")
        train_file.write(tab_ch * 2 + "print('Filename not provided. Exiting..')\n")
        train_file.write(tab_ch * 2 + "sys.exit(1)\n\n")
        train_file.write(tab_ch + "validation_filename = sys.argv[1]\n")
        train_file.write(tab_ch + "validation_data = pd.read_csv(validation_filename)\n\n")
        train_file.write(tab_ch + "for index, row in validation_data.iterrows():\n")
        count = 0
        # Iterate over each level of the target attribute for organizing the if-else statements.
        for rulevalue in set(one_rule.values()):
            count += 1  # counter to check first if condition
            keys = [key for key, value in one_rule.items() if value == rulevalue]
            if_or_elif = "" if count == 1 else "el"  # decide whether to use if or elif based on count
            train_file.write(tab_ch * 2 + if_or_elif+"if row['"+best_attribute+"'] in " + str(keys) + ":\n")
            train_file.write(tab_ch * 3 + "print("+str(rulevalue)+")\n")
        train_file.write(tab_ch * 2 + "else:\n")
        train_file.write(tab_ch * 3 + "print()\n\n\n")
        train_file.write("if __name__ == '__main__':\n")
        train_file.write(tab_ch + "\"\"\"\n")
        train_file.write(tab_ch + "Run if executed as a script.\n")
        train_file.write(tab_ch + "\"\"\"\n")
        train_file.write(tab_ch + "main()\n")


def split_training_data_on_best_attr(data, attribute_name='PeanutButter'):
    """
    This function splits the data using the given attribute.

    :param data: pandas dataframe.
    :param attribute_name: name of the attribute to use for splitting the data.
    :return peanut_butter_0, peanut_butter_1: 2 data frames after the split.
    """
    peanut_butter_0 = data[data[attribute_name] == 0]  # Extract data having PeanutButter = 0
    peanut_butter_1 = data[data[attribute_name] == 1]  # Extract data having PeanutButter = 1
    peanut_butter_0 = peanut_butter_0.drop(columns=attribute_name)  # Remove attribute PeanutButter
    peanut_butter_1 = peanut_butter_1.drop(columns=attribute_name)  # Remove attribute PeanutButter
    return peanut_butter_0, peanut_butter_1


def get_best_attr_and_rule(data):
    """
    This function accepts the input data-frame and computes the cross-correlation coefficient and one rule using
    each attribute. Based on the obtained results it decides which is the best attribute to classify the data.

    :param data: input data-frame
    :return attr_list[min_error_idx], one_rule_df[min_error_idx]: best attribute, one rule associated with best attr
    """
    # print(len(data.index))  # Total records

    # Get the list of all attributes except Sickness.
    attr_list = [attr for attr in data.columns if attr != 'Sickness']

    # Generate a data-frame for storing correlation coefficients
    cross_corr_coeff_df = pd.DataFrame(columns=['cross_corr_coeff'])
    # Generate a data-frame for storing errors in one rule.
    one_rule_error_df = pd.DataFrame(columns=['Missed Values', 'Miss-classification Rate'])
    one_rule_df = dict()  # Dictionary for storing the one rule for each attribute.

    # Iterate over each food item/ attribute.
    for attr in attr_list:
        # Compute Cross Correlation between attr and Sickness.
        cross_corr_coeff_df.loc[len(cross_corr_coeff_df)] = \
            (get_cross_correlation_coefficient(data[[attr, 'Sickness']]))
        # Compute the one rule for attr and Sickness.
        one_rule_error_df.loc[len(one_rule_error_df)], one_rule_df[len(one_rule_df)] = \
            get_one_rule(data[[attr, 'Sickness']])

    print(cross_corr_coeff_df)

    pos_best_corr_idx = cross_corr_coeff_df.idxmax().values[0]
    neg_best_corr_idx = cross_corr_coeff_df.idxmin().values[0]

    # Get the index of best correlation after considering both positive and negative values.
    if abs(cross_corr_coeff_df.at[pos_best_corr_idx, 'cross_corr_coeff']) < \
                                            abs(cross_corr_coeff_df.at[neg_best_corr_idx, 'cross_corr_coeff']):
        best_corr_idx = neg_best_corr_idx
    else:
        best_corr_idx = pos_best_corr_idx

    print("\nBest attribute using correlation : {}, Correlation Coefficient : {:.4f}".format(
        attr_list[best_corr_idx], cross_corr_coeff_df.at[best_corr_idx, 'cross_corr_coeff']))

    # Get the index of best one rule
    min_error_idx = one_rule_error_df.idxmin().values[1]
    print("Best attribute using One Rule : {}, Mis-classification Error : {:.4f}".format(
        attr_list[min_error_idx], one_rule_error_df.loc[min_error_idx, 'Miss-classification Rate']))
    one_rule_str = ""
    for att_1_level, att_2_level in one_rule_df[min_error_idx].items():
        one_rule_str += "if " + str(attr_list[min_error_idx]) + " == " + str(att_1_level) + \
                        " then Sickness = " + str(att_2_level) + "\n"
    # print(one_rule_df[min_error_idx], end="")
    print(one_rule_str, end="")
    return attr_list[min_error_idx], one_rule_df[min_error_idx]


def main():
    """
    This function drives the functionality of the program. It accepts the training file as a command line argument.
    Using the input file it makes call to respective function for computing the cross-correlation-coefficient and
    one rule, and to decide the best attribute to classify the data. It also gives a call to a function for writing
    the trained classification program.
    Finally this function tries to figure out the second best feature to classify the data.
    """

    # Check for valid command line arguments
    if len(sys.argv) < 2:
        print("No filename Provided. \nExiting..")
        sys.exit(1)

    filename = sys.argv[1]  # Accept the filename from command line
    input_data = pd.read_csv(filename)  # Read the csv file into a pandas data-frame
    training_data = input_data.drop(columns=['GuestID'])  # Remove the ID from the data.

    # Get the best attribute and one rule on complete training data.
    best_attribute, one_rule = get_best_attr_and_rule(training_data)

    # Write the trained program using above results.
    write_trained_program(best_attribute, one_rule)

    print("\nIdentify the second best attribute for classifying the data.")
    # Split the data using attribute 'PeanutButter'.
    data0, data1 = split_training_data_on_best_attr(training_data)

    print("\n------------------------- Split 1 -------------------------------")
    print(" PeanutButter = 0")
    # Get the best attribute and one rule on data0.
    best_attribute, one_rule = get_best_attr_and_rule(data0)

    print("\n------------------------- Split 2 -------------------------------")
    print(" PeanutButter = 1")
    # Get the best attribute and one rule on data1.
    best_attribute, one_rule = get_best_attr_and_rule(data1)


if __name__ == '__main__':
    main()
