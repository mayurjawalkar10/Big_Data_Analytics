"""
Author: Mayur Sunil Jawalkar (mj8628)
Big Data Analytics: Homework-02
Description: Find the best threshold to classify the sub-species of abominable snow folks according to region
            (Assam or Bhutan). Plot the cost function curves and ROC curves for visualization.
             Write a classifier program to classify unseen data.
"""

import sys
import math
import matplotlib.pyplot as plt
import numpy as np


def read_file(filename, delimit=","):
    """
    This function reads the file for the data input.
    It returns the list of data points.

    :param filename: Name of the file to read data from
    :param delimit: (Optional) Delimiter character to separate the column values. Default it is set to ",".
    :return data: list of all data points
    """
    # Open the file in read mode
    with open(filename, "r") as file:
        data = []
        # Read all rows of the file and iterate from the second row.
        # Ignore the first row containing header.
        for row in file.readlines()[1:]:
            # If the row is not empty add data point in the list.
            cols = row.strip().split(delimit)
            if len(cols) > 1:
                all_col_records = []
                for record in cols:
                    all_col_records.append(record)
                data.append(all_col_records)
            if len(cols) == 1:
                data.append(cols[0])
        # Return the list containing all data points.
        return data


def get_quantized_data(data, bin_size, bin_method='round'):
    """
    This method quantize each value in the data using a given bin size and binning method.

    :param data: list of data points to be quantized
    :param bin_size: int value for specifying the size of each bin
    :param bin_method: {"floor" or "ceil" or "round"}. Default binning method is "round".
    :return quantized_data: list of quantized data points.
    """
    quantized_data = []
    for row in data:
        if bin_method == "floor":
            quantized_data.append(math.floor(row / bin_size) * bin_size)
        elif bin_method == "ceil":
            quantized_data.append(math.ceil(row / bin_size) * bin_size)
        elif bin_method == "round":
            quantized_data.append(round(row / bin_size) * bin_size)
    return quantized_data


def extract_age_height_class(input_data):
    """
    Given the abominable input data, this function splits the data in age, height and class.

    :param input_data: abominable data where each record contains [age, height, class]
    :return age_data: list of ages
    :return height_data: list of heights
    :return class_data: lists of classes
    """
    age_data = []
    height_data = []
    class_data = []
    # Iterate over each record and split the record in age, height and class.
    for record in input_data:
        age_data.append(float(record[0]))
        height_data.append(float(record[1]))
        class_data.append(record[2])
    return age_data, height_data, class_data


def get_left_right_class_counts(data, class_label, threshold):
    """
    This function iterates over the input data and splits the data at a given threshold.

    It calculates the total number of records belonging to each class in each of the left
    and right region.

    :param data: Input data to split based on a given threshold.
    :param class_label: list of class labels associated with the data input.
    :param threshold: value based on which data is to split.
    :return l_assam: total data points on the left of the threshold having class label assam.
    :return l_bhutan: total data points on the left of the threshold having class label bhutan.
    :return r_assam: total data points on the right of the threshold having class label assam.
    :return r_bhutan: total data points on the right of the threshold having class label bhutan.
    """
    l_assam = l_bhutan = r_assam = r_bhutan = 0
    for rec_ind in range(len(data)):
        if data[rec_ind] <= threshold and class_label[rec_ind] == "+1":
            l_bhutan += 1
        elif data[rec_ind] <= threshold and class_label[rec_ind] == "-1":
            l_assam += 1
        elif data[rec_ind] > threshold and class_label[rec_ind] == "+1":
            r_bhutan += 1
        elif data[rec_ind] > threshold and class_label[rec_ind] == "-1":
            r_assam += 1
    return l_assam, l_bhutan, r_assam, r_bhutan


def get_classification_model(data, class_label, target_var, side, bin_size):
    """
    This function computes the best threshold value to classify the given data by minimizing the cost value.
    In this scenario, the cost value is nothing but the addition of all error values.

    Here we are passing target variable and side as arguments to decide which will our positive class
    and on which side of the threshold that class will be present.
    This information is passed using the domain knowledge.

    :param data: list of data elements to find the best classification threshold.
    :param class_label: class labels associated with data.
    :param target_var: Target class.
    :param side: side on which maximum distribution of the target variable is present (left/right)
    :param bin_size: int value indicating the size of each bin of the quantized data.
    :return best_threshold: best threshold value to classify the abominable data
    :return all_costs: list of all cost values associated with each threshold.
    :return all_thresholds: list of all threshold values
    :return true_positive_rate: list of true positive rates associated with each threshold value.
    :return false_positive_rate: list of false positive rates associated with each threshold value.
    :return best_idx: index of the best threshold value.
    """
    best_threshold = 0
    best_idx = 0
    all_costs = []
    best_cost = float("inf")
    false_positive_rate = []
    true_positive_rate = []
    all_thresholds = [threshold for threshold in range(min(data), max(data) + 1, bin_size)]

    # Iterate over each threshold level
    for threshold in all_thresholds:
        # Get the total records of each class on each side
        l_assam, l_bhutan, r_assam, r_bhutan = get_left_right_class_counts(data, class_label, threshold)

        # check on which side the maximum distribution of target variable is present.
        if side == "right":
            # decide the positive class depending on the target variable.
            if target_var == "+1":
                false_positive, false_negative, true_positive, true_negative = r_assam, l_bhutan, r_bhutan, l_assam
            else:
                false_positive, false_negative, true_negative, true_positive = l_bhutan, r_assam, r_bhutan, l_assam
        elif side == "left":
            # decide the positive class depending on the target variable.
            if target_var == "+1":
                false_positive, false_negative, true_positive, true_negative = l_assam, r_bhutan, l_bhutan, r_assam
            else:
                false_positive, false_negative, true_negative, true_positive = r_bhutan, l_assam, l_bhutan, r_assam

        cost = false_positive + false_negative

        # Compute the false positive rate
        try:
            false_positive_rate.append(false_positive/(false_positive+true_negative))
        except ZeroDivisionError:
            false_positive_rate.append(0)

        # Compute the true positive rate
        try:
            true_positive_rate.append(true_positive/(true_positive+false_negative))
        except ZeroDivisionError:
            true_positive_rate.append(0)

        # append and save the cost for current threshold value in all_costs.
        all_costs.append(cost)

        # check for the minimum cost value and update the best_cost value accordingly
        if cost <= best_cost:
            best_cost = cost
            best_threshold = threshold
            best_idx = (best_threshold - min(data)) // bin_size

    return best_threshold, all_costs, all_thresholds, true_positive_rate, false_positive_rate, best_idx


def plot_cost_function(best_threshold, all_costs, all_thresholds, best_idx, attribute):
    """
    This function plots the cost function curve for the given attribute information.

    :param best_threshold: best threshold value to classify the abominable data.
    :param all_costs: list of all cost values associated with each threshold.
    :param all_thresholds: list of all threshold values.
    :param best_idx: index of the best threshold value.
    :param attribute: name of the attribute we are using.
    """
    plt.plot(all_thresholds, all_costs, '-gs', markersize=5)
    plt.plot(best_threshold, all_costs[best_idx], 'ro', markersize=10)
    plt.title("Cost Function using attribute "+attribute)
    plt.xlabel("Threshold")
    plt.ylabel("Cost")
    plt.legend(['Cost Function', 'Best Threshold'], loc=5)
    plt.show()


def write_trained_classifier(attribute, best_threshold, side):
    """
    This function writes the trained program. The trained program is then used to
    classify the new data points into respective classes.

    :param attribute: Name of the attribute we are using to classify
    :param best_threshold: Best threshold value to use to split the data.
    :param side: side on which maximum distribution of target variable is present
    """
    tab_ch = "    "
    attribute_id = 0 if attribute == "age" else 1
    with open("HW02_Jawalkar_Mayur_Trained.py", "w+", encoding="utf-8") as train_file:
        train_file.write("import sys\n\n\n")
        train_file.write("def main():\n")
        train_file.write(tab_ch+"\"\"\"\n")
        train_file.write(tab_ch+"This function handles the functionality of 1D classifier.\n")
        train_file.write(tab_ch+"\"\"\"\n")
        train_file.write(tab_ch+"if len(sys.argv) < 2:\n")
        train_file.write(tab_ch*2+"print(\"Enter input data filename as a command line argument.\")\n")
        train_file.write(tab_ch * 2 + "sys.exit(1)\n\n")
        train_file.write(tab_ch+"validation_filename = sys.argv[1]\n")
        train_file.write(tab_ch+"attribute_id = "+str(attribute_id)+"\n")
        train_file.write(tab_ch+"best_threshold = "+str(best_threshold)+"\n")
        train_file.write(tab_ch + "side = '"+side+"'\n")
        train_file.write(tab_ch+"with open(validation_filename, \"r\") as file:\n")
        train_file.write(tab_ch*2+"# Read all rows of the file and iterate from the second row.\n")
        train_file.write(tab_ch*2+"# Ignore the first row containing header.\n")
        train_file.write(tab_ch*2+"for row in file.readlines()[1:]:\n")
        train_file.write(tab_ch*3+"# If the row is not empty add data point in the list.\n")
        train_file.write(tab_ch*3+"cols = row.strip().split(\",\")\n")
        train_file.write(tab_ch*3+"# check if validation data has age and height information\n")
        train_file.write(tab_ch*3+"if len(cols) >= 2:\n")
        train_file.write(tab_ch*4+"# print +1 for all attributes greater than the best_threshold\n")
        train_file.write(tab_ch*4+"if float(cols[attribute_id].strip()) > best_threshold:\n")
        train_file.write(tab_ch*5+"class_val = \"+1\" if side == \"right\" else \"-1\"\n")
        train_file.write(tab_ch*5+"print(class_val)  # Positive Class - Bhutan\n")
        train_file.write(tab_ch*4+"# print -1 for all attributes less than or equal to the best_threshold\n")
        train_file.write(tab_ch*4+"else:\n")
        train_file.write(tab_ch*5+"class_val = \"-1\" if side == \"right\" else \"+1\"\n")
        train_file.write(tab_ch*5+"print(class_val)  # Negative Class - Assam\n\n\n")
        train_file.write("if __name__ == '__main__':\n")
        train_file.write(tab_ch + "\"\"\"\n")
        train_file.write(tab_ch + "Run if executed as a script.\n")
        train_file.write(tab_ch + "\"\"\"\n")
        train_file.write(tab_ch+"main()\n")


def main():
    """
    This function drive the flow of the program.
    It handles end-to-end functioning of the 1D classification program.
    """
    if len(sys.argv) < 2:
        print("Enter input data filename as a command line argument.")
        sys.exit(1)

    # Take the command line input
    ip_data_filename = sys.argv[1]

    # Read the input file and get the data in list format
    input_data = read_file(ip_data_filename)

    # Extract age, height and class from the input_data
    age_data, height_data, class_data = extract_age_height_class(input_data)

    # Get quantized age data using flooring method in buckets of size 2 yrs.
    quant_age_data = get_quantized_data(age_data, 2, "floor")

    # Get quantized height data using flooring method in buckets of size 5 cm.
    quant_height_data = get_quantized_data(height_data, 5, "floor")

    # Initialize lists for storing assam and bhutan snowfolks age and height data
    ht_bhutan = []
    ht_assam = []
    age_bhutan = []
    age_assam = []
    # Iterate over each record and save values of assam and bhutan snowfolks in respective lists
    for rec_idx in range(len(class_data)):
        if class_data[rec_idx] == "+1":
            age_bhutan.append(quant_age_data[rec_idx])
            ht_bhutan.append(quant_height_data[rec_idx])
        else:
            age_assam.append(quant_age_data[rec_idx])
            ht_assam.append(quant_height_data[rec_idx])

    # Visualize the age data
    plt.hist(age_bhutan, 75, color="red")
    plt.hist(age_assam, 75, color="blue")
    plt.title("Visualize snowfolks age data")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.legend(['Age of snowfolks from Bhutan', 'Age of snowfolks from Assam'])
    plt.show()

    # Visualize the height data
    plt.hist(ht_bhutan, 35, color="red")
    plt.hist(ht_assam, 35, color="blue")
    plt.title("Visualize snowfolks height data")
    plt.xlabel("Height")
    plt.ylabel("Frequency")
    plt.legend(['Height of snowfolks from Bhutan', 'Height of snowfolks from Assam'])
    plt.show()

    # Decide on the target variable
    target_var = "+1"
    # Decide the side at which the maximum distribution of the target variable is present
    if np.mean(age_assam) < np.mean(age_bhutan):
        age_side = "right" if target_var == "+1" else "left"
    else:
        age_side = "left" if target_var == "+1" else "right"

    # Find the best threshold using age
    age_best_threshold, age_all_costs, age_all_thresholds, age_true_positive_rate, age_false_positive_rate, \
        age_best_idx = get_classification_model(quant_age_data, class_data, target_var, age_side, bin_size=2)

    print("Age\t\t: \tBest Threshold: {}, \tBest Cost: {}".format(age_best_threshold, age_all_costs[age_best_idx]))

    # Plot the cost function curve for results obtained by using age.
    plot_cost_function(age_best_threshold, age_all_costs, age_all_thresholds, age_best_idx, attribute="AGE")

    # Decide on the target variable
    target_var = "+1"
    # Decide the side at which the maximum distribution of the target variable is present
    if np.mean(ht_assam) < np.mean(ht_bhutan):
       ht_side = "right" if target_var == "+1" else "left"
    else:
        ht_side = "left" if target_var == "+1" else "right"

    # Find the best threshold using height
    ht_best_threshold, ht_all_costs, ht_all_thresholds, ht_true_positive_rate, ht_false_positive_rate, ht_best_idx\
        = get_classification_model(quant_height_data, class_data, target_var, ht_side, bin_size=5)

    print("Height\t: \tBest Threshold: {}, \tBest Cost: {}".format(ht_best_threshold, ht_all_costs[ht_best_idx]))

    # Plot the cost function curve for results obtained by using height.
    plot_cost_function(ht_best_threshold, ht_all_costs, ht_all_thresholds, ht_best_idx, attribute="HEIGHT")

    # Plot the ROC Curve for both Age and Height
    plt.plot(age_false_positive_rate, age_true_positive_rate, '-bv', markersize=5)
    plt.plot(age_false_positive_rate[age_best_idx], age_true_positive_rate[age_best_idx], "ro", markersize=10)
    plt.plot(ht_false_positive_rate, ht_true_positive_rate, '-yv', markersize=5)
    plt.plot(ht_false_positive_rate[ht_best_idx], ht_true_positive_rate[ht_best_idx], "ko", markersize=10)
    plt.plot([0, 1], [0, 1], '--g')
    plt.title("Receiver Operator (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(['ROC Curve for Age', 'Best Threshold for Age', 'ROC Curve for Height',
                'Best Threshold for Height', 'Coin Toss Line'], loc=4)
    plt.grid()
    plt.gca().set_aspect("equal")
    plt.show()

    # Decide which attribute to choose based on the minimum best cost value
    # and write the trained program accordingly.
    if age_all_costs[age_best_idx] < ht_all_costs[ht_best_idx]:
        print("Best Attribute / Attribute used to write trained classifier : AGE")
        write_trained_classifier("age", age_best_threshold, age_side)
    else:
        print("Best Attribute / Attribute used to write trained classifier : HEIGHT")
        write_trained_classifier("height", ht_best_threshold, ht_side)


if __name__ == '__main__':
    """
    Run if executed using script.
    """
    main()
