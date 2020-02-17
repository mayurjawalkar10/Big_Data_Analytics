"""
@author : Jawalkar Mayur (mj8628)
Big Data Analytics
Homework - 01
"""

# import statements
from matplotlib import pyplot as plt # for plotting graph
import math  # for mathematical operations like floor.


def read_file(filename, delimit=","):
    """
    This function reads the file for the data input.
    It returns the list of data points.

    :filename: Name of the file to read data from
    :delimit: (Optional) Delimiter character to seperate the column values. Default it is set to ",".
    :return: data
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
                    all_col_records.append(float(record))
                data.append(all_col_records)
            if len(cols) == 1:
                data.append(float(cols[0]))
        # Return the list containing all data points.
        return data


def get_mean(data):
    """
    This function returns the mean for a given list of data points.

    :param: data - list containing the data points
    :return: mean - float value
    """
    addition = 0

    # Iterate over all data points and calculate the sum
    for value in data:
        addition += value

    # Calculate the mean
    try:
        mean = addition / len(data)
    except ZeroDivisionError:
        mean = 0
    return mean


def get_std_deviation(data, mean=None):
    """
    This function returns the standard deviation for a given list of data points.
    If the mean value is not passed, it internally calls the get_mean function to calculate the mean value.

    :data:  list containing the data points
    :mean: (Optional) Float value containing a mean of the given data points. Default value is None.
    :return: mean - Float value
    """
    # Calculate the mean of given data points if mean value is not passed.
    if mean is None:
        mean = get_mean(data)

    sqr_sum_of_diff = 0

    # Iterate over all data points and calculate the squared sum of the difference between a point and the mean.
    for value in data:
        # Calculate the squared sum of the difference between a point and the mean.
        sqr_sum_of_diff += (value - mean)**2

    # Calculate the standard deviation for a given list of data points.
    try:
        std = math.sqrt(sqr_sum_of_diff/len(data))
    except ZeroDivisionError:
        std = 0
    return std


def get_quantized_data(data, bin_size, bin_method='floor'):
    """
    This method quantize each value in the data using a given bin size and binning method.

    :data: list of data points to be quantized
    :bin_size: int value for specifying the size of each bin
    :bin_method: {"floor" or "ceil" or "round"}. Default binning method is "floor".
    :return: list of quantized data points.
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


def calc_freq(data):
    """
    This function calculates the frequency of occurrence of each data point in the given list.
    It returns the dictionary containing the frequency of the quantized values.

    :data: list of input data
    :return: value_freq: dictionary containing frequency of each quantized value.
    """
    value_freq = dict()
    for value in data:
        if value not in value_freq.keys():
            value_freq[value] = 1
        else:
            value_freq[value] += 1
    return value_freq


def get_left_right_values(data, threshold):
    """
    This function iterates over the input data and splits the data at a given threshold.

    Values less than or equal to the threshold will be added to left_values list and values greater than the
    threshold will be added to right_values list. left_values and right_values are returned.

    :data: Input data to split based on a given threshold.
    :threshold: value based on which data is to split.
    :return: left_value : list of values less than or equal to the threshold
    :return: right_value: list of values greater than the threshold
    """
    left_values = []
    right_values = []
    for value in data:
        if value <= threshold:
            left_values.append(value)
        else:
            right_values.append(value)
    return left_values, right_values


def otsu_classification(data, bin_size, perform_regularization=False, alpha=0):
    """
    This function achieves the behavior of otsu classifier. It is a binary classifier.
    It gives the best threshold value for splitting the 1D data.

    :data: input data for classification
    :bin_size: size of each bin
    :perform_regularization: boolean value to decide whether to perform regularization or not. Default it is False.
    :alpha: alpha value to use in the regularization. Default it is 0.
    :return: best_threshold (left most value is returned in case of a tie.),
             best_cost_value (minimum/best cost value),
             all_best_thresholds (list of all thresholds where we can get best cost value),
             all_cost_values (list of cost values achieved at each threshold level).
    """
    best_threshold = 0
    best_cost_value = float("inf")
    all_cost_values = []
    all_best_thresholds = []
    norm_factor = 100

    # Iterate over all threshold values to calculate the value of cost function
    for threshold in range(min(data), max(data)+1, bin_size):
        left, right = get_left_right_values(data, threshold)  # split the given data based on the threshold value.
        wt_left = len(left) / len(data)  # Calculate the fraction of left list
        wt_right = len(right) / len(data)  # Calculate the fraction of right list

        variance_left = get_std_deviation(left)**2  # Calculate the variance of left list
        variance_right = get_std_deviation(right)**2  # Calculate the variance of right list

        # If perform_regularization is True calculate the regularization value else keep it as 0.
        if perform_regularization:
            regularization = (abs(len(left)-len(right))/norm_factor)*alpha  # regularization function
        else:
            regularization = 0

        # Calculate the cost for each threshold
        cost = (wt_left*variance_left) + (wt_right*variance_right) + regularization  # Cost function
        all_cost_values.append(cost)  # save the cost value for each threshold

        # Update the best_cost_value to be minimum.
        if cost < best_cost_value:
            best_cost_value = cost
            best_threshold = threshold
            # maintain a list of all possible threshold where we can get best / minimum cost.
            all_best_thresholds = [threshold]
        # check for tie
        elif cost == best_cost_value:
            # maintain a list of all possible threshold where we can get best / minimum cost.
            all_best_thresholds.append(threshold)

    # if there is a tie, Warn user to pay extra attention.
    if len(all_best_thresholds) > 1:
        print("\nNOTE: More than one thresholds found for best cost value. "
              "Currently considering the leftmost value for the best threshold.")

    return best_threshold, best_cost_value, all_best_thresholds, all_cost_values


def otsu_regularization_helper(data, bin_size, all_alpha):
    """
    This is a helper function for performing regularization on otsu classification.
    It accepts a list of all alpha values to use for regularization.

    :data: input data for performing classification
    :bin_size: size of each bin
    :all_alpha: list of all alpha values to consider for regularization.
    :return: results: dictionary containing results of otsu with regularization for each alpha value.
    """
    results = dict()
    # Iterate through all alpha values and perform the regularization.
    for alpha in all_alpha:
        results[alpha] = otsu_classification(data, bin_size, True, alpha)
    return results


def plot_variance_vs_quantized_data_graph(data, min_threshold, best_threshold, best_mixed_var, bin_size):
    """
    This function plots the graph of the mixed variance for the snowfolk’s data based on age versus the quantized age.
    Plot a circular point indicating the value used to segment the data into two clusters.

    :data: mixed variance data for all points
    :min_threshold: minimum threshold value present in the data.
    :best_threshold: threshold value to get the best split.
    :best_mixed_var: mixed variance value to get the best split.
    :bin_size: size of each bin.
    """
    plt.plot([min_threshold + (i * bin_size) for i in range(0, len(data))], data, "-gD", markersize=3, linewidth=1)
    plt.plot(best_threshold, best_mixed_var, "ro", markersize=7)
    plt.title("Mixed Variance vs Quantized age")
    plt.xlabel("Quantized Age")
    plt.ylabel("Mixed Variance")
    plt.legend(['Mixed Variance', 'Best Mixed Variance'], loc=4)
    plt.show()


def main():
    """
    This function calls all other functions.

    Note**  "All file handling operations are performed under the assumption
             that those files are present in the same directory."
    """
    data = read_file("Mystery_Data_2195.csv")  # Read file and get the list of data points

    mean = get_mean(data)  # Get the mean of the data points
    std = get_std_deviation(data, mean)  # Get the standard deviation of the data points

    # Print the results for complete data
    print("\nComplete Data:\n\t{:19} : {:.3f}\n\t{:19} : {:.3f}"
          .format('Mean', mean, 'Standard Deviation', std))

    new_data = data[:-1]  # Create new data sample by removing last data point.
    new_mean = get_mean(new_data)  # Get the mean of the new data
    new_std = get_std_deviation(new_data, mean)  # Get the standard deviation of the new data

    # Print Results for new data
    print("\nData Without Last Value:\n\t{:19} : {:.3f}\n\t{:19} : {:.3f}"
          .format('Mean', new_mean, 'Standard Deviation', new_std))

    abominable_data = read_file("../HW02/Abominable_Data_For_1D_Classification__v93_HW3_720_final.csv")  # Read the abominable data from the file

    age_list = [row[0] for row in abominable_data]  # Extract the ages from the abominable data

    # quantize the age data into the bins of size 2 using floor function
    quantized_age_list = get_quantized_data(age_list, 2, "floor")

    print("\nOTSU Classification on Abominable Age data :-")

    # Apply OTSU classification with bin_size= 2 without regularization.
    best_threshold, best_mixed_var, all_best_thresholds, all_mixed_var = otsu_classification(quantized_age_list, 2)

    a, b = get_left_right_values(quantized_age_list, best_threshold)  # split data using best threshold
    print("\tBest Threshold: {}, Best Mixed Variance: {:.4f}, All possible best thresholds: {}, Split: ({} ,{})"
          .format(best_threshold, best_mixed_var, all_best_thresholds, len(a), len(b)))

    # Plot the Mixed Variance vs Quantized Age graph
    plot_variance_vs_quantized_data_graph(all_mixed_var, min(quantized_age_list), best_threshold,
                                          best_mixed_var, bin_size=2)

    # List of alpha values to use for the regularization.
    all_alpha = [100, 1, 1/5, 1/10, 1/20, 1/25, 1/50, 1/100, 1/1000]

    # Apply the OTSU classification on age data with regularization using alpha values from all_alpha.
    # result is the dictionary containing results of otsu with regularization for each alpha value.
    results = otsu_regularization_helper(quantized_age_list, 2, all_alpha)

    print("\n[Abominable Age Data] : Results after regularization for different values of alpha:")

    # Iterate through the results for each alpha value and print the best thresholds.
    for alpha in results.keys():
        a, b = get_left_right_values(quantized_age_list, results[alpha][0])
        print("\tAlpha {} \t: Best threshold: {}, Best Cost Value: {:.4f}, All Best Thresholds: {}, Split: ({} ,{})"
              .format(alpha, results[alpha][0], results[alpha][1], results[alpha][2], len(a), len(b)))

    height_list = [row[1] for row in abominable_data]  # Extract the heights from the abominable data

    # quantize the height data into the bins of size 5 using floor function.
    quantized_height_list = get_quantized_data(height_list, 5, "floor")

    print("\nOTSU Classification on Abominable Height data")

    # Apply OTSU classification with bin_size=5 without regularization.
    best_threshold, best_mixed_var, all_best_thresholds, all_mixed_var = otsu_classification(quantized_height_list, 5)

    a, b = get_left_right_values(quantized_height_list, best_threshold)  # split data using best threshold
    print("Best Threshold: {}, Best Mixed Variance: {:.4f}, All possible best thresholds: {}, Split: ({} ,{})"
          .format(best_threshold, best_mixed_var, all_best_thresholds, len(a), len(b)))

    # Apply the OTSU classification on height data with regularization using alpha values from all_alpha.
    # result is the dictionary containing results of otsu with regularization for each alpha value.
    results = otsu_regularization_helper(quantized_height_list, 5, all_alpha)

    print("\n[Abominable Height Data] : Results after regularization for different values of alpha:")
    # Iterate through the results for each alpha value and print the best thresholds.
    for alpha in results.keys():
        a, b = get_left_right_values(quantized_height_list, results[alpha][0])  # split data using best threshold
        print("\tAlpha {} \t: Best threshold: {}, Best Cost Value: {:.4f}, All Best Thresholds: {}, Split: ({} ,{})"
              .format(alpha, results[alpha][0], results[alpha][1], results[alpha][2], len(a), len(b)))


if __name__ == '__main__':
    main()
