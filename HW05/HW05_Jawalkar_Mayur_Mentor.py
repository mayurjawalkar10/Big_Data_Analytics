"""
Authors: Mayur Sunil Jawalkar (mj8628)
         Kunjan Suresh Mhaske (km1556)
Big Data Analytics: Homework-05
Description: In this assignment we are predicting the class of abominable snowfolks data.
             We are creating a fast decision cascade for making the decisions/ predictions.
"""

# Import Statements
import pandas as pd  # Pandas library to handle data inputs from files.
import math  # Math library to handle log operations.


class DecisionTree:
    """
    This class implements the behavior of a decision tree. It makes use of an Information gain
    as a measure of purity of any node/split. It also performs a 10-fold cross validation to
    fine tuning the generated tree.
    """

    # Variables to use all over the class
    __slots__ = 'best_tree', 'all_tree_dict', 'iterator'

    def __init__(self):
        """
        Initialize all class level variables.
        """
        self.all_tree_dict = dict()  # saves all trees for each cross validation
        self.best_tree = dict()  # best tree amongst all trees generated in the cross validation.
        self.iterator = 0  # Iterator to iterate over trees generated using cross validation.

    def train(self, data, recursion_depth=8):
        """
        This method helps to perform cross validation and get the best decision tree from
        the given input data. It gives call to a recursive function test_helper to actual
        train the decision tree.

        :param data: input data to train the decision tree on.
        :recursion_depth: limit of maximum recursions.
        """
        best_accuracy = 0.0  # initialize the accuracy value.
        chunk_size = len(data) // 10  # Initialize the size of each chunk that we will be using for cross validation.

        # Iterate for 10 times to perform 10 fold cross validation
        for chunk_id in range(10):
            # Extract the chunk of a testing data from original data.
            test_data = data[chunk_id * chunk_size:(chunk_id * chunk_size) + chunk_size]
            # Save remaining data as a training data.
            train_data = data[~data.isin(test_data)].dropna()

            # Set iterator to save the tree generated using the current cross validation setting.
            self.iterator = chunk_id
            self.all_tree_dict[self.iterator] = dict()  # dictionary to save the current decision tree

            # Call to the helper function to actually train the tree using current setting.
            self.train_helper(train_data, recursion_depth)

            # calculate the accuracy by testing the generated decision tree using test data.
            accuracy = self.test(test_data, self.all_tree_dict[self.iterator])

            # Compare the accuracy and save the best decision tree with maximum accuracy.
            if accuracy > best_accuracy:
                best_accuracy = accuracy  # update the accuracy
                self.best_tree = self.all_tree_dict[self.iterator]  # update the best tree

        # Print all trees generated during the cross validation.
        # print("Printing all generated Trees: ")
        # for key, value in self.all_tree_dict.items():
        #     print(value)

        # Print the best tree generated during the cross validation.
        print("Best Tree after Cross Validation : \n")
        tab_space = "  "
        count = 0
        print("Predictions are made at each node.")
        for node_values in self.best_tree.values():
            print(tab_space*count, "Attribute : ", str(node_values[0]), "| Threshold : ", str(node_values[1]), end=' |')
            print(" Prediction : ", str(node_values[2]))
            if node_values[3] is not None:
                print(tab_space*(count+1), "Prediction : ", str(node_values[3]))
            count += 1
        print("\nAccuracy for a above tree: ", best_accuracy)

    def train_helper(self, data, recursion_depth):
        """
        This function performs the actual task of training the decision tree and saving it in a dictionary.
        It uses Information Gain as a measure of purity.

        :param data: input data to train the decision tree
        :param recursion_depth: maximum limit of the allowed recursions.
        """

        # return if recursion depth is reached.
        if recursion_depth <= 0:
            return

        all_attributes = list(data.columns)  # get the list of all attributes to iterate
        best_attribute = all_attributes[0]  # initialize the first attribute as a best attribute.
        best_gain = -999999  # Initialize th best_gain
        best_threshold = 0  # Initialize the best_threshold to 0

        # Iterate over all attributes to see if it is the best attribute to split the data.
        for attribute in all_attributes:
            # Ignore the attribute 'Class'
            if attribute == 'Class':
                continue

            # Extract the attribute data which we will be considering
            considered_data = data[[attribute, 'Class']]

            # Compute the information gain and the threshold for current attribute.
            threshold, gain = self.get_best_threshold_and_gain(considered_data, attribute)

            # Update the best attribute and threshold values to maximize the gain.
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
                best_threshold = threshold

        # Split the data using the best attribute and threshold. Extract the information of the split.
        left_data, right_data, left_class_freq, right_class_freq, left_purity, right_purity = \
            self.split_data_using_threshold(data, best_attribute, best_threshold)

        # Check for the exit conditions for our recursion. i.e., Decide the class you will be
        # expanding further. Check if it is more than 95% pure or it has less than 15 elements
        # present in it. If so, don't recurse.
        if left_purity < right_purity and left_purity < 0.95 and len(left_data) > 15:
            # Call a function to write the currently generated tree on a dictionary.
            self.write_tree(best_attribute, best_threshold, recursion_depth, right_class_freq)
            self.train_helper(left_data, recursion_depth - 1)  # Recursion call
        elif left_purity >= right_purity and right_purity < 0.95 and len(right_data) > 15:
            # Call a function to write the currently generated tree on a dictionary.
            self.write_tree(best_attribute, best_threshold, recursion_depth, left_class_freq)
            self.train_helper(right_data, recursion_depth - 1)  # Recursion call
        else:
            # If we decide to break out of the recursion mark the leaf nodes of the tree and
            # write the final values on the dictionary.
            self.write_tree(best_attribute, best_threshold, recursion_depth, left_class_freq, right_class_freq)

    def split_data_using_threshold(self, data, best_attribute, best_threshold):
        """
        This function splits the data based on the best attribute and threshold.
        It also computes the information of the split.

        :param data: Data to split.
        :param best_attribute: attribute to split the data.
        :param best_threshold: threshold to use to split the data.
        :return left_data: Data on the left half after the split.
        :return right_data: Data on the right half after the split.
        :return left_class_freq: Class frequency of the left half of the data.
        :return right_class_freq: Class frequency of the right half of the data.
        :return left_purity: Purity of the left half of the data.
        :return right_purity: Purity of the right half of the data.
        """

        # Split the data using threshold.
        left_data = data[data[best_attribute] <= best_threshold]  # Values less than or equal to threshold to the left.
        right_data = data[data[best_attribute] > best_threshold]  # Values greater than threshold to the right.

        # Compute the frequency of Assam and Bhuttan on each half of the data.
        left_class_freq = left_data['Class'].value_counts().to_dict()
        right_class_freq = right_data['Class'].value_counts().to_dict()

        # Compute the purity of left half of the data. Here purity is
        # the ratio of a class frequency with total records.
        left_purity = max([left_class_freq[class_value] / sum(left_class_freq.values())
                           for class_value in left_class_freq.keys()])

        # Compute the purity of right half of the data. Here purity is
        # the ratio of a class frequency with total records.
        right_purity = max([right_class_freq[class_value] / sum(right_class_freq.values())
                            for class_value in right_class_freq.keys()])
        return left_data, right_data, left_class_freq, right_class_freq, left_purity, right_purity

    def write_tree(self, best_attribute, best_threshold, recursion_depth, left_class_freq, right_class_freq=None):
        """
        This function writes the generated decision tree on a dictionary.

        :param best_attribute: Attribute to use on a node to split the data.
        :param best_threshold: Threshold to use for the split.
        :param recursion_depth: Current depth of a recursion.
        :param left_class_freq: Frequencies of the classes on the left side.
        :param right_class_freq: Frequencies of the classes on the right side.
        """
        # Save the prediction class on the left side of the data.
        left_pred_class = max(left_class_freq, key=left_class_freq.get)
        # Standardize the prediction values: -1 for Assam and +1 for Bhuttan
        left_predicted_class = "-1" if left_pred_class == 'Assam' else "+1"
        # Save the information on a dictionary.
        self.all_tree_dict[self.iterator][8 - recursion_depth] = \
            [best_attribute, best_threshold, left_predicted_class, None]
        # Check if it is a last split.
        if right_class_freq is not None:
            # Predict value for right node.
            right_pred_class = max(right_class_freq, key=right_class_freq.get)
            # Standardize the prediction values: -1 for Assam and +1 for Bhuttan
            right_predicted_class = "-1" if right_pred_class == 'Assam' else "+1"
            # Update the right prediction value.
            self.all_tree_dict[self.iterator][8 - recursion_depth][3] = right_predicted_class

    def get_best_threshold_and_gain(self, data, attribute):
        """
        This function computes the best gain and threshold value for a given attribute.

        :param data: input data to find gain
        :param attribute: attribute to use to find gain.
        :return best_threshold: Threshold where we get the maximum gain for a given attribute.
        :return best_gain: Best gain value obtained.
        """

        best_threshold = 0  # Initialize the best threshold value.
        best_gain = -999999  # Initialize the best gain value.
        all_threshold = sorted(set(data[attribute]))  # Get a sorted set of all unique values in the attribute.
        # Iterate over all threshold and compute the gain
        for threshold in all_threshold:
            # Compute the gain for current threshold.
            gain = self.calculate_gain(data, attribute, threshold)
            # Update the best_gain to be maximum.
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        return best_threshold, best_gain

    def calculate_gain(self, data, attribute, threshold):
        """
        This function computes the gain for a given data, attribute and threshold pair.

        :param data: data to compute gain.
        :param attribute: attribute to use to compute the gain.
        :param threshold: threshold value to use to compute the gain.
        :return (entropy-weighted_entropy): It is a gain value
        """
        # Get the class frequencies
        total_class_frequencies = data['Class'].value_counts().to_dict()
        # Frequency of assam in complete data.
        all_fq_assam = total_class_frequencies['Assam'] if 'Assam' in total_class_frequencies.keys() else 0
        # Frequency of Bhuttan in complete data.
        all_fq_bhuttan = total_class_frequencies['Bhuttan'] if 'Bhuttan' in total_class_frequencies.keys() else 0

        # Dictionary to store the frequency information of a split.
        split_class_freq = dict()
        # Frequency counts in left data.
        split_class_freq['left'] = data[data[attribute] <= threshold]['Class'].value_counts().to_dict()
        # Frequency counts in right data.
        split_class_freq['right'] = data[data[attribute] > threshold]['Class'].value_counts().to_dict()

        # Calculate the entropy
        entropy = self.calculate_entropy(all_fq_assam, all_fq_bhuttan)
        # Calculate the weighted entropy.
        weighted_entropy = self.calculate_weighted_entropy(split_class_freq, len(data[attribute]))
        return entropy - weighted_entropy

    def calculate_entropy(self, freq_assam, freq_bhuttan):
        """
        This function returns the entropy for a given input.

        :param freq_assam: Frequency count of assam.
        :param freq_bhuttan: frequncy count of Bhuttan.
        :return: entropy
        """
        if freq_assam == 0 and freq_bhuttan == 0:
            return 0
        ratio = freq_assam / (freq_assam + freq_bhuttan)
        if ratio == 0 or ratio == 1:
            return 0
        return -1 * ((ratio * math.log2(ratio)) + ((1 - ratio) * math.log2(1 - ratio)))

    def calculate_weighted_entropy(self, split_class_freq, total_data):
        """
        This functin calculates the weighted entropy for a given split data.

        :param split_class_freq: Class Frequency information.
        :param total_data: total records in a parent node for a given split.
        :return: weighted entropy
        """
        weighted_entropy = 0.0  # Initialize the value of weighted entropy.
        # Iterate over each side of a split
        for side in split_class_freq.keys():
            # Frequency of assam on a considered side of a split.
            freq_assam = split_class_freq[side]['Assam'] if 'Assam' in split_class_freq[side].keys() else 0
            # Frequency of bhuttan on a considered side of a split.
            freq_bhuttan = split_class_freq[side]['Bhuttan'] if 'Bhuttan' in split_class_freq[side].keys() else 0

            # Ratio of frequencies of assam and bhutan to the total frequency count of parent.
            ratio = (freq_assam + freq_bhuttan) / total_data

            # Add results to the weighted_entropy
            weighted_entropy += (ratio * self.calculate_entropy(freq_assam, freq_bhuttan))
        return weighted_entropy

    def test(self, data, tree_dict, print_confusion_matrix=False):
        """
        This function tests the given decision tree on a given data and it outputs the accuracy of a decision tree.
        It also computes the confusion matrix for generated results. By default it won't print the confusion matrix.
        But you can print it by setting the print_confusion_matrix parameter to True.

        :param data: data to test decision tree on.
        :param tree_dict: tree to be tested.
        :param print_confusion_matrix: boolean value to decide whether to print confusion matrix.
        :return: It returns the accuracy.
        """
        # Return accuracy = 0 if no data is passed.
        if len(data) == 0:
            return 0

        correct = 0  # Initialize to keep track of correct predictions.

        # Create confusion matrix dataframe
        confusion_matrix = pd.DataFrame(columns=['Assam', 'Bhuttan'], index=['Assam', 'Bhuttan'])
        confusion_matrix = confusion_matrix.fillna(0)  # Update NA values with 0

        # Iterate over each record to check the prediction.
        for record_idx in range(data.index.start, data.index.stop, 1):
            record = data.iloc[[record_idx - data.index.start]]  # current record value
            correct_pred_flag = False  # Flag to check correct prediction.
            # Iterate over each condition.
            for condition in tree_dict.values():
                if condition is None:  # break if condition is None.
                    break
                # Check the value of a attribute and condition in the decision tree
                if float(record[condition[0]]) <= condition[1]:
                    if condition[2] == '+1':  # if decision tree predicts Bhutan
                        if record['Class'][record_idx] in ('Bhuttan', '+1', 1):  # if record contains bhutan
                            confusion_matrix.loc['Bhuttan', 'Bhuttan'] += 1  # add it in confusion matrix
                            correct_pred_flag = True  # mark it as correct prediction
                            break
                    elif condition[2] == '-1':  # if decision tree predicts Assam
                        if record['Class'][record_idx] in ('Assam', '-1', -1):  # it is actually Assam
                            confusion_matrix.loc['Assam', 'Assam'] += 1  # add it in confusion matrix
                            correct_pred_flag = True  # mark it as correct prediction
                            break
                elif condition[3] is not None:
                    if condition[3] == '+1':  # if decision tree predicts Bhutan
                        if record['Class'][record_idx] in ('Bhuttan', '+1', 1):  # it is actually Bhutan
                            confusion_matrix.loc['Bhuttan', 'Bhuttan'] += 1  # add it in confusion matrix
                            correct_pred_flag = True  # mark it as correct prediction
                            break
                        else:
                            confusion_matrix.loc['Assam', 'Bhuttan'] += 1  # add it in confusion matrix
                    elif condition[3] == '-1':  # if decision tree predicts Assam
                        if record['Class'][record_idx] in ('Assam', '-1', -1):  # it is actually Assam
                            confusion_matrix.loc['Assam', 'Assam'] += 1  # add it in confusion matrix
                            correct_pred_flag = True  # mark it as correct prediction
                            break
                        else:
                            confusion_matrix.loc['Bhuttan', 'Assam'] += 1  # add it in confusion matrix
            # Check for a correct prediction and update the correct count
            if correct_pred_flag:
                correct += 1
        # Check whether to print confusion matrix.
        if print_confusion_matrix:
            print("\nConfusion Matrix : (Actual values as ROWS and Predicted values as COLUMNS)")
            print(confusion_matrix)
        return correct / len(data)

    def write_trained_decision_tree(self, tree_dict):
        """
        This function writes the trained decision tree on a file. This file takes a command line argument
        for a validation data. It prints out the results as standard output and file output.

        :param tree_dict: Decision tree to use
        """
        tab_ch = "    "
        with open("HW05_Jawalkar_Mayur_Trained_Classifier.py", "w+", encoding="utf-8") as train_file:
            train_file.write("# Import statements\n")
            train_file.write("import sys\n")
            train_file.write("import pandas as pd\n\n\n")
            train_file.write("def main():\n")
            train_file.write(tab_ch + "\"\"\"\n")
            train_file.write(tab_ch + "This function predicts the class of abominable snowfolks data.\n")
            train_file.write(tab_ch + "\"\"\"\n")
            train_file.write(tab_ch + "if len(sys.argv) < 2:\n")
            train_file.write(tab_ch * 2 + "print('Filename not provided. Exiting..')\n")
            train_file.write(tab_ch * 2 + "sys.exit(1)\n\n")
            train_file.write(tab_ch + "validation_filename = sys.argv[1]\n")
            train_file.write(tab_ch + "validation_data = pd.read_csv(validation_filename)\n\n")
            train_file.write(tab_ch + "with open('HW05_Jawalkar_Mayur__MyClassifications.csv', 'w', encoding='utf8') as file:\n")
            train_file.write(tab_ch*2 + "for index, row in validation_data.iterrows():\n")
            count = 0
            # Iterate over each rule of a decision tree
            for rulekey, rulevalue in tree_dict.items():
                count += 1  # counter to check first if condition
                if rulevalue is not None:  # If there is a rule present
                    if_or_elif = "" if count == 1 else "el"  # decide whether to use if or elif based on count
                    train_file.write(tab_ch * 3 + if_or_elif + "if float(row['" +
                                     str(rulevalue[0]) + "']) <= " + str(rulevalue[1]) + ":\n")
                    train_file.write(tab_ch * 4 + "print('" + rulevalue[2] + "')\n")
                    train_file.write(tab_ch * 4 + "file.write('" + rulevalue[2] + "\\n')\n")
                    if rulevalue[3] is not None:
                        train_file.write(tab_ch * 3 + "else:\n")
                        train_file.write(tab_ch * 4 + "print('" + str(rulevalue[3]) + "')\n")
                        train_file.write(tab_ch * 4 + "file.write('" + rulevalue[3] + "\\n')\n\n\n")
            train_file.write("if __name__ == '__main__':\n")
            train_file.write(tab_ch + "\"\"\"\n")
            train_file.write(tab_ch + "Run if executed as a script.\n")
            train_file.write(tab_ch + "\"\"\"\n")
            train_file.write(tab_ch + "main()\n")

def get_quantized_data(data):
    """
    This function creates a quantized copy of a input data. It quantize the height data to nearest 5,
    age to nearest 2 and rest of the attributes to the nearest 1 value. It updates the data inplace.

    :param data: input data to quantize.
    :return data: quantized data.
    """
    # Iterate over each attribute to quantize the data.
    for attrib in list(data.columns):
        # ignore 'Class' attribute.
        if attrib == 'Class':
            continue
        # Set the bin size according to attributes.
        # 2 for age, 5 for height and 1 for others.
        bin_size = 2 if attrib == 'Age' else 5 if attrib == 'Ht' else 1

        # Get the index value for the column.
        column_idx = data.columns.get_loc(attrib)
        # Iterate over each record and update the data to the quantized value.
        for row in range(len(data[attrib])):
            cell_value = data[attrib][row]  # original value
            # update the data to the quantized value.
            data.iloc[row, column_idx] = (round(cell_value / bin_size) * bin_size)
    return data


def main():
    """
    This function drives the execution of the program. It gives call for decision tree creation,
    validation, confusion matrix generation, etc. It also gives a call to write a decision tree
    on a file.
    """
    # Read the input data file
    input_data = pd.read_csv("Abominable_Data_HW05_v720.csv")
    # Convert the data into quantized data.
    quantized_data = get_quantized_data(input_data)

    # Create an object of a decision tree.
    decision_tree = DecisionTree()

    print("\nTraining Decision Tree...")
    # Train the decision tree using the quantized data.
    decision_tree.train(quantized_data)

    print("\nTesting Decision Tree on a Training data...")
    # check the result of the generated decision tree on a Training data.
    print(decision_tree.test(input_data, decision_tree.best_tree, True))

    print("\nTesting Decision Tree on a Validation data...")
    # check the result of the generated decision tree on a validation data.
    valid_data = pd.read_csv("VALIDATION_DATA_TO_RELEASE.csv")
    print(decision_tree.test(valid_data, decision_tree.best_tree, True))

    # Write the decision tree on a file.
    decision_tree.write_trained_decision_tree(decision_tree.best_tree)


# Execute as a script
if __name__ == '__main__':
    main()
