"""
Author: Mayur Sunil Jawalkar (mj8628)
        Kunjan Suresh Mhaske (km1556)

Big Data Analytics: Homework-09
Description: This program performs tasks asked in Homework 09 that includes
            1. Re-balance the Data
            2. Feature Generation
            3. Feature Selection using Cross correlation
            4. Feature Selection Using PCA
            5. Create a Linear 2D Classifier
            6. Classify the Unknown data
            Finally it produces the results of scatter plots of most important features and accuracy of training.
"""

# Import statements
import pandas as pd  # To handle data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # To classify data using LDA
from sklearn.metrics import accuracy_score  # To compute the accuracy of LDA
import numpy as np  # To handle the data
import numpy.linalg as la  # To compute the eigen vectors and values.
from matplotlib import pyplot as plt  # To plot the graph


def get_balanced_data(input_data):
    """
    This function checks if data is balanced or not. It then balances the data by taking the
    random values from the majority class.

    :param input_data: Input data
    :return balanced_data: Balanced Data
    """
    # Check the frequency counts of each class from the data
    ip_data_class_freq = input_data['Class'].value_counts().to_dict()
    # Identify the class with minimum frequency
    min_class_freq = min(ip_data_class_freq.values())

    # Extract the Bhuttan data.
    bhutan_data = input_data.loc[input_data['Class'] == 'Bhuttan']

    # Extract the Assam data.
    assam_data = input_data.loc[input_data['Class'] == 'Assam']

    # Identify which class has maximum frequency
    max_freq_class_name = max(ip_data_class_freq, key=ip_data_class_freq.get)

    # Choose the class with maximum frequency to select random sample of data to make it equal to freq of another class.
    if max_freq_class_name == 'Assam':
        # Take sample equal to minimum class frequency.
        assam_data = assam_data.sample(n=min_class_freq, random_state=0)
    else:
        # Take sample equal to minimum class frequency.
        bhutan_data = bhutan_data.sample(n=min_class_freq, random_state=0)

    # Append the two classes to make the balanced data.
    balanced_data = assam_data.append(bhutan_data, ignore_index=True)

    # Compute the frequency counts of the balanced data to verify.
    balanced_data_class_freq = balanced_data['Class'].value_counts().to_dict()
    # print(balanced_data_class_freq)

    return balanced_data


def generate_features(balanced_data):
    """
    Generate the features using the pre-existing features from the data. This is done by performing basic mathematical
    operations on the data. This function computes various features like tail_less_hair, tail_less_bang, shag_factor,
    tail_and_hair, tail_and_bang, hair_and_bangs, all_lengths, ape_factor and height_age.

    return data: Data with added features
    """
    # Create lists to store each newly generated feature
    tail_less_hair = []  # tail length minus hair length
    tail_less_bang = []  # tail length minus bang length
    shag_factor = []  # hair length minus bang length
    tail_and_hair = []  # tail length plus hair length
    tail_and_bang = []  # tail length plus bang length
    hair_and_bangs = []  # hair length plus bang length
    all_lengths = []  # hair length plus bang length plus tail length
    ape_factor = []  # reach minus Height
    height_age = []  # Height minus age

    # Compute new feature values for each record in  the training data.
    for index, record in balanced_data.iterrows():
        tail_less_hair.append(record['TailLn'] - record['HairLn'])
        tail_less_bang.append(record['TailLn'] - record['BangLn'])
        shag_factor.append(record['HairLn'] - record['BangLn'])
        tail_and_hair.append(record['TailLn'] + record['HairLn'])
        tail_and_bang.append(record['TailLn'] + record['BangLn'])
        hair_and_bangs.append(record['HairLn'] + record['BangLn'])
        all_lengths.append(record['TailLn'] + record['HairLn'] + record['BangLn'])
        ape_factor.append(record['Reach'] - record['Ht'])
        height_age.append(record['Ht'] - record['Age'])

    # Create a copy of the data.
    data = balanced_data.copy()
    # Add the newly computed features in the original pandas dataframe.
    data['TailLessHair'] = tail_less_hair
    data['TailLessBang'] = tail_less_bang
    data['ShagFactor'] = shag_factor
    data['TailAndHair'] = tail_and_hair
    data['TailAndBangs'] = tail_and_bang
    data['HairAndBangs'] = hair_and_bangs
    data['AllLengths'] = all_lengths
    data['ApeFactor'] = ape_factor
    data['HeightAge'] = height_age

    return data


def get_best_2_features_brute_force(data):
    """
    This function performs brute force technique to identify the best performing pair
    of the features to classify the data.

    :param data: input data
    :return first_best_feature_pair: The pair of features which gives the best accuracy
    :return first_best_accuracy: The accuracy of the model using the first_best_feature_pair
    :return second_best_feature_pair: The pair of features which gives the second best accuracy
    :return second_best_accuracy: The accuracy of the model using the second_best_feature_pair
    """
    print("\nBrute Force Technique for feature selection:")
    # Extract the list of features
    feature_list = list(data.columns)
    # Remove Class from the list of features.
    feature_list.remove('Class')

    # Extract the target class
    target_class = data[['Class']]

    # Initialize the first two best performing feature pairs and their accuracies
    first_best_accuracy = 0.0
    first_best_feature_pair = [1, 2]
    second_best_accuracy = 0.0
    second_best_feature_pair = [1, 2]

    # Iterate over feature list
    for feature_index_1 in range(len(feature_list)):
        # Iterate over feature list
        for feature_index_2 in range(feature_index_1+1, len(feature_list)):
            # Consider data of only considered features i.e., feature_1 and feature_2
            considered_data = data[[feature_list[feature_index_1], feature_list[feature_index_2]]]

            # Instantiate the LDA model from the sklearn package
            lda_model = LDA()

            # Fit the model using the considered data.
            lda_model.fit(considered_data, target_class.values.ravel())

            # Test the model by predicting the classes for the training data.
            prediction = lda_model.predict(considered_data)

            # Compute the accuracy of our model
            accuracy = accuracy_score(target_class, prediction)

            # if the accuracy is better than first_best_accuracy and update the first
            # and second best accuracies accordingly
            if accuracy > first_best_accuracy:
                second_best_accuracy = first_best_accuracy
                second_best_feature_pair = first_best_feature_pair
                first_best_accuracy = accuracy
                first_best_feature_pair = [feature_index_1, feature_index_2]
            # if the accuracy is better than second_best_accuracy, update the second best accuracy.
            elif accuracy > second_best_accuracy:
                second_best_accuracy = accuracy
                second_best_feature_pair = [feature_index_1, feature_index_2]
    # print(feature_list[first_best_feature_pair[0]], feature_list[first_best_feature_pair[1]], first_best_accuracy)
    # print(feature_list[second_best_feature_pair[0]], feature_list[second_best_feature_pair[1]], second_best_accuracy)

    # Update the first_best_feature_pair by storing the names of the features in place of the indices.
    first_best_feature_pair = [feature_list[first_best_feature_pair[0]], feature_list[first_best_feature_pair[1]]]
    # Update the second_best_feature_pair by storing the names of the features in place of the indices.
    second_best_feature_pair = [feature_list[second_best_feature_pair[0]], feature_list[second_best_feature_pair[1]]]
    return first_best_feature_pair, first_best_accuracy, second_best_feature_pair, second_best_accuracy


def perform_principal_component_analysis(data):
    """
    This function performs the principal component analysis on the data. It also gives a call to the function to
    plot the scatter plot between the first two components.

    :param data: Input data
    """

    print("\nPrincipal Component Analysis:")
    # Extract the target class information from the data.
    target_class = data[['Class']]
    # Drop the class column from the data.
    data = data.drop(columns=['Class'])

    # Subtract the mean value from the data to get the mean subtracted data.
    mean_subtracted_data = data - data.mean()

    # Compute the covariance of the mean_subtracted_data.
    covariance_matrix = mean_subtracted_data.cov()

    # Compute the eigenvectors and eigenvalues using the covariance matrix.
    evalsh, evectsh = la.eigh(covariance_matrix)

    # Sort the eigenvalues in the descending order.
    evalsh = evalsh[::-1]

    # Sort the eigenvectors in the descending order to match the order of the eigenvalues.
    evectsh = evectsh[::-1]

    # Compute the relative importance using the eigenvalues.
    rel_importance = evalsh/sum(evalsh)

    # Compute the cumulative sum of the relative importance.
    cum_importance = np.cumsum(rel_importance)

    # Update the cumulative importance results to store the floating point values of upto 3 decimal point precision.
    cum_importance = [float('%.3f' % elem) for elem in cum_importance]

    # Update the eigenvector results to store the floating point values of upto 3 decimal point precision.
    evectsh = [[float('%.3f' % elem) for elem in row] for row in evectsh]

    # Store the eigenvectors in the pandas dataframe
    evectsh = pd.DataFrame(np.array(evectsh), columns=list(data.columns))

    # Set the option to display all columns in the pandas dataframe.
    pd.set_option('display.max_columns', None)

    # Print the cumulative importance
    print("\tCumulative importance: ", cum_importance)

    # Print the largest two eigenvalues
    print("\tLargest two eigenvalues are: ", evalsh[:2])

    # Print the eigenvectors associated with the largest two eigenvalues.
    print("\nEigen vectors: \n", evectsh.iloc[:2])

    # print("Eigen vectors transpose: \n", evectsh.iloc[:4].transpose())
    # print("Mean subtracted values: \n", mean_subtracted_data)
    # change to [:2]

    # Project the data on the eigenvectors by computing the dot product
    projected_data = mean_subtracted_data.dot(evectsh.iloc[:2].transpose())

    # Add column of classes to the projected data
    projected_data["Class"] = target_class

    # print("Projected mean subtracted data: \n", projected_data)

    # Plot the scatter-plot for the projected data.
    plot_scatter_plot(projected_data, 0, 1)


def plot_scatter_plot(projected_data, component_1, component_2):
    """
    This function plots the scatterplot for the input projected data.
    :param projected_data: input data to plot
    :param component_1: Component 1 Name
    :param component_2: Component 2 Name
    """
    # Create square shaped figure
    fig = plt.figure(figsize=(8, 8))
    # create a subplot with option nrows=1, ncols=1, index=1
    ax = fig.add_subplot(1, 1, 1)
    # Label for x axis
    ax.set_xlabel('Principal Component '+str(component_1), fontsize=15)
    # Label for y axis
    ax.set_ylabel('Principal Component '+str(component_2), fontsize=15)
    # Figure title
    ax.set_title('2D Principal Component Analysis between '+str(component_1)+" and "+str(component_2), fontsize=20)
    # Target classes
    targets = ['Assam', 'Bhuttan']
    # Colors list
    colors = ['r', 'b']
    # simultaneously iterate over class and colors
    for target, color in zip(targets, colors):
        # check for the selected target class
        indicesToKeep = projected_data['Class'] == target
        # draw the points for the selected target class on the scatter plot using the selected color
        ax.scatter(projected_data.loc[indicesToKeep, component_1]
                   , projected_data.loc[indicesToKeep, component_2]
                   , c=color
                   , alpha=0.5
                   , s=50)
    # add the legends to identify 'Assam' and 'Bhuttan' on the plot.
    ax.legend(targets)
    # Display the grid
    ax.grid()
    # show the plot
    plt.show()


def classify_data(training_data, test_data, best_features):
    """
    This function classifies the data using the best performing classifier.
    It then outputs the results on a file "HW_09_Classifed_Results.csv" with '-1' for Assam and '+1' for Bhuttan.
    i.e., (in this case) LDA using 'ShagFactor' and 'ApeFactor'.

    :param training_data: input training data to train the best performing LDA model.
    :param test_data: input data to classify using the best performing model.
    :param best_features: List of best features used by our best performing model.
    """

    test_data = generate_features(test_data)  # Generate features for the test data.
    train_target_class = training_data[['Class']]  # Extract the class labels from the training data.
    train_data = training_data[best_features]  # Select only the data for the best features from the training data.

    # Open the output file in write mode
    with open("HW_09_Classifed_Results.csv", 'w') as result_file:
        # Instantiate the LDA model.
        lda_model = LDA()

        # Train the model using the training data
        lda_model.fit(train_data, train_target_class.values.ravel())

        # Predict the classes for the test data.
        predictions = lda_model.predict(test_data[best_features])

        # Iterate over each prediction and write the results on the file.
        for predicted_class in predictions:
            if predicted_class == 'Assam':  # write -1 for Assam
                result_file.write('-1\n')
            elif predicted_class == 'Bhuttan':  # Write +1 for Bhuttan
                result_file.write('+1\n')
            else:
                result_file.write("\n")


def cross_correlation(dataframe):
    """
    This function generates the cross correlation between each feature and class.
    :param dataframe: pandas dataframe of data
    :return: dictionary of feature : co-efficients
    """
    # Getting feature list from dataframe
    features = [feature for feature in dataframe.columns if feature != 'Class']
    cc_coefficient_dict = dict()
    for feature in features:
        # Generating co-efficient for each feature with respect to cross-correlation with class
        cc_coefficient_dict[feature] = dataframe[feature].corr(dataframe['Class']).round(3)
    return cc_coefficient_dict


def feature_selection_using_cc(dataframe):
    """
    This function selects the feature1 and feature 2 from cross correlation between each feature with the class.
    :param dataframe: Pandas dataframe of data
    :return: Feature 1 and Feature 2
    """
    # Cross Correlation of all features with respect to class feature
    cc_coefficient_dict = cross_correlation(dataframe)
    print("\nCross correlation with Class:")
    for key, val in cc_coefficient_dict.items():
        print(key,':',val)

    # Find feature 1 (Highest correlated with class : either largest positive or negative)
    feature1 = sorted(cc_coefficient_dict, key=lambda dict_key: abs(cc_coefficient_dict[dict_key]))[-1]
    feature1_coeff = cc_coefficient_dict[feature1]
    print("\nFor Feature 1:")
    print("\tHighest absolute cross correlation coefficient:",feature1 ,feature1_coeff)

    # Find Feature 2
    # if feature 1 is positive, find feature with most negative CC as feature 2
    if feature1_coeff > 0:
        # Most negative coefficient
        feature2 = min(cc_coefficient_dict, key=cc_coefficient_dict.get)
    else:
        # Most positive coefficient
        feature2 = max(cc_coefficient_dict, key=cc_coefficient_dict.get)
    feature2_coeff = cc_coefficient_dict[feature2]
    print("\nFor Feature 2:")
    print("\tCross Correlation coefficient pair that is most different from Feature 1:", feature2, feature2_coeff)
    return feature1, feature2


def generate_scatterplot(dataframe, feature1, feature2):
    """
    This function generates the scatter plots for data of feature 1 and 2
    provided to it. It also generates the projection vector and decision boundary
    to classify the data according the feature 1 and 2 into two classes.
    Classes here are Assam == -1 and Bhuttan == 1.
    :param dataframe: Pandas dataframe of data
    :param feature1: Best Feature based on CC with class
    :param feature2: 2nd Best Feature based on CC with class
    :return: None
    """
    x_assam = []
    x_bhuttan = []
    y_assam = []
    y_bhuttan = []
    # Getting feature 1 and 2 values
    for _, row in dataframe.iterrows():
        # For Assam
        if row['Class'] == -1:
            x_assam.append(row[feature1])
            y_assam.append(row[feature2])
        # For Bhuttan
        if row['Class'] == 1:
            x_bhuttan.append(row[feature1])
            y_bhuttan.append(row[feature2])

    # Q3. Building 2-Dimensional Classifier ###
    # Center of Assam Class
    x_assam_center = np.mean(x_assam).round(3)
    y_assam_center = np.mean(y_assam).round(3)
    # Center of Bhuttan Class
    x_bhuttan_center = np.mean(x_bhuttan).round(3)
    y_bhuttan_center = np.mean(y_bhuttan).round(3)
    # Center of projection vector between Assam and Bhuttan
    x_center_of_projection = ((x_bhuttan_center + x_assam_center) / 2).round(3)
    y_center_of_projection = ((y_bhuttan_center + y_assam_center) / 2).round(3)

    # Check if projection vector is vertical straight line
    if x_bhuttan_center-x_assam_center != 0:
        # If projection vector is not vertical straight line i.e., Y = m*(X-yx) + y
        # Calculate the slope of projection line i.e.,  m = y2 - y1 / x2 - x1
        slope = ((y_bhuttan_center - y_assam_center) / (x_bhuttan_center - x_assam_center)).round(3)
        # Slope of it's perpendicular line i.e.,  new_slope = (-1/m)
        db_slope = -np.reciprocal(slope) if slope != 0 else float('inf')
        # Generating x values for decision boundary
        new_x1 = np.linspace(x_center_of_projection - 2, x_center_of_projection + 2, 100)  # 3
        # Getting respective y values for decision boundary based on the
        # slope and two point form i.e., Y = new_slope*(X-x) + y
        new_y1 = (db_slope * (new_x1 - x_center_of_projection)) + y_center_of_projection
    else:
        # If projection vector is vertical straight line
        slope = float('inf')    # Slope of projection vector is Not Defined.
        db_slope = 0            # Slope of it's perpendicular bisector is 0 (Horizontal line)

    # Plotting all necessary things
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.grid()
    # Plotting Assams class data points with red color
    ax.scatter(x_assam, y_assam, c='red', alpha=0.5, edgecolors='none', label='Assam')
    # Plotting Bhuttan class data points with blue color
    ax.scatter(x_bhuttan, y_bhuttan, c='blue', alpha=0.5, edgecolors='none', label='Bhuttan')
    # Plotting Center of Assam class
    ax.plot(x_assam_center, y_assam_center, 'ko')
    # Plotting Center of Bhuttan class
    ax.plot(x_bhuttan_center, y_bhuttan_center, 'ko')
    # Plotting projection vector between centers of classes
    ax.plot([x_assam_center, x_bhuttan_center], [y_assam_center, y_bhuttan_center], 'y-', lw=2, label='Projection Vector')
    # Plotting center of projection
    ax.plot(x_center_of_projection, y_center_of_projection, 'ko')
    # Q3. Plotting Decision Boundary based on the slope of projection vector  ##
    if slope == float('inf'):
        # If projection vector is vertical line then plot decision boundary as horizontal line
        ax.axhline(y=y_center_of_projection, color='k', ls='--', lw=2, label='Decision Boundary')
    elif slope == 0:
        # If projection vector is horizontal line then plot decision boundary as vertical line
        ax.axvline(x=x_center_of_projection, color='k', ls='--', lw=2, label='Decision Boundary')
    else:
        # If projection vector is neither horizontal nor vertical then plot the decision boundary
        # based on the new x and y generated values
        ax.plot(new_x1, new_y1, 'k--', lw=2, label='Decision Boundary')

    # Q3. Testing classifier based on class predicted by decision boundary ##
    # Getting extreme coordinates of decision line
    lower_pt = [new_x1[0], new_y1[0]]
    higher_pt = [new_x1[-1], new_y1[-1]]

    # Dictionary for Assam and Bhuttan classified data points
    assam_classifier_dict = dict()
    bhuttan_classifier_dict = dict()
    points_on_classifier = 0

    # For all Assam and Bhuttan data points
    for ax, ay, bx, by in zip(x_assam, y_assam, x_bhuttan, y_bhuttan):
        # Get value for checking the side of data point with respect to line
        assam_classifier_dict[(ax,ay)] = ((higher_pt[0]-lower_pt[0])*(ay-lower_pt[1]) - (ax-lower_pt[0])*(higher_pt[1]-lower_pt[1])).round(1)
        bhuttan_classifier_dict[(bx,by)] = ((higher_pt[0]-lower_pt[0])*(by-lower_pt[1]) - (bx-lower_pt[0])*(higher_pt[1]-lower_pt[1])).round(1)
        # Count the points that lies on classifier line
        if (assam_classifier_dict[(ax,ay)] == 0.0) or (bhuttan_classifier_dict[(bx,by)] == 0.0):
            points_on_classifier += 1

    # Getting correctly classified Assam and Bhuttan counts
    correct_assam = classification_of_data(dataframe,feature1,feature2, assam_classifier_dict, -1)
    correct_bhuttan = classification_of_data(dataframe,feature1,feature2, bhuttan_classifier_dict, 1)
    print("Correctly classified Assam:",correct_assam,"Bhuttan:",correct_bhuttan,"Out of Total Data:", len(dataframe.index))
    print("Data Points on classifier line:",points_on_classifier)
    accuracy = (correct_assam + correct_bhuttan) / len(dataframe.index)
    print("Training Accuracy of classifier line",accuracy*100,"%")

    # Plotting names and all subplots on pyplot figure
    plt.title("Scatter Plot")   # Title of plot
    plt.xlabel(feature1)        # Label of X axis
    plt.ylabel(feature2)        # Label of Y axis
    plt.legend()                # Plot legends
    plt.show()                  # Show the plot


def classification_of_data(dataframe, feature1, feature2, classifier_dict, flag):
    """
    This function used in generating_scatterplot() function to get the count of correctly classified data points.
    Given a directed line from point p0(x0, y0) to p1(x1, y1), you can use the following condition to decide
    whether a point p2(x2, y2) is on the left of the line, on the right, or on the same line:
        value = (x1 - x0)(y2 - y0) - (x2 - x0)(y1 - y0)
            if value > 0, p2 is on the left side of the line. (Assam Class)
            if value = 0, p2 is on the same line.
            if value < 0, p2 is on the right side of the line. (Bhuttan Class)
    Reference: https://stackoverflow.com/questions/22668659/calculate-on-which-side-of-a-line-a-point-is
    :param dataframe: Pandas dataframe of data
    :param feature1: Best feature
    :param feature2: 2nd Best feature
    :param classifier_dict: Dictionary for Assam and Bhuttan classified data points
    :param flag: Flag -1 for Assam and 1 for Bhuttan.
    :return: correctly classified count
    """
    correct = 0
    # For each row in dataframe
    for _,row in dataframe.iterrows():
        if flag == -1:  # If function called for Assam class
            if row['Class'] == -1:
                if classifier_dict[(row[feature1], row[feature2])] > 0:  # If data point is on left side of line
                    correct += 1
        if flag == 1:   # If function called for Bhuttan class
            if row['Class'] == 1:
                if classifier_dict[(row[feature1], row[feature2])] < 0:  # If data point is on right side of line
                    correct += 1
    return correct


def main():
    """
    This function drives the execution of the program. It reads the data from the input files, generates the features,
    performs feature selection using cross-correlation and brute force technique, perform principal component analysis
    and plots the scatter plots wherever necessary.
    """
    # Read the input from the file.
    input_data = pd.read_csv("Abominable_Data_HW19_v420.csv")

    # Balance the input data.
    balanced_data = get_balanced_data(input_data)

    # Generate the features for the data.
    data = generate_features(balanced_data)

    factored_data = data.copy()
    # Replace Assam label with -1 and Bhuttan with +1
    factored_data["Class"].replace({"Assam": -1, "Bhuttan": +1}, inplace=True)
    # Q3. Feature Selection using cross correlation
    feature1, feature2 = feature_selection_using_cc(factored_data)
    # Scatter Plot, Decision boundary and Classifier
    generate_scatterplot(factored_data, feature1, feature2)

    # Extract the two best features for the best performing classification (LDA) models.
    best_feat_pair_1, accuracy_1, best_feat_pair_2, accuracy_2 = get_best_2_features_brute_force(data)

    print("\t{} and {}, when used with LDA, gives the best accuracy of {:.2f}%"
          "".format(best_feat_pair_1[0], best_feat_pair_1[1], accuracy_1*100))

    print("\t{} and {}, when used with LDA, gives the second best accuracy of {:.2f}%"
          "".format(best_feat_pair_2[0], best_feat_pair_2[1], accuracy_2 * 100))

    # Perform the principal component analysis on the data
    perform_principal_component_analysis(data)

    # Read the unclassified data from the validation data file
    test_data = pd.read_csv("Abominable_UNCLASSIFIED_Data_HW19_v420.csv")

    # Classify the data using the best performing classifier
    classify_data(data, test_data, best_feat_pair_1)


if __name__ == '__main__':
    main()
