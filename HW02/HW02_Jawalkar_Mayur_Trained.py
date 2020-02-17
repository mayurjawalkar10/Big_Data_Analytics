import sys


def main():
    """
    This function handles the functionality of 1D classifier.
    """
    if len(sys.argv) < 2:
        print("Enter input data filename as a command line argument.")
        sys.exit(1)

    validation_filename = sys.argv[1]
    attribute_id = 1
    best_threshold = 135
    side = 'left'
    with open(validation_filename, "r") as file:
        # Read all rows of the file and iterate from the second row.
        # Ignore the first row containing header.
        for row in file.readlines()[1:]:
            # If the row is not empty add data point in the list.
            cols = row.strip().split(",")
            # check if validation data has age and height information
            if len(cols) >= 2:
                # print +1 for all attributes greater than the best_threshold
                if float(cols[attribute_id].strip()) > best_threshold:
                    class_val = "+1" if side == "right" else "-1"
                    print(class_val)  # Positive Class - Bhutan
                # print -1 for all attributes less than or equal to the best_threshold
                else:
                    class_val = "-1" if side == "right" else "+1"
                    print(class_val)  # Negative Class - Assam


if __name__ == '__main__':
    """
    Run if executed as a script.
    """
    main()
