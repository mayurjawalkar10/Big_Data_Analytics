import sys
import pandas as pd


def main():
    """
    This function predicts if the person will get sick based on the given data.
    """
    if len(sys.argv) < 2:
        print('Filename not provided. Exiting..')
        sys.exit(1)

    validation_filename = sys.argv[1]
    validation_data = pd.read_csv(validation_filename)

    for index, row in validation_data.iterrows():
        if row['PeanutButter'] in [0]:
            print(0)
        elif row['PeanutButter'] in [1]:
            print(1)
        else:
            print()


if __name__ == '__main__':
    """
    Run if executed as a script.
    """
    main()
