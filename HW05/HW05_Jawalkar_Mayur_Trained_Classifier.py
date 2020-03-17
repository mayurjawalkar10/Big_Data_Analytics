# Import statements
import sys
import pandas as pd


def main():
    """
    This function predicts the class of abominable snowfolks data.
    """
    if len(sys.argv) < 2:
        print('Filename not provided. Exiting..')
        sys.exit(1)

    validation_filename = sys.argv[1]
    validation_data = pd.read_csv(validation_filename)

    with open('HW05_Jawalkar_Mayur__MyClassifications.csv', 'w', encoding='utf8') as file:
        for index, row in validation_data.iterrows():
            if float(row['BangLn']) <= 6.0:
                print('-1')
                file.write('-1\n')
            elif float(row['EarLobes']) <= 0.0:
                print('+1')
                file.write('+1\n')
            elif float(row['BangLn']) <= 5.0:
                print('-1')
                file.write('-1\n')
            elif float(row['Reach']) <= 139.0:
                print('+1')
                file.write('+1\n')
            elif float(row['TailLn']) <= 13.0:
                print('+1')
                file.write('+1\n')
            elif float(row['Reach']) <= 142.0:
                print('+1')
                file.write('+1\n')
            elif float(row['Age']) <= 42.0:
                print('-1')
                file.write('-1\n')
            else:
                print('-1')
                file.write('-1\n')


if __name__ == '__main__':
    """
    Run if executed as a script.
    """
    main()
