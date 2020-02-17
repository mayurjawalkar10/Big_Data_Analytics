"""
# Author: Mayur Sunil Jawalkar
# E-mail: mj8628@rit.edu
# HW00 - Big Data Analytics
"""

# Import Statements
from datetime import datetime


class Mentor:
    """
    This is a meta-program. This class creates another python program to display personal information.
    The generated program is saved with 'HW_00_MSJ_Trained.py'.
    """
    __slots__ = 'indent'

    def __init__(self):
        """
        Initialize the indent variable to use throughout the program.
        """
        self.indent = "    "

    def t_imports(self):
        """
        Add the import statements to the trained program.
        """
        import_text = "from datetime import datetime as dt\n\n"
        return import_text

    def t_personal_info(self):
        """
        Adds personal info to the trained program.
        """
        comment = "# Print the personal information.\n"
        personal_info_text = "print(\"Jawalkar Mayur\")\n" \
                             "print(\"I Love Driving Car.\\n\")\n\n"
        return comment+personal_info_text

    def t_creation_time(self):
        """
        Add the creation date of the newly generated trained program in it.
        """
        comment = "# Print the creation date.\n"
        file_creation_time_text = "print(\"Creation Time : " + datetime.now().strftime('%d/%m/%Y %H:%M:%S') \
                                  + "\\n\")\n\n"
        return comment+file_creation_time_text

    def t_curr_time(self):
        """
        Add the code to print current time in the trained program.
        """
        comment = "# Print the current date.\n"
        todays_datetime_text = "print(\"Current DateTime:\", dt.now().strftime(\"%d/%m/%Y %H:%M:%S\")+\"\\n\")\n\n"
        return comment+todays_datetime_text

    def t_csv_row_col_cnt(self):
        """
        Add the code to print the row and column count in the trained program.
        """
        com_file = "# Print the row and column count.\n"
        csv_file_text = "with open(\"../A_DATA_FILE.csv\", \"r+\") as file:\n" \
                        + self.indent+"row_cnt = col_cnt = 0\n"\
                        + self.indent+"all_rows = file.readlines()\n"\
                        + self.indent+"for row in all_rows:\n"\
                        + self.indent*2+"if len(row.strip()) > 0:\n"\
                        + self.indent*3+"row_cnt += 1 \n"\
                        + self.indent+"if len(all_rows) > 0:\n"\
                        + self.indent*2+"for col in all_rows[0].strip().split(\",\"):\n"\
                        + self.indent*3+"if len(col.strip()) > 0:\n"\
                        + self.indent*4+"col_cnt += 1 \n"\
                        + "print(\"Columns :\\t\" + str(col_cnt) + \"\\n\" + \"Rows :\\t\\t\" + str(row_cnt))\n"
        return com_file+csv_file_text

    def write_trainer(self):
        """
        Open the file and write all the content to the trained program.
        """
        with open("HW_00_MSJ_Trained.py", "w+", encoding="utf-8") as train_file:
            train_file.write(self.t_imports())
            train_file.write(self.t_personal_info())
            train_file.write(self.t_creation_time())
            train_file.write(self.t_curr_time())
            train_file.write(self.t_csv_row_col_cnt())


if __name__ == '__main__':
    Mentor().write_trainer()
