from datetime import datetime as dt

# Print the personal information.
print("Jawalkar Mayur")
print("I Love Driving Car.\n")

# Print the creation date.
print("Creation Time : 08/02/2020 04:24:56\n")

# Print the current date.
print("Current DateTime:", dt.now().strftime("%d/%m/%Y %H:%M:%S")+"\n")

# Print the row and column count.
with open("../A_DATA_FILE.csv", "r+") as file:
    row_cnt = col_cnt = 0
    all_rows = file.readlines()
    for row in all_rows:
        if len(row.strip()) > 0:
            row_cnt += 1 
    if len(all_rows) > 0:
        for col in all_rows[0].strip().split(","):
            if len(col.strip()) > 0:
                col_cnt += 1 
print("Columns :\t" + str(col_cnt) + "\n" + "Rows :\t\t" + str(row_cnt))
