import xlrd
import csv
import pandas as pd

df = pd.read_excel("/home/lva/Bureau/dataScientest/projet_pompiers_Londres/data/Mobilisation.xlsx") 
df.to_csv("/home/lva/Bureau/dataScientest/projet_pompiers_Londres/data/Mobilisation.csv", sep=",")


#wb = xlrd.open_workbook('/home/lva/Bureau/dataScientest/projet_pompiers_Londres/data/Mobilisation.xlsx')
#sh = wb.sheet_by_name('Sheet1')
#your_csv_file = open('/home/lva/Bureau/dataScientest/projet_pompiers_Londres/data/Mobilisation.csv', 'w')
#wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

#for rownum in range(sh.nrows):
#	wr.writerow(sh.row_values(rownum))

#your_csv_file.close()
