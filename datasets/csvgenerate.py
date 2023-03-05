import csv
import os

with open('blackened.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for name in os.listdir('blackened'):

        writer.writerow([name])