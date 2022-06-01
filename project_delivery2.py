import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from datetime import datetime

import iso8601
df = pd.read_excel('IssueTickets.ods', engine='odf')
df = df.drop(['ISSUE_ID', 'JIRANAME', 'WORKER'], axis=1)
creation_date = df['CREATION_DATE']
res_date = df['RESOLUTION_DATE']

cre_year_list = []
cre_month_list = []
cre_day_list = []

res_year_list = []
res_month_list = []
res_day_list = []

solved_time_list = []
for i in range(len(creation_date)):
    # timestamp = 1625309472.357246
    # convert to datetime
    date_time = str(creation_date[i]).split('-')
    date_time_hour = date_time[2].split(' ')
    cre_year_list.append(date_time[0])
    cre_month_list.append(date_time[1])
    cre_day_list.append(date_time_hour[0])


for i in range(len(res_date)):
    # timestamp = 1625309472.357246
    # convert to datetime
    date_time = str(res_date[i]).split('-')
    date_time_hour = date_time[2].split(' ')
    res_year_list.append(date_time[0])
    res_month_list.append(date_time[1])
    res_day_list.append(date_time_hour[0])

df['cre_year_list'] = cre_year_list
df['cre_month_list'] = cre_month_list
df['cre_day_list'] = cre_day_list

df['res_year_list'] = res_year_list
df['res_month_list'] = res_month_list
df['res_day_list'] = res_day_list
for i in range(len(cre_year_list)):
    solved_year = int(res_year_list[i]) - int(cre_year_list[i])
    solved_month = int(res_month_list[i]) - int(cre_month_list[i])
    solved_day = int(res_day_list[i]) - int(cre_day_list[i])

    if solved_day < 0:
        solved_month -= 1
        solved_day += 30
    if solved_month < 0:
        solved_month += 12

    solved_time_list.append(str(solved_year) + 'Y-'+str(solved_month) + 'M-' + str(solved_day)+'D')

    print(f'year: {solved_year} month: {solved_month} day: {solved_day}')

# print(df['CREATION_DATE'])
df['solved_time'] = solved_time_list
print(df)
