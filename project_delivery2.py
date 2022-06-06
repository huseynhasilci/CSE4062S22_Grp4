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
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from datetime import datetime
import matplotlib.pyplot as plt
classifiers = [
    KNeighborsClassifier(10),
    SVC(kernel="linear", C=0.35),
    SVC(gamma=3, C=2),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=20),
    RandomForestClassifier(max_depth=20, n_estimators=10, max_features=1),
    MLPClassifier(alpha=5, max_iter=3000),
    AdaBoostClassifier(),
    GaussianNB(),
    LogisticRegression(solver='lbfgs', max_iter=3000),
    QuadraticDiscriminantAnalysis()

]
classifiers_names = ['KNN', 'SVC(linear)', 'SVC', 'Decision Tree Classifier', 'Random Forest Classifier',
                     'MLP Classifier', 'AdaBoostClassifier', 'GaussianNB', 'LogisticRegression',
                     'QuadraticDiscriminantAnalysis']
# import iso8601
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

    # print(f'year: {solved_year} month: {solved_month} day: {solved_day}')


# print(df['CREATION_DATE'])
df['solved_time'] = solved_time_list
le = LabelEncoder()
for column in df.columns:
    temp_new = le.fit_transform(df[column].astype('category'))
    df.drop(labels=[column], axis="columns", inplace=True)
    df[column] = temp_new
X = df[['REPORTER', 'ISSUE_TYPE', 'PRIORITY', 'COMPNAME', 'EMPLOYEE_TYPE', 'WORK_LOG', 'WORK_LOG_TOTAL','solved_time']]
#X = df[['ISSUE_TYPE', 'PRIORITY','EMPLOYEE_TYPE']]
y = df['ISSUE_CATEGORY']

print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

for c in range(len(classifiers)):
    classifier = classifiers[c]

    classifier.fit(X_train, np.ravel(y_train))

    y_pred = classifier.predict(X_test)
    print(classifier.score(X_test, y_test))
    """conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(f'Confusion Matrix {classifiers_names[c]}', fontsize=18)
    plt.show()"""
    #print(precision_score(y_test, y_pred, average='macro', zero_division=1))
    #print(precision_score(y_test, y_pred, average='micro', zero_division=1))
    #print(precision_score(y_test, y_pred, average='weighted', zero_division=1))
    #print(precision_score(y_test, y_pred, average=None, zero_division=1))
    print(classifiers_names[c])
    print(classification_report(y_test, y_pred, zero_division=1))
#print(df)