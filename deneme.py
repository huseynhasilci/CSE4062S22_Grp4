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

df = pd.read_excel('IssueTickets.ods', engine='odf')
#df = df.drop(['ISSUE_ID', 'ISSUE_TYPE', 'PRIORITY', 'JIRANAME', 'WORK_LOG_TOTAL', 'WORK_LOG_RATIO', 'ISSUE_CATEGORY', 'ISSUE_SUB_CATEGORY', 'LABEL', 'CREATION_DATE', 'RESOLUTION_DATE'], axis = 1)
df = df.drop(['ISSUE_ID', 'REPORTER', 'JIRANAME', 'COMPNAME', 'WORK_LOG', 'WORK_LOG_TOTAL', 'WORK_LOG_RATIO', 'ISSUE_CATEGORY', 'ISSUE_SUB_CATEGORY', 'LABEL', 'CREATION_DATE', 'RESOLUTION_DATE'], axis=1)
le = LabelEncoder()
for column in df.columns:
    temp_new = le.fit_transform(df[column].astype('category'))
    df.drop(labels=[column], axis="columns", inplace=True)
    df[column] = temp_new

X = df.iloc[:, :]
y = df.iloc[:, 1:2]

print(y)
X = X.drop(['PRIORITY'], axis=1)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# classifier = DecisionTreeClassifier()
# classifier = LogisticRegression(solver='lbfgs', max_iter=3000)
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    LogisticRegression(solver='lbfgs', max_iter=3000)

]
for i in classifiers:
    classifier = i

    classifier.fit(X_train, np.ravel(y_train))

    y_pred = classifier.predict(X_test)
    print(classifier.score(X_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    #print(precision_score(y_test, y_pred, average='macro', zero_division=1))
    #print(precision_score(y_test, y_pred, average='micro', zero_division=1))
    #print(precision_score(y_test, y_pred, average='weighted', zero_division=1))
    #print(precision_score(y_test, y_pred, average=None, zero_division=1))
    print(classification_report(y_test, y_pred, zero_division=1))
