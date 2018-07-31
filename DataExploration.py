import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

str_FilePathTrain = 'Data/train.csv'
str_FilePathTest = 'Data/test.csv'

df_TrainingData = pd.read_csv(str_FilePathTrain, index_col=0)
df_TestData = pd.read_csv(str_FilePathTest, index_col=0)

df_Data = pd.concat([df_TrainingData, df_TestData], sort=False)

# print(df_Data.head())
# print(df_Data.info())
# print(df_Data.describe())


# Exploratory Analysis + DataCleaning
# It's so much faster in Excel with PivotTables and PivotCharts...admittedly this is a small data set
# Based on simple analysis using Excel, key variables that seem to affect survivability are:
#     age, PClass, Fare, Sex, Embarked.
# Need to dummify and impute these columns, then do a heatmap to justify correlations.


# Out of these, the fare, age and Embarked are missing data.

# Impute missing ages
# Forums on Kaggle suggest to estimate age based on mean of similar titles
def Titanic_get_title(str_passenger_name):
    """Returns the title from the passenger name in the Titanic dataset.
    The title is squished between ", " and "." """
    first_part, second_part = str(str_passenger_name).split(", ", 1)
    title, third_part =str(second_part).split(".", 1)
    if title in ['Mr', 'Mrs', 'Miss']:
        return str(title)
    else:
        return "Rare"

df_Data['Title'] = df_Data['Name'].apply(Titanic_get_title)
list_Titles = ['Mr', 'Mrs', 'Miss', 'Rare']
dict_Title_MeanAge = {title:np.mean(df_Data['Age'][df_Data['Title'] == title]) for title in list_Titles}

df_Data.Age = df_Data.Age.fillna(df_Data.Title.map(dict_Title_MeanAge))

# Impute embarked 
# Based on forums, they suggest to assign Embarked to its mode by class.
def get_series_mode(pseries):
    ar_unique = pseries.unique()
    dict_count_unique = {sum(pseries == uniq):uniq for uniq in ar_unique}
    return dict_count_unique[max(dict_count_unique.keys())]

list_Pclass = [1, 2, 3]
dict_Pclass_ModeEmbarked = {pclass:get_series_mode(df_Data['Embarked'][df_Data['Pclass'] == pclass]) for pclass in list_Pclass}

df_Data.Embarked = df_Data.Embarked.fillna(df_Data.Pclass.map(dict_Pclass_ModeEmbarked))

# Impute Fare
# One of the test rows is missing its fare. A reasonable assumption would be to take the mean of its Pclass.
dict_Pclass_MeanFare = {pclass:np.mean(df_Data['Fare'][df_Data['Pclass'] == pclass]) for pclass in list_Pclass}
df_Data.Fare = df_Data.Fare.fillna(df_Data.Pclass.map(dict_Pclass_MeanFare))


# Drop unused columns Name, Cabin, Ticket
df_Data = df_Data.drop(columns=['Name', 'Cabin', 'Ticket'])

# Prepare dummified string-Columns
df_Data = pd.get_dummies(df_Data, drop_first=True)

# print(df_Data.head())
# print(df_Data.info())
# print(df_Data.describe())

#Split them back again and output to cleaned
df_TrainingData = df_Data.iloc[:891, :]
df_TestData = df_Data.iloc[891:, :]

# print(df_TrainingData.head())
# print(df_TrainingData.info())
# print(df_TrainingData.describe())

# print(df_TestData.head())
# print(df_TestData.info())
# print(df_TestData.describe())

df_TrainingData.to_csv("Data/train_cleaned.csv")

# df_TestData = df_TestData.drop(columns='Survived')
df_TestData.to_csv("Data/test_cleaned.csv")


#matrix_CorrCoef = df_TrainingData.corr(method='pearson')
#sns.heatmap(matrix_CorrCoef, center=0, vmin=-1, vmax=1, cmap='bwr_r')


#plt.figure(1)

#sns.countplot(x='Sex', hue='Survived', data=df_TrainingData)

#plt.show()


