# Calculate Feature importance

# random forest for feature importance on a classification problem
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset/25_projects_PRs.csv", sep=',', encoding='utf-8')

df['src_churn'] = df['Additions'] + df['Deletions']
df['num_comments'] = df['Review_Comments_Count'] + df['Comments_Count']
df['is_God_Class'] = df['GodClass'].apply(lambda x: 1 if x>0 else 0)
df['is_Data_Class'] = df['DataClass'].apply(lambda x: 1 if x>0 else 0)
df['is_Long_Method'] = df['ExcessiveMethodLength'].apply(lambda x: 1 if x>0 else 0)
df['is_Long_Parameter_List'] = df['ExcessiveParameterList'].apply(lambda x: 1 if x>0 else 0)
df.loc[(df['GodClass'] > 0) | (df['DataClass'] > 0) | (df['ExcessiveMethodLength'] > 0) |
       (df['ExcessiveParameterList'] > 0), 'is_smelly'] = 1
df.loc[df['is_smelly'].isnull(), 'is_smelly'] = 0

# Previous work features
accept_baseline = ['src_churn', 'Commits_PR', 'Files_Changed', 'num_comments','Followers','Participants_Count',
                   'Team_Size', 'File_Touched_Average', 'Commits_Average', 'Prev_PRs', 'is_smelly', #'Project_Size',
                   'User_Accept_Rate', 'PR_Time_Created_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At',
                   'PR_Date_Created_At', 'Project_Name', 'PR_accept']

df = df[accept_baseline]
target = 'is_smelly'
predictors = [x for x in df.columns if x not in [target, 'PR_Date_Created_At', 'PR_Time_Created_At', 'PR_Date_Closed_At',
                                                 'PR_Time_Closed_At', 'Project_Name', 'PR_accept']]
df = df.dropna()

X = df[predictors]
y = df[target]

# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_


#do code to support model
#"data" is the X dataframe and model is the SKlearn object
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(df.columns, model.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Feature_importance'})
importances = importances.head(-1)

sns.set(rc={"figure.figsize":(20, 10)}) #width=3, #height=4
sns.set(font_scale=2)  # crazy big
plt.xticks(rotation=45)
sns.barplot(data=importances.sort_values(by='Feature_importance'), x=importances.index, y="Feature_importance", palette=sns.color_palette(["#477ba8"]))