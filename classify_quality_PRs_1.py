#!/usr/bin/env python
# coding: utf-8


import numpy as np
np.logspace(0,-9, num=100) 



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from scipy.stats import uniform
from tensorflow import keras
import tensorflow as tf
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
import xgboost as xgb
import operator
from xgboost import plot_importance
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import time as t
import seaborn as sns


df = pd.read_csv("Dataset/25_projects_PRs.csv", sep=',', encoding='utf-8')


# #for HP tuning
# def get_classifiers_without_params():
#     return {
#         'MLP': MLPClassifier(max_iter=100),
#         'RandomForest': RandomForestClassifier(bootstrap=False, class_weight='balanced'),
#         'LinearSVC': LinearSVC(max_iter=2000),
#         'LogisticRegression': LogisticRegression(multi_class='auto', max_iter=1200),
#         'XGBoost': xgb.XGBClassifier(objective = 'binary:logistic', eval_metric = 'auc', seed = 201703, missing = 1),
#         'BaggedDT': BaggingClassifier(),
#         'NaiveBayes': GaussianNB(),
#         'KNN': KNeighborsClassifier()
        
#     }


# # Train with HP tuning
def get_classifiers():
    return {
        'randomforest': randomforestclassifier(n_jobs=-1, n_estimators=1000, max_features='sqrt',bootstrap=false, class_weight='balanced'),
        'linearsvc': linearsvc(max_iter=2000,c = 1),
        'logisticregression': logisticregression(c = 100, penalty = 'l2', solver= 'lbfgs', n_jobs=4, multi_class='auto', max_iter=1200),
        'xgboost': xgb.xgbclassifier(**params), #max_depth=11, min_child_weight=9,
        'mlp': mlpclassifier(max_iter=100, activation='tanh', alpha=0.0001, hidden_layer_sizes=(50, 100, 50), learning_rate='adaptive', solver='adam')
        'baggeddt': baggingclassifier(n_estimators = 1000),
        'naivebayes': gaussiannb(var_smoothing = 1.00e-09),
        'knn': kneighborsclassifier(metric = 'manhattan', n_neighbors=19, weights='distance') 
    }

# params = {
#     'objective': 'binary:logistic',
#     'colsample_bytree': 0.2,
#     'max_depth': 9,
#     'gamma': 0.1,
#     'verbose_eval': True,
#     'eval_metric': 'auc',
#     'seed': 201703,
#     'missing':-1,
#     'learning_rate': 0.1,
#     'n_estimators': 150,   
# }

# # Train without HP tuning
# def get_classifiers():
#     return {
#         'RandomForest': RandomForestClassifier(bootstrap=False, class_weight='balanced'),          
#         'LinearSVC': LinearSVC(max_iter=2000),
#         'LogisticRegression': LogisticRegression(multi_class='auto', max_iter=1200),
#         'XGBoost': xgb.XGBClassifier(objective = 'binary:logistic', eval_metric = 'auc', seed = 201703, missing = 1),
#         'MLP': MLPClassifier(max_iter=100),    
#         'BaggedDT': BaggingClassifier(),
#         'NaiveBayes': GaussianNB(),
#         'KNN': KNeighborsClassifier()
#     }




def encode_labels(df1, column_name):
    encoder = LabelEncoder()
    df1[column_name] = [str(label) for label in df1[column_name]]
    encoder.fit(df1[column_name])
    one_hot_vector = encoder.transform(df1[column_name])
    return one_hot_vector

df['Language'] = encode_labels(df, 'Language')
# df['Project_Domain'] = encode_labels(df, 'Project_Domain')
df['src_churn'] = df['Additions'] + df['Deletions']
df['num_comments'] = df['Review_Comments_Count'] + df['Comments_Count']
df['is_God_Class'] = df['GodClass'].apply(lambda x: 1 if x>0 else 0)
df['is_Data_Class'] = df['DataClass'].apply(lambda x: 1 if x>0 else 0)
df['is_Long_Method'] = df['ExcessiveMethodLength'].apply(lambda x: 1 if x>0 else 0)
df['is_Long_Parameter_List'] = df['ExcessiveParameterList'].apply(lambda x: 1 if x>0 else 0)
df.loc[(df['GodClass'] > 0) | (df['DataClass'] > 0) | (df['ExcessiveMethodLength'] > 0) |
       (df['ExcessiveParameterList'] > 0), 'is_smelly'] = 1
df.loc[df['is_smelly'].isnull(), 'is_smelly'] = 0

project_features = ['Project_Age', 'Team_Size', 'Stars', 'File_Touched_Average', 'Forks_Count', 'Watchers', 'Language',
                    'Project_Domain', 'Contributor_Num', 'Comments_Per_Closed_PR', 'Additions_Per_Week', 'Deletions_Per_Week',
                    'Merge_Latency', 'Comments_Per_Merged_PR', 'Churn_Average', 'Close_Latency', 'Project_Accept_Rate',
                    'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Workload', 'Commits_Average',
                    'Open_Issues', 'PR_Time_Created_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At', 'PR_Date_Created_At',
                    'Project_Name', 'PR_accept']
PR_features = ['Intra_Branch', 'Assignees_Count', 'Label_Count', 'Files_Changed', 'Contain_Fix_Bug', 'Wait_Time', 'Day',
               'src_churn', 'Deletions', 'Commits_PR', 'first_response_time', 'first_response',
               'latency_after_first_response', 'conflict', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6', 'X1_7',
               'X1_8', 'X1_9', 'PR_Latency', 'title_words_count', 'body_words_count','Point_To_IssueOrPR', 'PR_Time_Created_At',
               'PR_Date_Closed_At', 'PR_Time_Closed_At', 'PR_Date_Created_At', 'Project_Name', 'PR_accept']
integrator_features = ['Participants_Count', 'num_comments', 'Last_Comment_Mention',
                       'line_comments_count', 'comments_reviews_words_count', 'PR_Time_Created_At', 'PR_Date_Closed_At',
                       'PR_Time_Closed_At', 'PR_Date_Created_At', 'Project_Name', 'PR_accept']
contributor_features = ['Followers', 'Closed_Num', 'Contributor', 'Public_Repos', 'Organization_Core_Member',
                        'Contributions', 'User_Accept_Rate', 'Accept_Num', 'Closed_Num_Rate', 'Prev_PRs', 'Following',
                        'PR_Time_Created_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At', 'PR_Date_Created_At',
                        'Project_Name', 'PR_accept']
quality_features = ['AbstractClassWithoutAbstractMethod', 'AccessorClassGeneration', 'AccessorMethodGeneration',
                     'ArrayIsStoredDirectly', 'AvoidMessageDigestField', 'AvoidPrintStackTrace',
                     'AvoidReassigningLoopVariables', 'AvoidReassigningParameters', 'AvoidStringBufferField',
                     'AvoidUsingHardCodedIP', 'CheckResultSet', 'ConstantsInInterface',
                     'DefaultLabelNotLastInSwitchStmt', 'DoubleBraceInitialization', 'ForLoopCanBeForeach',
                     'ForLoopVariableCount', 'GuardLogStatement', 'JUnit4SuitesShouldUseSuiteAnnotation',
                     'JUnit4TestShouldUseAfterAnnotation', 'JUnit4TestShouldUseBeforeAnnotation',
                     'JUnit4TestShouldUseTestAnnotation', 'JUnitAssertionsShouldIncludeMessage',
                     'JUnitTestContainsTooManyAsserts', 'JUnitTestsShouldIncludeAssert', 'JUnitUseExpected',
                     'LiteralsFirstInComparisons', 'LooseCoupling', 'MethodReturnsInternalArray', 'MissingOverride',
                     'OneDeclarationPerLine', 'PositionLiteralsFirstInCaseInsensitiveComparisons',
                     'PositionLiteralsFirstInComparisons', 'PreserveStackTrace', 'ReplaceEnumerationWithIterator',
                     'ReplaceHashtableWithMap', 'ReplaceVectorWithList', 'SwitchStmtsShouldHaveDefault',
                     'SystemPrintln', 'UnusedFormalParameter', 'UnusedImports', 'UnusedLocalVariable',
                     'UnusedPrivateField', 'UnusedPrivateMethod', 'UseAssertEqualsInsteadOfAssertTrue',
                     'UseAssertNullInsteadOfAssertTrue', 'UseAssertSameInsteadOfAssertTrue',
                     'UseAssertTrueInsteadOfAssertEquals', 'UseCollectionIsEmpty', 'UseTryWithResources', 'UseVarargs',
                     'WhileLoopWithLiteralBoolean', 'AbstractNaming', 'AtLeastOneConstructor', 'AvoidDollarSigns',
                     'AvoidFinalLocalVariable', 'AvoidPrefixingMethodParameters', 'AvoidProtectedFieldInFinalClass',
                     'AvoidProtectedMethodInFinalClassNotExtending', 'AvoidUsingNativeCode', 'BooleanGetMethodName',
                     'CallSuperInConstructor', 'ClassNamingConventions', 'CommentDefaultAccessModifier',
                     'ConfusingTernary', 'ControlStatementBraces', 'DefaultPackage', 'DontImportJavaLang',
                     'DuplicateImports', 'EmptyMethodInAbstractClassShouldBeAbstract', 'ExtendsObject',
                     'FieldDeclarationsShouldBeAtStartOfClass', 'FieldNamingConventions', 'ForLoopShouldBeWhileLoop',
                     'ForLoopsMustUseBraces', 'FormalParameterNamingConventions', 'GenericsNaming',
                     'IdenticalCatchBranches', 'IfElseStmtsMustUseBraces', 'IfStmtsMustUseBraces', 'LinguisticNaming',
                     'LocalHomeNamingConvention', 'LocalInterfaceSessionNamingConvention', 'LocalVariableCouldBeFinal',
                     'LocalVariableNamingConventions', 'LongVariable', 'MDBAndSessionBeanNamingConvention',
                     'MethodArgumentCouldBeFinal', 'MethodNamingConventions', 'MIsLeadingVariableName', 'NoPackage',
                     'UseUnderscoresInNumericLiterals', 'OnlyOneReturn', 'PackageCase', 'PrematureDeclaration',
                     'RemoteInterfaceNamingConvention', 'RemoteSessionInterfaceNamingConvention', 'ShortClassName',
                     'ShortMethodName', 'ShortVariable', 'SuspiciousConstantFieldName', 'TooManyStaticImports',
                     'UnnecessaryAnnotationValueElement', 'UnnecessaryConstructor', 'UnnecessaryFullyQualifiedName',
                     'UnnecessaryLocalBeforeReturn', 'UnnecessaryModifier', 'UnnecessaryReturn', 'UseDiamondOperator',
                     'UselessParentheses', 'UselessQualifiedThis', 'UseShortArrayInitializer',
                     'VariableNamingConventions', 'WhileLoopsMustUseBraces', 'AbstractClassWithoutAnyMethod',
                     'AvoidCatchingGenericException', 'AvoidDeeplyNestedIfStmts', 'AvoidRethrowingException',
                     'AvoidThrowingNewInstanceOfSameException', 'AvoidThrowingNullPointerException',
                     'AvoidThrowingRawExceptionTypes', 'AvoidUncheckedExceptionsInSignatures',
                     'ClassWithOnlyPrivateConstructorsShouldBeFinal', 'CollapsibleIfStatements',
                     'CouplingBetweenObjects', 'CyclomaticComplexity', 'DataClass', 'DoNotExtendJavaLangError',
                     'ExceptionAsFlowControl', 'ExcessiveClassLength', 'ExcessiveImports', 'ExcessiveMethodLength',
                     'ExcessiveParameterList', 'ExcessivePublicCount', 'FinalFieldCouldBeStatic', 'GodClass',
                     'ImmutableField', 'LawOfDemeter', 'LogicInversion', 'LoosePackageCoupling',
                     'ModifiedCyclomaticComplexity', 'NcssConstructorCount', 'NcssCount', 'NcssMethodCount',
                     'NcssTypeCount', 'NPathComplexity', 'SignatureDeclareThrowsException', 'SimplifiedTernary',
                     'SimplifyBooleanAssertion', 'SimplifyBooleanExpressions', 'SimplifyBooleanReturns',
                     'SimplifyConditional', 'SingularField', 'StdCyclomaticComplexity', 'SwitchDensity',
                     'TooManyFields', 'TooManyMethods', 'UselessOverridingMethod', 'UseObjectForClearerAPI',
                     'UseUtilityClass', 'AssignmentInOperand', 'AssignmentToNonFinalStatic',
                     'AvoidAccessibilityAlteration', 'AvoidAssertAsIdentifier', 'AvoidBranchingStatementAsLastInLoop',
                     'AvoidCallingFinalize', 'AvoidCatchingNPE', 'AvoidCatchingThrowable',
                     'AvoidDecimalLiteralsInBigDecimalConstructor', 'AvoidDuplicateLiterals', 'AvoidEnumAsIdentifier',
                     'AvoidFieldNameMatchingMethodName', 'AvoidFieldNameMatchingTypeName',
                     'AvoidInstanceofChecksInCatchClause', 'AvoidLiteralsInIfCondition',
                     'AvoidLosingExceptionInformation', 'AvoidMultipleUnaryOperators', 'AvoidUsingOctalValues',
                     'BadComparison', 'BeanMembersShouldSerialize', 'BrokenNullCheck', 'CallSuperFirst',
                     'CallSuperLast', 'CheckSkipResult', 'ClassCastExceptionWithToArray', 'CloneMethodMustBePublic',
                     'CloneMethodMustImplementCloneable', 'CloneMethodReturnTypeMustMatchClassName',
                     'CloneThrowsCloneNotSupportedException', 'CloseResource', 'CompareObjectsWithEquals',
                     'ConstructorCallsOverridableMethod', 'DataflowAnomalyAnalysis', 'DetachedTestCase',
                     'DoNotCallGarbageCollectionExplicitly', 'DoNotCallSystemExit', 'DoNotExtendJavaLangThrowable',
                     'DoNotHardCodeSDCard', 'DoNotThrowExceptionInFinally', 'DontImportSun',
                     'DontUseFloatTypeForLoopIndices', 'EmptyCatchBlock', 'EmptyFinalizer', 'EmptyFinallyBlock',
                     'EmptyIfStmt', 'EmptyInitializer', 'EmptyStatementBlock', 'EmptyStatementNotInLoop',
                     'EmptySwitchStatements', 'EmptySynchronizedBlock', 'EmptyTryBlock', 'EmptyWhileStmt', 'EqualsNull',
                     'FinalizeDoesNotCallSuperFinalize', 'FinalizeOnlyCallsSuperFinalize', 'FinalizeOverloaded',
                     'FinalizeShouldBeProtected', 'IdempotentOperations', 'ImportFromSamePackage',
                     'InstantiationToGetClass', 'InvalidSlf4jMessageFormat', 'InvalidLogMessageFormat',
                     'JumbledIncrementer', 'JUnitSpelling', 'JUnitStaticSuite', 'LoggerIsNotStaticFinal',
                     'MethodWithSameNameAsEnclosingClass', 'MisplacedNullCheck', 'MissingBreakInSwitch',
                     'MissingSerialVersionUID', 'MissingStaticMethodInNonInstantiatableClass', 'MoreThanOneLogger',
                     'NonCaseLabelInSwitchStatement', 'NonStaticInitializer', 'NullAssignment',
                     'OverrideBothEqualsAndHashcode', 'ProperCloneImplementation', 'ProperLogger',
                     'ReturnEmptyArrayRatherThanNull', 'ReturnFromFinallyBlock', 'SimpleDateFormatNeedsLocale',
                     'SingleMethodSingleton', 'SingletonClassReturningNewInstance', 'StaticEJBFieldShouldBeFinal',
                     'StringBufferInstantiationWithChar', 'SuspiciousEqualsMethodName', 'SuspiciousHashcodeMethodName',
                     'SuspiciousOctalEscape', 'TestClassWithoutTestCases', 'UnconditionalIfStatement',
                     'UnnecessaryBooleanAssertion', 'UnnecessaryCaseChange', 'UnnecessaryConversionTemporary',
                     'UnusedNullCheckInEquals', 'UseCorrectExceptionLogging', 'UseEqualsToCompareStrings',
                     'UselessOperationOnImmutable', 'UseLocaleWithCaseConversions', 'UseProperClassLoader',
                     'HardCodedCryptoKey', 'InsecureCryptoIv']
all_features = ['Label_Count', 'Review_Comments_Count', 'Following', 'Stars', 'Contributions', 'Merge_Latency',
                'Closed_Num_Rate', 'Followers',  'Workload', 'Wednesday', 'Closed_Num', 'Public_Repos', 'Comments_Count',
                'Deletions_Per_Week', 'Contributor', 'File_Touched_Average', 'Forks_Count', 'Organization_Core_Member',
                'Monday', 'Contain_Fix_Bug', 'src_churn', 'Team_Size', 'Last_Comment_Mention', 'Sunday',
                'Thursday', 'Project_Age', 'Open_Issues', 'Intra_Branch', 'Saturday', 'Participants_Count',
                'Comments_Per_Closed_PR', 'Watchers', 'Project_Accept_Rate', 'Point_To_IssueOrPR', 'Accept_Num',
                'Close_Latency', 'Contributor_Num', 'Commits_Average', 'Assignees_Count', 'Friday', 'Commits_PR',
                'Wait_Time', 'line_comments_count', 'Prev_PRs', 'Comments_Per_Merged_PR', 'Files_Changed', 'Day',
                'Churn_Average', 'Language', 'Tuesday', 'Additions_Per_Week', 'User_Accept_Rate', 'X1_0', 'X1_1',
                'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6',  'X1_7', 'X1_8', 'X1_9', 'PR_Latency', 'Project_Name',
                'PR_Date_Created_At', 'PR_Time_Created_At', 'PR_Date_Closed_At',
                'PR_Time_Closed_At', 'first_response_time', 'first_response', 'latency_after_first_response',
                'title_words_count', 'body_words_count', 'comments_reviews_words_count', 'is_smelly', 'PR_accept'
                 #'is_God_Class', 'is_Data_Class', #'Project_Domain',
                ]

# Previous work features
accept_baseline = ['src_churn', 'Commits_PR', 'Files_Changed', 'num_comments','Followers','Participants_Count',
                   'Team_Size', 'File_Touched_Average', 'Commits_Average', 'Prev_PRs', 'is_smelly', #'Project_Size',
                   'User_Accept_Rate', 'PR_Time_Created_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At',
                   'PR_Date_Created_At', 'Project_Name', 'PR_accept']

df = df[accept_baseline]
target = 'is_smelly'


predictors = [x for x in df.columns if x not in [target, 'PR_Date_Created_At', 'PR_Time_Created_At', 'PR_Date_Closed_At',
                                                 'PR_Time_Closed_At', 'Project_Name', 'PR_accept']]

predictors_with_label = [x for x in df.columns if x not in ['PR_accept', 'PR_Date_Created_At', 'PR_Time_Created_At',
                                                            'Project_Name']]

# Scale the training dataset: StandardScaler
def scale_data_standardscaler(df_):
    scaler_train =StandardScaler()
    df_scaled = scaler_train.fit_transform(np.array(df_).astype('float64'))
    df_scaled = pd.DataFrame(df_scaled, columns=predictors)

    return df_scaled

def extract_metric_from_report(report):
    report = list(report.split("\n"))
    report = report[-2].split(' ')
    # print(report)
    mylist = []
    for i in range(len(report)):
        if report[i] != '':
            mylist.append(report[i])

    return mylist[3], mylist[4], mylist[5]

def extract_each_class_metric_from_report(report):
    report = list(report.split("\n"))

    mydict2 = {}
    mydict = {}
    index = 0
    for line in range(len(report)):
        if report[line] != '':
            values_list = report[line].split(' ')
            mydict[index] = values_list
            index+=1
    count=0
    for value in mydict:
        mylist = []
        if value != 0:
            for item in range(len(mydict[value])):
                if mydict[value][item] != '':
                    mylist.append(mydict[value][item])
            mydict2[count] = mylist
            count+=1
    return mydict2[0], mydict2[1], mydict2[2], mydict2[3]

def train_MLP_model(clf, x_train, y_train, x_test, name=None):
    start_training_time = t.time()
#     clf.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    clf.fit(x_train, y_train)
    # test_loss, test_acc = model.evaluate(x_test, y_test)    
    y_pred_train = clf.predict(x_train)
    y_predprob_train = clf.predict_proba(x_train)[:, 1]
    training_time = round(t.time() - start_training_time, 3)
    print(f'Training time of the model: {training_time} seconds')
    start_testing_time = t.time()
    y_pred = clf.predict(x_test)
    y_predprob = clf.predict_proba(x_test)[:, 1]
    testing_time = round(t.time() - start_testing_time, 3)
    print(f'Testing time of the model: {testing_time} seconds')
    return y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time

def train_XGB_feature_importance(clf, x_train, y_train):
    clf = clf.fit(x_train, y_train, verbose=11)
    # importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    f_gain = clf.get_booster().get_score(importance_type='gain')
    importance = sorted(f_gain.items(), key=operator.itemgetter(1))
    print(importance)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df.to_csv('Results/features_fscore.csv', encoding='utf-8', index=True)

def train_XGB_model(clf, x_train, y_train, x_test, name=None):
    start_training_time = t.time()
    clf = clf.fit(x_train, y_train, verbose=11)
    # Save the model
    # with open('Saved_Models/3_labels/xgb_selected_features.pickle.dat', 'wb') as f:
    #     pickle.dump(clf, f)

    # Load the model
    # with open('response_xgb_16.pickle.dat', 'rb') as f:
    #     load_xgb = pickle.load(f)

    y_pred_train = clf.predict(x_train)
    y_predprob_train = clf.predict_proba(x_train)[:, 1]
    training_time = round(t.time() - start_training_time, 3)
    print(f'Training time of the model: {training_time} seconds')
    start_testing_time = t.time()
    y_pred = clf.predict(x_test)
    y_predprob = clf.predict_proba(x_test)[:, 1]
    testing_time = round(t.time() - start_testing_time, 3)
    print(f'Testing time of the model: {testing_time} seconds')

    return y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time

def train_SVM_model(clf, x_train, y_train, x_test, name=None):
    start_training_time = t.time()
    clf.fit(x_train, y_train)
    svm = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
    svm.fit(x_train, y_train)

    # with open('Saved_Models/3_labels/'+name+'.pickle.dat', 'wb') as f:
    #     pickle.dump(clf, f)
    # train
    y_pred_train = svm.predict(x_train)
    y_predprob_train = svm.predict_proba(x_train)[:, 1]
    training_time = round(t.time() - start_training_time, 3)
    print(f'Training time of the model: {training_time} seconds')
    start_testing_time = t.time()
    # test
    y_pred = svm.predict(x_test)
    y_predprob = svm.predict_proba(x_test)[:, 1]
    testing_time = round(t.time() - start_testing_time, 3)
    print(f'Testing time of the model: {testing_time} seconds')
    return y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time

def train_RF_LR_model(clf, x_train, y_train, x_test, name=None):
    start_training_time = t.time()
    clf.fit(x_train, y_train)
    # with open('Saved_Models/3_labels/'+name+'.pickle.dat', 'wb') as f:
    #     pickle.dump(clf, f)
    # train
    y_pred_train = clf.predict(x_train)
    y_predprob_train = clf.predict_proba(x_train)[:, 1]
    training_time = round(t.time() - start_training_time, 3)
    print(f'Training time of the model: {training_time} seconds')
    start_testing_time = t.time()
    # test
    y_pred = clf.predict(x_test)
    y_predprob = clf.predict_proba(x_test)[:, 1]
    testing_time = round(t.time() - start_testing_time, 3)
    print(f'Testing time of the model: {testing_time} seconds')
    return y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time


def calcuate_average_of_10_folds_for_each_project(df):
    avg_result = pd.DataFrame(columns=['Model', 'Project', 'AUC', 'neg_precision', 'pos_precision', 'neg_recall', 'pos_recall',
                                    'neg_f_score', 'pos_f_score', 'Precision', 'Recall', 'F-Score', 'Test_Accuracy',
                                    'Train_Accuracy', 'Training_time', 'Testing_time'])
    classifiers = get_classifiers()
    for project in df['Project_Name'].unique():
        df_project = df.loc[df.Project_Name == project]
        print('Project {} is under processing'.format(project))
        for name, value in classifiers.items():
            model_result = df_project.loc[df_project.Model == name]
            avg_result = avg_result.append(
                {'Model': name, 'Project': project,
                 'AUC': model_result['AUC'].mean(),
                 'neg_precision': model_result['neg_precision'].mean(),
                 'pos_precision': model_result['pos_precision'].mean(),
                 'neg_recall': model_result['neg_recall'].mean(),
                 'pos_recall': model_result['pos_recall'].mean(),
                 'neg_f_score': model_result['neg_f_score'].mean(),
                 'pos_f_score': model_result['pos_f_score'].mean(),
                 'Precision': model_result['Precision'].mean(),
                 'Recall': model_result['Recall'].mean(),
                 'F-Score': model_result['F-Score'].mean(),
                 'Test_Accuracy': model_result['Test_Accuracy'].mean(),
                 'Train_Accuracy': model_result['Train_Accuracy'].mean(),
                 'Training_time': model_result['Training_time'].mean(),
                 'Testing_time': model_result['Testing_time'].mean()},
                ignore_index=True)
    return avg_result

def calcuate_average_of_10_folds(df):
    # df = pd.read_csv('Results/results_10_fold_3.csv')
    avg_result = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F-measure', 'Test_Accuracy', 'Train_Accuracy'])
    classifiers = get_classifiers()
    for name, value in classifiers.items():
        model_result = df.loc[df.Model == name]
        avg_result = avg_result.append(
            {'Model': name,
             'Precision': model_result['Precision'].mean(),
             'Recall': model_result['Recall'].mean(),
             'F-measure': model_result['F-measure'].mean(),
             'Test_Accuracy': model_result['Test_Accuracy'].mean(),
             'Train_Accuracy': model_result['Train_Accuracy'].mean()},
            ignore_index=True)
    avg_result.to_csv('Results/results_10_fold_avg_3.csv', sep=',', encoding='utf-8', index=False)

def calcuate_average_of_10_folds_1():
    df = pd.read_csv('Results/accept/results_projects_2.csv')
    avg_result = pd.DataFrame(columns=['Model', 'AUC', 'neg_precision', 'pos_precision', 'neg_recall', 'pos_recall',
                                    'neg_f_score', 'pos_f_score', 'Precision', 'Recall', 'F-Score', 'Test_Accuracy',
                                    'Train_Accuracy'])
    classifiers = get_classifiers()
    for name, value in classifiers.items():
        model_result = df.loc[df.Model == name]
        avg_result = avg_result.append(
            {'Model': name, 'AUC': model_result['AUC'].mean(),
             'neg_precision': model_result['neg_precision'].mean(),
             'pos_precision': model_result['pos_precision'].mean(),
             'neg_recall': model_result['neg_recall'].mean(),
             'pos_recall': model_result['pos_recall'].mean(),
             'neg_f_score': model_result['neg_f_score'].mean(),
             'pos_f_score': model_result['pos_f_score'].mean(),
             'Precision': model_result['Precision'].mean(),
             'Recall': model_result['Recall'].mean(),
             'F-Score': model_result['F-Score'].mean(),
             'Test_Accuracy': model_result['Test_Accuracy'].mean(),
             'Train_Accuracy': model_result['Train_Accuracy'].mean()},
            ignore_index=True)
    avg_result.to_csv('Results/accept/results_project_avg_2.csv', sep=',', encoding='utf-8', index=False)

def extract_selected_features():
    df_f = pd.read_csv('Results/accept/features_fscore_2.csv')
    df_f = df_f.loc[df_f.fscore >= 15]
    print(df_f.sort_values(by=['fscore']))
    print(list(df_f.feature))
    print(df_f.shape)

def draw_features_barplot():
    df = pd.read_csv("Results/accept/features_fscore_2.csv", sep=",")
    df = df.sort_values(by=['fscore'], ascending=False)
    df = df[df.fscore >= 15]
    print(len(df.feature))
    print(list(df.feature))
    fig, ax = plt.subplots(figsize=(8, 3))
    df['fscore_log'] = np.log(df['fscore'])
    sns.set(style="whitegrid")
    sns.barplot(x="feature", y="fscore_log", data=df, ax=ax, palette="GnBu_d") #palette="Blues_d" GnBu_d ch:2.5,-.2,dark=.3
    ax.xaxis.set_tick_params(labelsize=9)
    ax.set_xlabel('Features')
    ax.set_ylabel('Average Gain (log-scaled)')
    plt.xticks(rotation=90)
    plt.savefig('Results/accept/plots/accept_SF_1.png', bbox_inches='tight')
    plt.show()


def baseline_classifer(X, y):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    start_training_time = t.time()
    dummy_clf.fit(X, y)
    training_time = round(t.time() - start_training_time, 3)
    start_testing_time = t.time()
    y_pred = dummy_clf.predict(X)
    y_predprob = dummy_clf.predict_proba(X)[:, 1]
    testing_time = round(t.time() - start_testing_time, 3)
    
    print(metrics.classification_report(y, y_pred, digits=2))
    print(f'AUC: {metrics.roc_auc_score(y, y_predprob)}')
    print(f'Accuracy: {dummy_clf.score(X, y)}')


def train_models_Using_TT_split(df_):
    results = pd.DataFrame(
        columns=['Model', 'AUC', 'Precision', 'Recall', 'F-Score', 'Test_Accuracy',
                 'Train_Accuracy', 'Training_time', 'Testing_time'])

    df = shuffle(df_)
    x_train, x_test, y_train, y_test = train_test_split(df[predictors], df[target],
                                                        test_size=0.1, stratify=df[target])
    X_train_scaled = scale_data_standardscaler(x_train)
    X_test_scaled = scale_data_standardscaler(x_test)
    classifiers = get_classifiers()
    for name, value in classifiers.items():
        clf = value
        print('Classifier: ', name)
        if name == 'XGBoost':
            y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_XGB_model(
                clf, x_train, y_train, x_test, name)
        elif name == 'LinearSVC':
            y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_SVM_model(
                clf, X_train_scaled, y_train, X_test_scaled, name)
        elif name == 'LogisticRegression':
            y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_RF_LR_model(
                clf, X_train_scaled, y_train, X_test_scaled, name)
        elif name == 'MLP':
            y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_MLP_model(
                clf, X_train_scaled, y_train, X_test_scaled, name)
        elif name == 'DT':
                y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_RF_LR_model(
                    clf, x_train, y_train, x_test, name)
        elif name == 'NaiveBayes':
            y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_RF_LR_model(
                clf, x_train, y_train, x_test, name)
        elif name == 'KNN':
            y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_RF_LR_model(
                clf, x_train, y_train, x_test, name)
        else:
            y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_RF_LR_model(
                clf, x_train, y_train, x_test, name)

        print("\nModel Report")
        print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
        print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
        print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
        print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))
        print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
        print("Recall : %f" % metrics.recall_score(y_test, y_pred))
        print("Precision : %f" % metrics.precision_score(y_test, y_pred))
        print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
        # print(metrics.confusion_matrix(y_test, y_pred))
        c_matrix = metrics.confusion_matrix(y_test, y_pred)
        print('========Confusion Matrix==========')
        print("          Smelly    Non-Smelly")
        print('Smelly      {}           {}'.format(c_matrix[0][0], c_matrix[0][1]))
        print('Non-Smelly  {}           {}'.format(c_matrix[1][0], c_matrix[1][1]))
        results = results.append(
            {'Model': name, 'AUC': metrics.roc_auc_score(y_test, y_predprob),
             'Precision': metrics.precision_score(y_test, y_pred),
             'Recall': metrics.recall_score(y_test, y_pred),
             'F-Score': metrics.f1_score(y_test, y_pred),
             'Test_Accuracy': metrics.accuracy_score(y_test, y_pred),
             'Train_Accuracy': metrics.accuracy_score(y_train, y_pred_train),
             'Training_time': training_time,
             'Testing_time': testing_time,
             }, ignore_index=True)
    results.to_csv('Results/results_1.csv',
                   sep=',', encoding='utf-8', index=False)
    print('CSV files saved...')


def train_models_using_10FoldCV(df_):
    results = pd.DataFrame(
        columns=['Model', 'Fold', 'AUC', 'Precision', 'Recall', 'F-Score', 'Test_Accuracy',
                 'Train_Accuracy', 'Training_time', 'Testing_time'])
    X = df_[predictors]
    y = df_[target]
    
    kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)  
    fold = 0
    for train_index, test_index in kf.split(X, y):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # print(y_train, y_test)
        X_train_scaled = scale_data_standardscaler(x_train)
        X_test_scaled = scale_data_standardscaler(x_test)

        classifiers = get_classifiers()
        for name, value in classifiers.items():
            clf = value
            print('Classifier: ', name)
            if name == 'XGBoost':
                y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_XGB_model(
                    clf, x_train, y_train, x_test, name)
            elif name == 'LinearSVC':
                y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_SVM_model(
                    clf, X_train_scaled, y_train, X_test_scaled, name)
            elif name == 'LogisticRegression':
                y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_RF_LR_model(
                    clf, X_train_scaled, y_train, X_test_scaled, name)
            elif name == 'MLP':
                y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_MLP_model(
                    clf, X_train_scaled, y_train, X_test_scaled, name)
            elif name == 'DT':
                y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_RF_LR_model(
                    clf, x_train, y_train, x_test, name)
            elif name == 'NaiveBayes':
                y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_RF_LR_model(
                    clf, x_train, y_train, x_test, name)
            elif name == 'KNN':
                y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_RF_LR_model(
                    clf, x_train, y_train, x_test, name)
            else:
                y_pred_train, y_predprob_train, y_pred, y_predprob, training_time, testing_time = train_RF_LR_model(
                    clf, x_train, y_train, x_test, name)

            print("\nModel Report")
            print("Train Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred_train))
            print("Test Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
            print('Train error: {:.3f}'.format(1 - metrics.accuracy_score(y_train, y_pred_train)))
            print('Test error: {:.3f}'.format(1 - metrics.accuracy_score(y_test, y_pred)))
            print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
            print("Recall : %f" % metrics.recall_score(y_test, y_pred))
            print("Precision : %f" % metrics.precision_score(y_test, y_pred))
            print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
            # print(metrics.confusion_matrix(y_test, y_pred))
            c_matrix = metrics.confusion_matrix(y_test, y_pred)
            print('========Confusion Matrix==========')
            print("          Smelly    Non-Smelly")
            print('Smelly      {}           {}'.format(c_matrix[0][0], c_matrix[0][1]))
            print('Non-Smelly  {}           {}'.format(c_matrix[1][0], c_matrix[1][1]))
            fold += 1
            results = results.append(
                {'Model': name, 'Fold': fold, 'AUC': metrics.roc_auc_score(y_test, y_predprob),
                 'Precision': metrics.precision_score(y_test, y_pred),
                 'Recall': metrics.recall_score(y_test, y_pred),
                 'F-Score': metrics.f1_score(y_test, y_pred),
                 'Test_Accuracy': metrics.accuracy_score(y_test, y_pred),
                 'Train_Accuracy': metrics.accuracy_score(y_train, y_pred_train),
                 'Training_time': training_time,
                 'Testing_time': testing_time,
                 }, ignore_index=True)
    avg_result = pd.DataFrame(columns=['Model', 'AUC', 'Precision', 'Recall', 'F-Score', 'Test_Accuracy',
                                       'Train_Accuracy', 'Training_time', 'Testing_time'])
    for name, value in classifiers.items():
        model_result = results.loc[results.Model == name]
        avg_result = avg_result.append(
            {'Model': name, 'AUC': model_result['AUC'].mean(),
             'Precision': model_result['Precision'].mean(),
             'Recall': model_result['Recall'].mean(),
             'F-Score': model_result['F-Score'].mean(),
             'Test_Accuracy': model_result['Test_Accuracy'].mean(),
             'Train_Accuracy': model_result['Train_Accuracy'].mean(),
             'Training_time': model_result['Training_time'].mean(),
             'Testing_time': model_result['Testing_time'].mean()
             },
            ignore_index=True)

    avg_result.to_csv('Results/10_fold_result_avg_1.csv',
                      sep=',', encoding='utf-8', index=False)
    results.to_csv('Results/10_fold_results_1.csv',
                   sep=',', encoding='utf-8', index=False)
    print('CSV files saved...')

def Baseline_10FoldCV(X, y): 
    results = pd.DataFrame(
        columns=['Model', 'Fold', 'AUC', 'Precision', 'Recall', 'F-Score', 'Test_Accuracy',
                 'Train_Accuracy', 'Training_time', 'Testing_time'])
    
    kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)  
    fold = 0
    for train_index, test_index in kf.split(X, y):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
         
        
        dummy_clf = DummyClassifier(strategy="most_frequent")
        start_training_time = t.time()
        dummy_clf.fit(x_train, y_train)
        training_time = round(t.time() - start_training_time, 3)
        start_testing_time = t.time()
        y_pred = dummy_clf.predict(x_train)
        y_predprob = dummy_clf.predict_proba(x_train)[:, 1]
        testing_time = round(t.time() - start_testing_time, 3)
        
        fold += 1
        results = results.append(
            {'Model': 'Baseline', 'Fold': fold, 'AUC': metrics.roc_auc_score(y_train, y_predprob),
             'Precision': metrics.precision_score(y_train, y_pred),
             'Recall': metrics.recall_score(y_train, y_pred),
             'F-Score': metrics.f1_score(y_train, y_pred),
             'Test_Accuracy': dummy_clf.score(x_train, y_train),
             'Train_Accuracy': dummy_clf.score(x_train, y_train),
             'Training_time': training_time,
             'Testing_time': testing_time,
             }, ignore_index=True)

#         print(metrics.classification_report(y_test, y_pred, digits=2))
#         print(f'AUC: {metrics.roc_auc_score(y_test, y_predprob)}')
#         print(f'Accuracy: {dummy_clf.score(x_train, y_train)}')
    return results

def hyperparamGridSearch(df_):  
    
    x_train, x_test, y_train, y_test = train_test_split(df[predictors], df[target],
                                                        test_size=0.1, stratify=df[target])
       
    X = x_train
    y = y_train
    X_scaled = scale_data_standardscaler(X)       
    kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)    
    classifiers = get_classifiers_without_params()    
    res = []
    for name, value in classifiers.items():
            clf = value
            print('Classifier: ', name)
            if name == 'XGBoost':
               # define search space
                space = dict()
                space['n_estimators']= [50, 100, 150, 200]
                space['learning_rate'] = [0.01, 0.1, 0.2, 0.3]
                space['gamma'] = [i/10.0 for i in range(3)]
                space['colsample_bytree'] = [i/10.0 for i in range(1, 3)]
                space['max_depth'] = range(3, 10)  
#                 space['gamma'] = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
#                 space['max_depth'] = [3, 5, 7, 9, 12, 15, 17, 25]
#                 space['min_child_weight'] = [1, 3, 5, 7]
#                 space['subsample'] = [0.6, 0.7, 0.8, 0.9, 1.0] 
#                 space['colsample_bytree'] = [0.6, 0.7, 0.8, 0.9, 1.0]
#                 space['lambda'] = [0.01, 0.1, 1.0]
#                 space['alpha'] = [0, 0.1, 0.5, 1.0]                
                # define search
                search = GridSearchCV(clf, space, scoring='accuracy', cv=kf)               
                # execute search
                result = search.fit(X, y)
                # saving result
                record = {'model': name, 'accuracy': result.best_score_}
                record.update(result.best_params_)
                res.append(record)
            elif name == 'LinearSVC':
                # define search space
                space = dict()
#                 space['kernel'] = ['poly', 'rbf', 'sigmoid']
                space['C'] = [1e-6, 1e+6, 1.0, 'log-uniform']
#                 space['gamma'] = ['scale']
                # define search
                search = GridSearchCV(clf, space, scoring='accuracy', n_jobs=-1, cv=kf)               
                # execute search
                result = search.fit(X_scaled, y)
                # saving result
                record = {'model': name, 'accuracy': result.best_score_}
                record.update(result.best_params_) 
                res.append(record)
            elif name == 'LogisticRegression':
                # define search space
                space = dict()
                space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
                space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
                space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]              
                # define search
                search = GridSearchCV(clf, space, scoring='accuracy', n_jobs=-1, cv=kf)               
                # execute search
                result = search.fit(X_scaled, y)
                # saving result
                record = {'model': name, 'accuracy': result.best_score_}
                record.update(result.best_params_)  
                res.append(record)
            elif name == 'BaggedDT':
                # define search space
                space = dict()
                space['n_estimators'] = [10, 100, 1000]             
                # define search
                search = GridSearchCV(clf, space, scoring='accuracy', n_jobs=-1, cv=kf)               
                # execute search
                result = search.fit(X, y)
                # saving result
                record = {'model': name, 'accuracy': result.best_score_}
                record.update(result.best_params_)  
                res.append(record)
            elif name == 'NaiveBayes':
                # define search space
                space = dict()
                space['var_smoothing'] = [1e-9 , np.logspace(0,-9, num=100) ]            
                # define search
                search = GridSearchCV(clf, space, scoring='accuracy', n_jobs=-1, cv=kf)               
                # execute search
                result = search.fit(X, y)
                # saving result
                record = {'model': name, 'accuracy': result.best_score_}
                record.update(result.best_params_)  
                res.append(record)
            elif name == 'KNN':
                # define search space
                space = dict()
                space['n_neighbors'] = range(1, 21, 2)
                space['weights'] = ['uniform', 'distance']
                space['metric'] = ['euclidean', 'manhattan', 'minkowski']              
                # define search
                search = GridSearchCV(clf, space, scoring='accuracy', n_jobs=-1, cv=kf)               
                # execute search
                result = search.fit(X, y)
                # saving result
                record = {'model': name, 'accuracy': result.best_score_}
                record.update(result.best_params_)  
                res.append(record)
            elif name == 'RandomForest':
                # define search space
                space = dict()
                space['n_estimators'] = [10, 100, 1000]
                space['max_features'] = ['sqrt', 'log2']
                # define search
                search = GridSearchCV(clf, space, scoring='accuracy', n_jobs=-1, cv=kf)               
                # execute search
                result = search.fit(X, y)
                # saving result
                record = {'model': name, 'accuracy': result.best_score_}
                record.update(result.best_params_) 
                res.append(record)
            elif name == 'MLP':
                # define search space
#                 space = dict()
#                 space['hidden_layer_sizes'] = [(50,50,50), (50,100,50), (100,)],
#                 space['activation'] = ['tanh', 'relu'],
#                 space['solver'] = ['sgd', 'adam'],
#                 space['alpha'] = [0.0001, 0.5],
#                 space['learning_rate'] = ['constant','adaptive']
                parameter_space = {
                    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                    'activation': ['tanh', 'relu'],
                    'solver': ['sgd', 'adam'],
                    'alpha': [0.0001, 0.05],
                    'learning_rate': ['constant','adaptive'],
                }
                # define search
                search = GridSearchCV(clf, parameter_space, scoring='accuracy', n_jobs=-1, cv=kf)               
                # execute search
                result = search.fit(X_scaled, y)
                # saving result
                record = {'model': name, 'accuracy': result.best_score_}
                record.update(result.best_params_) 
                res.append(record)
    resdf = pd.DataFrame(res)
    resdf.to_csv('Results/hptuning_results.csv',sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':

    print('Processing')

    df = df.dropna()
    print(df['Project_Name'].unique().tolist())
#     hyperparamGridSearch(df)
#     train_models_using_10FoldCV(df)
#     train_models_Using_TT_split(df)
#     train_baseline_df = Baseline_10FoldCV(df[predictors], df[target])
#     baseline_classifer(df[predictors], df[target])




