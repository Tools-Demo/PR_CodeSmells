# PR_CodeSmells
Replication package for the paper detection of code smells in pull requests on GitHub.

### How-to
1. Clone the repository
```
$ git clone https://github.com/Tools-Demo/PR_CodeSmells.git
```
2. Place the dataset file in PR_CodeSmells-main/Dataset (Download from [25_projects_PRs.csv](https://github.com/Tools-Demo/PR_CodeSmells/releases/download/v1.0/25_projects_PRs.csv))

3. Change directory
```
$ cd PR_CodeSmells-main
```

4. Execute python classify_quality_PRs_1.py

5. Output for each executed method will be generated as a CSV file under "Results" folder (Existing results are placed in "Results_old" folder)

6. Methods that can be executed in the main function:
```
$ train_models_using_10FoldCV(df) [Default]  -- Train all classifiers with 10-fold cross-validation
$ Baseline_10FoldCV(df) -- Train baseline classifier with 10-fold cross-validation
$ baseline_classifer(df)  -- Train baseline classifier
$ hyperparamGridSearch(df)  -- Execute grid search for HPO

```


