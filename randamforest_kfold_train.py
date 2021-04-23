import os
import datetime
import yaml
import pprint
import csv
import collections
from tqdm import tqdm
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import pydotplus as pdp
import pandas as pd
import numpy as np
import joblib
import pickle

# Global variables required at runtime
execution_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
save_folder_name = "./result/kfold/" + execution_time
# Create a folder of results
os.makedirs(save_folder_name, exist_ok=True)
# Keep the log file open. There's a better way to write this... Well, forgive me ...
logfile = open(save_folder_name + "/result.log", mode="w")

def logprint(string):
    """
    func : Output to log and output to terminal.
           I know about logging and stuff, but I don't use it ... . I really don't! ;(
    """
    print(string)
    print(string, file=logfile)

def save_model(forest, save_folder_name):
    """ func : Save the model trained by randomforest. """
    # Create a folder of model.
    save_folder_name = save_folder_name + "/model/"
    os.makedirs(save_folder_name, exist_ok=True)
    # Save in joblib and pickle.
    joblib.dump(forest, save_folder_name + "model.joblib")
    with open(save_folder_name + 'model.pkl', 'wb') as model_file:
        pickle.dump(forest, model_file)

def make_dataset(x_train, y_train, x_test, y_test, save_folder_name):
    """
    func :  Create the data set used for each Fold.
            (so that you can conduct replicated experiments)
    """
    # Creating training data.
    with open(save_folder_name + "/traindata.csv", mode="w") as traindata_file:
        for (x, y) in zip(x_train, y_train):
            traindata_file.write(",".join(map(str, x.tolist())) + "," + str(y) + "\n")

    # Creating test data.
    with open(save_folder_name + "/testdata.csv", mode="w") as traindata_file:
        for (x, y) in zip(x_test, y_test):
            traindata_file.write(",".join(map(str, x.tolist())) + "," + str(y) + "\n")

def output_correctdata(forest, x_test, y_test, save_folder_name):
    """
    func : Compare the prediction data with the correct data, and store the actual correct and incorrect data in a csv.
    """
    correct_list, uncorrect_list = [], []

    # Compare prediction data with correct answer data.
    predictdata = forest.predict(x_test)
    for count, (prd, y) in enumerate(zip(predictdata, y_test.tolist())):
        if prd == y:
            # TODO : I didn't check to see if it was right.
            correct_list.append(x_test[count].tolist())
            correct_list[len(correct_list) - 1].append(y)
        else:
            uncorrect_list.append(x_test[count].tolist())
            uncorrect_list[len(uncorrect_list) - 1].append(y)
    # Save each result in csv.
    with open(save_folder_name + "/correct_data.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(correct_list)
    with open(save_folder_name + "/uncorrect_data.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(uncorrect_list)

def output_feature_importances(forest, config, save_folder_name):
    """
    func : Extract the feature importances of a randomforest and plot it on a graph.
    """
    # get feature importances
    feature_importances = forest.feature_importances_

    # Save the file as a text file.
    with open(save_folder_name + "/feature_importance.txt", mode="w") as w_file:
        logprint("Feature Importances:")
        w_file.write("Feature importances\n")
        for i, category in enumerate(config["train"]["data_detail"]):
            logprint("\t {0:20s} : {1:.6f}".format(category, feature_importances[i]))
            w_file.write("{0:20s} : {1:.6f}\n".format(category, feature_importances[i]))

    # Create a data frame for feature_importance (to make it easier to create diagrams)
    f = pd.DataFrame({"category": config["train"]["data_detail"],
                      "feature" : feature_importances[:]}).sort_values('feature',ascending=False)
    indices = f.index.tolist()
    # Create a bar chart and save it.
    plt.figure(figsize=(9, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(feature_importances)),feature_importances[indices], color='lightblue', align='center')
    plt.xticks(range(len(feature_importances)), f["category"].tolist(), rotation=90)
    plt.xlim([-1, len(feature_importances)])
    plt.tight_layout()
    plt.savefig(save_folder_name + "/feature_importances.png")

def visualize_decision_tree(forest, config, save_directory):
    """
    func : visualize decision tree.
    """
    # Create a folder to store the decision tree.
    os.makedirs(save_directory + "/tree/", exist_ok=True)
    print("visualize decision tree. please wait...")

    logprint("forest_number : {}".format(len(forest.estimators_)))

    # Extract all decision items
    estimators = forest.estimators_[:config["result"]["trees"]]
    with tqdm(estimators, leave=True, position=0) as pbar:
        for i, estimator in enumerate(pbar):
            # Read decision eyes as image data in graphviz.(dot data format.)
            dot_data = tree.export_graphviz(estimator,
                                        out_file=None,
                                        filled=True, # Color-code the frequency of nodes
                                        rounded=True,
                                        feature_names=config["train"]["data_detail"],# TODO : List of features
                                        class_names=config["train"]["target"],
                                        special_characters=True)
            graph = pdp.graph_from_dot_data(dot_data)
            # Specify the directory of the graphviz exe file. (To avoid errors)
            graph.progs = {'dot': config["result"]["graphviz_path"]}
            # save image.
            graph.write_png(save_directory + "/tree/tree_visualization_{:0=4}.png".format(i+1))
        tqdm._instances.clear()

def make_cm(matrix, columns):
    cm = pd.DataFrame(matrix, columns=[['Predicted Results'] * len(columns), columns], index=[['Correct answer data'] * len(columns), columns])
    logprint(cm)
    return cm

def output_result(forest , X_train, X_test, Y_test, config, save_folder_name):
    """
    func : Test the training results and output the accuracy.
            (In confusion matrix and classification report)
    """
    # Produce inference results (there is a better way) ;((
    y_pred = forest.predict(X_test)

    # Output and save the confusion matrix.
    matrix = confusion_matrix(Y_test, y_pred, labels=range(len(config["train"]["target"])))
    matrix_df = make_cm(matrix, config["train"]["target"])
    matrix_df.to_csv(save_folder_name + "/confusion_matrix.csv")
    # Save the image of the confusion matrix.
    plt.figure(figsize=(9, 6))
    sns.heatmap(matrix, annot=True, square=True, cmap='Blues', fmt='g')
    plt.savefig(save_folder_name + "/confusion_matrix.png")

    # Output and save the classification report.
    csv_name = save_folder_name + "/classification_report.csv"
    classifycation_repo_df = pd.DataFrame(classification_report(Y_test, y_pred, target_names=config["train"]["target"], output_dict=True))
    classifycation_repo_df.T.to_csv(csv_name)
    score_df = pd.DataFrame([[forest.score(X_test, Y_test),
                         accuracy_score(Y_test, y_pred),
                         precision_score(Y_test, y_pred, average="micro"),
                         recall_score(Y_test, y_pred, average="micro"),
                         f1_score(Y_test, y_pred, average="micro"),
                         forest.score(X_train, forest.oob_decision_function_.argmax(axis = 1))]],
                         columns=["score","accuracy","precision","recall","f1 score","oob data accuracy"])
    score_df.to_csv(csv_name, header=False, index=False, mode="a")

    pprint.pprint(classifycation_repo_df)
    pprint.pprint(classifycation_repo_df, stream=logfile)
    logprint(score_df)

    return score_df

def main():
    # Load the configuration file.
    with open("kfold_train_config.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Create a data set.
    df = pd.read_csv(config["train"]["csv"])
    # Set the number of data types to be used for training.
    data_number = config["train"]["data_number"]
    # Extracting data
    data = df[df.columns[df.columns != df.columns[data_number]]].values
    target = df[df.columns[data_number]]

    logprint("all data : {}".format(data.shape[0]))

    # Split the data for K-cross validation()
    kfold = StratifiedKFold(n_splits=config["train"]["K_fold"], shuffle=True)
    result_df = pd.DataFrame(columns=["score","accuracy","precision","recall","f1 score","oob data accuracy"])
    # Start each fold experiment.
    for fold, (train_idx, test_idx) in enumerate(kfold.split(data, target)):
        logprint("=== Fold {} ===".format(fold + 1))
        # Retrieve real data from the index
        X_train, Y_train, X_test, Y_test  = data[train_idx], target[train_idx], data[test_idx], target[test_idx]

        logprint("train data : {}".format(X_train.shape[0]))
        train_counter = collections.Counter(Y_train)
        logprint(train_counter)
        logprint("test  data : {}".format(X_test.shape[0]))
        test_counter = collections.Counter(Y_test)
        logprint(test_counter)

        # Run training on a random forest.
        forest = RandomForestClassifier(random_state=1234, oob_score=True)
        forest.fit(X_train, Y_train)

        save_folder_name = "./result/kfold/" + execution_time
        # Create a folder to store the results of each fold experiment.
        save_folder_name = save_folder_name + "/fold{}".format(fold+1)
        os.makedirs(save_folder_name, exist_ok=True)

        # Create a data set.
        make_dataset(X_train, Y_train, X_test, Y_test,save_folder_name)

        # Output and save each learning result.
        save_model(forest, save_folder_name)
        score_df = output_result(forest, X_train, X_test, Y_test, config, save_folder_name)

        output_correctdata(forest, X_test, Y_test, save_folder_name)
        output_feature_importances(forest, config, save_folder_name)
        visualize_decision_tree(forest, config, save_folder_name)

        # k-cross-validation method concatenate data frames to display all training results
        result_df = pd.concat([result_df, score_df])
        logprint("==============")

    # Output and save all training results of k-cross-validation method.
    logprint("=== result ===")
    result_df.index = result_df.index + 1
    logprint(result_df)
    logprint("==============")
    result_df.to_csv("./result/kfold/" + execution_time + "/experiment_result.csv")

if __name__ == "__main__":
    main()