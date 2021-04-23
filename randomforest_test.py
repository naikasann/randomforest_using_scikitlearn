import os
import sys
import datetime
import yaml
import pprint
import csv
import collections
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import pickle

# Global variables required at runtime
execution_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
save_folder_name = "./result/test/" + execution_time
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

def make_cm(matrix, columns):
    cm = pd.DataFrame(matrix, columns=[['Predicted Results'] * len(columns), columns], index=[['Correct answer data'] * len(columns), columns])
    logprint(cm)
    return cm

def output_result(forest, X_test, Y_test, config, save_folder_name):
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
                         f1_score(Y_test, y_pred, average="micro"),]],
                         columns=["score","accuracy","precision","recall","f1 score"])
    score_df.to_csv(csv_name, header=False, index=False, mode="a")

    pprint.pprint(classifycation_repo_df)
    pprint.pprint(classifycation_repo_df, stream=logfile)
    logprint(score_df)

    return score_df

def main():
    # Load the configuration file.
    with open("config.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    logprint("=== Random forest test ===")

    # Create a data set.
    test_df  = pd.read_csv(config["test"]["testpath"])
    # Set the number of data types to be used for training.
    data_number = config["train"]["data_number"]
    # Extracting data
    X_test  = test_df[test_df.columns[test_df.columns != test_df.columns[data_number]]].values
    Y_test  = test_df[test_df.columns[data_number]]

    logprint("test  data : {}".format(X_test.shape[0]))
    test_counter = collections.Counter(Y_test)
    logprint(test_counter)

    # load randomforest model.
    modelfile = config["test"]["modelpath"]
    # pickle object loading.
    if "pkl" in modelfile:
        forest = pickle.load(open(modelfile, 'rb'))
    # joblib model loading.
    elif "joblib" in modelfile:
        forest = joblib.load(modelfile)
    else:
        print("not found model path. exit program.")
        sys.exit()

    save_folder_name = "./result/test/" + execution_time
    # Create a folder to store the results of each fold experiment.
    save_folder_name = save_folder_name
    os.makedirs(save_folder_name, exist_ok=True)

    # Output and save each learning result.
    _ = output_result(forest, X_test, Y_test, config, save_folder_name)
    output_correctdata(forest, X_test, Y_test, save_folder_name)

    logprint("==============")

if __name__ == "__main__":
    main()