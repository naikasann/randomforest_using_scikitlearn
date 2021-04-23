import os
import sys
import csv
import serial
import ctypes
import numpy as np
import joblib
import pickle

def getkey(key):
    return(bool(ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000))

def main():
    category = ["walk", "run"]
    ser = serial.Serial("COM",115200,timeout=None)
    csvfile = ""

    # load randomforest model.
    modelfile = ""
    # pickle object loading.
    if "pkl" in modelfile:
        forest = pickle.load(open(modelfile, 'rb'))
    # joblib model loading.
    elif "joblib" in modelfile:
        forest = joblib.load(modelfile)
    else:
        print("not found model path. exit program.")
        sys.exit()

    with open(csvfile, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        while True:
            # BTserial read.
            line = ser.readline()
            # Make a list of the read data.
            readdata = line.decode("utf-8").strip().split("\n")
            # Write data read by serial to csv
            writer.writerow(readdata)
            # for debug.
            print("read data :",readdata)

            # Predict with Random Forests
            result_category = forest.predict(readdata)
            # result output.
            print("forest predict :", category[result_category])

            # If you press the ESC key, exit the program.
            if getkey(0x1B):
                break

    ser.close()

if __name__ == "__main__":
    main()