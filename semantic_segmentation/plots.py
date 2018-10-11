import pandas
from matplotlib import pyplot as plt
import argparse

dict_ = {
    "Loss" : "run_.-tag-Monitor_Total_Loss.csv",
    "Training Accuracy" : "run_.-tag-Monitor_training_accuracy.csv",
    "Training Mean IOU" : "run_.-tag-Monitor_training_mean_IOU.csv",
    "Test Accuracy" : "run_test-tag-Monitor_test_accuracy.csv",
    "Test Mean IOU" : "run_test-tag-Monitor_test_mean_IOU.csv",
    "Test Mean Per-Class Accuracy" : "run_test-tag-Monitor_test_mean_per_class_accuracy.csv"
}

def main(logdir):
    for key,value in dict_.items(): 
        dataframe = pandas.read_csv(logdir + value)
        steps = dataframe["Step"]
        values = dataframe["Value"]
        plt.plot(steps, values)
        plt.ylabel(key)
        plt.xlabel("Steps")
        if key == "Loss" :
            plt.ylim(0.0, 20.0)
        elif key == "Training Accuracy":
            plt.ylim(0.0, 1.0)
        elif key == "Training Mean IOU":
            plt.ylim(0.0, 0.4)
        elif key == "Test Accuracy":
            plt.ylim(0.7, 1.0) # 0.8-1.0
        elif key == "Test Mean IOU":
            plt.ylim(0.3, 0.5)
        elif key == "Test Mean Per-Class Accuracy":
            plt.ylim(0.6, 0.9) # 0.5-0.8
        plt.savefig(logdir + key.lower().replace(" ","_") + ".png", format = 'png', dpi = 200)
        plt.clf()
        print("Saved " + logdir + key.lower().replace(" ","_") + ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plotsdir", required=True, help="Path to plot csvs")
    args = parser.parse_args()
    main(args.plotsdir)