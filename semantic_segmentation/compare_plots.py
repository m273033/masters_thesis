import pandas
import matplotlib.pyplot as plt

dict_ = {
    "Loss" : "run_.-tag-Monitor_Total_Loss.csv",
    "Training Accuracy" : "run_.-tag-Monitor_training_accuracy.csv",
    "Training Mean IOU" : "run_.-tag-Monitor_training_mean_IOU.csv",
    "Test Accuracy" : "run_test-tag-Monitor_test_accuracy.csv",
    "Test Mean IOU" : "run_test-tag-Monitor_test_mean_IOU.csv",
    "Test Mean Per-Class Accuracy" : "run_test-tag-Monitor_test_mean_per_class_accuracy.csv"
}

def ENet_weights():
    for key,value in dict_.items(): 
        
        #Single run data
        single_df = pandas.read_csv("./log/Cityscapes_ENET/plots/" + value)
        steps_single = single_df["Step"]
        values_single = single_df["Value"]

        #Transfer learn data
        tl_df = pandas.read_csv("./log/transfer_learning_ENET_3/plots/" + value)
        steps_tl = tl_df["Step"]
        values_tl = tl_df["Value"]

        #Combined env data
        combined_df = pandas.read_csv("./log/combined_ENET/plots/" + value)
        steps_combined = combined_df["Step"]
        values_combined = combined_df["Value"]

        plt.plot(steps_combined, values_combined, color='green', label='Combined dataset')
        plt.plot(steps_single, values_single, color='red', label='without TL')
        plt.plot(steps_tl, values_tl, color='blue', label='Transfer Learning')

        plt.legend(loc='best')
        plt.ylabel(key)
        plt.xlabel("Steps")
        if key == "Loss" :
            plt.ylim(0.0, 20.0)
            plt.xlim(xmax=100000)
        elif key == "Training Accuracy":
            plt.ylim(0.0, 1.0)
            plt.xlim(xmax=100000)
        elif key == "Training Mean IOU":
            plt.ylim(0.0, 0.7)
            plt.xlim(xmax=100000)
        elif key == "Test Accuracy":
            plt.ylim(0.7, 1.0)
        elif key == "Test Mean IOU":
            plt.ylim(0.2, 0.8)
        elif key == "Test Mean Per-Class Accuracy":
            plt.ylim(0.4, 0.9)
        
        plt.savefig("./log/enet_plots/" + key.lower().replace(" ","_") + ".png", format = 'png', dpi = 200)
        plt.clf()
        print("Saved " + "./log/enet_plots/" + key.lower().replace(" ","_") + ".png")

def MFB_weights():
    for key,value in dict_.items(): 

        #Single run data
        single_df = pandas.read_csv("./log/Cityscapes_MFB/plots/" + value)
        steps = single_df["Step"]
        values = single_df["Value"]

        #Transfer learn data
        tl_df = pandas.read_csv("./log/transfer_learn_MFB/plots/" + value)
        steps_ = tl_df["Step"]
        values_ = tl_df["Value"]

        plt.plot(steps_, values_, color='blue', label='Transfer Learning')
        plt.plot(steps, values, color='red', label='without TL')
        plt.legend(loc='best')
        plt.ylabel(key)
        plt.xlabel("Steps")
        if key == "Loss" :
            plt.ylim(0.0, 0.22)
        elif key == "Training Accuracy":
            plt.ylim(0.0, 1.0)
        elif key == "Training Mean IOU":
            plt.ylim(0.0, 0.4)
        elif key == "Test Accuracy":
            plt.ylim(0.7, 1.0)
        elif key == "Test Mean IOU":
            plt.ylim(0.3, 0.5)
        elif key == "Test Mean Per-Class Accuracy":
            plt.ylim(0.5, 0.7)
        plt.savefig("./log/mfb_plots/" + key.lower().replace(" ","_") + ".png", format = 'png', dpi = 200)
        plt.clf()
        print("Saved " + "./log/mfb_plots/" + key.lower().replace(" ","_") + ".png")

def main():
    ENet_weights()
    #MFB_weights()

if __name__ == "__main__":
    main()