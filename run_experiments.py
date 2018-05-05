from src.DeepCNN import DeepCNN as DeepCNN
from src.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork as ShallowCNN
import numpy as np
from pathlib import Path

datasets = {
    "test": str(Path("./test").absolute()),
    "UCMerced": str(Path("../UCMerced_Original").absolute())
}
img_size = 256
folds = 5
layers = 5

results = {
    "DRNN": {
        "UCMerced": [],
        "test": []
    },
    "DCNN": {
        "UCMerced": [],
        "test": []
    },
    "ShallowCNN": {
        "UCMerced": [],
        "test": []
    }
}

for dataset in datasets:

    for num_epochs in np.arange(10, 60, 10):

        ################
        # Shallow CNN
        ###############
        print("Running ShallowCNN on {} for {} epochs".format(
            dataset, num_epochs))
        cnn = ShallowCNN(
            datasets[dataset],
            num_layers=layers,
            num_folds=folds,
            num_epochs=num_epochs,
            img_dimension=img_size)
        cnn.compile()
        cnn.fit()
        results["ShallowCNN"][dataset].append(cnn.average_accuracy)

        ################
        # DEEP CNN
        ###############
        print("Running DCNN on {} for {} epochs".format(dataset, num_epochs))
        deep_cnn = DeepCNN(
            datasets[dataset],
            img_size=img_size,
            residual=False,
            num_epochs=num_epochs,
            num_folds=folds)
        deep_cnn.compile()
        deep_cnn.fit()
        results["DCNN"][dataset].append(deep_cnn.average_accuracy)

        ################
        # DEEP RNN
        ###############
        print("Running DRNN on {} for {} epochs".format(dataset, num_epochs))
        deep_rnn = DeepCNN(
            datasets[dataset],
            img_size=img_size,
            residual=True,
            num_epochs=num_epochs,
            num_folds=folds)
        deep_rnn.compile()
        deep_rnn.fit()
        results["DRNN"][dataset].append(deep_rnn.average_accuracy)

        ###############
        # Delete models
        ################
        del cnn
        del deep_cnn
        del deep_rnn

#########
# Write out Results
#########
with open("experiment_results.csv", "w") as file:
    for d in datasets:
        file.write("\n\n\n")
        file.write("{} Dataset\n".format(d))
        file.write("Network, 10, 20, 30, 40, 50\n")
        for network in results:
            line = network + ","
            for score in results[network][d]:
                line += str(score) + ","
            file.write(line + "\n")
