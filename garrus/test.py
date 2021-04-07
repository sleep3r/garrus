import pickle

from garrus.metrics import ECE


if __name__ == "__main__":
    with open("/Users/alexander/Desktop/ШАД/garrus/test_acc.pickle", "rb") as f:
        acc = pickle.load(f)

    with open("/Users/alexander/Desktop/ШАД/garrus/test_confs.pickle", "rb") as f:
        conf = pickle.load(f)

    ece_metric = ECE(n_bins=10)
    ece_metric.compute(confidences=conf, accuracies=acc)