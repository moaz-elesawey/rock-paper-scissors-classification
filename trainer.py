import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, device, verbose=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose

        self.results = {
            "train_loss": [],
            "valid_loss": [],
            "train_acc" : [],
            "valid_acc" : [],
        }

        self.fitted = False


    def __train_step(self, train_dl: DataLoader, valid_dl: DataLoader):

        train_acc, train_loss = 0, 0
        valid_acc, valid_loss = 0, 0

        for batch, (images, labels) in enumerate(train_dl):
            images, labels = images.to(self.device), labels.to(self.device)

            predictions = self.model(images)

            loss = self.criterion(predictions, labels)

            train_loss += loss.item()
            train_acc += torch.sum( torch.argmax(
                torch.softmax(predictions, dim=1), dim=1) == labels
                ).item() / len(labels)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        train_loss /= len(train_dl)
        train_acc /= len(train_dl)

        self.model.eval()

        with torch.inference_mode():
            for batch, (images, labels) in enumerate(valid_dl):
                images, labels = images.to(self.device), labels.to(self.device)

                predictions = self.model(images)

                loss = self.criterion(predictions, labels)

                valid_loss += loss.item()
                valid_acc += torch.sum( torch.argmax(
                    torch.softmax(predictions, dim=1), dim=1) == labels
                    ).item() / len(labels)

            valid_loss /= len(valid_dl)
            valid_acc /= len(valid_dl)

        return train_loss, train_acc, valid_loss, valid_acc


    def __test_step(self, test_dl: DataLoader):

        test_acc, test_loss = 0, 0

        self.model.eval()

        with torch.inference_mode():
            for batch, (images, labels) in enumerate(test_dl):
                images, labels = images.to(self.device), labels.to(self.device)

                predictions = self.model(images)

                loss = self.criterion(predictions, labels)

                test_loss += loss.item()
                test_acc += torch.sum( torch.argmax(
                    torch.softmax(predictions, dim=1), dim=1) == labels
                    ).item() / len(labels)

            test_loss /= len(test_dl)
            test_acc /= len(test_dl)

        return test_acc, test_loss


    def train(self, train_dl: DataLoader, valid_dl: DataLoader, epochs: int=10):

        self.model.train()
        print("[INFO] Started Training for {}".format(epochs))

        for i in tqdm(range(epochs)):

            train_loss, train_acc, valid_loss, valid_acc = self.__train_step(train_dl, valid_dl)

            self.results["train_loss"].append(train_loss)
            self.results["valid_loss"].append(valid_loss)
            self.results["train_acc"].append(train_acc)
            self.results["valid_acc"].append(valid_acc)

            if self.verbose:
                print("Epoch [{:2d}/{:2d}] Train Loss : {:8.3f} | Train Acc: {:8.3f} | Valid Loss: {:8.3f} | Valid Acc: {:8.3f}"\
                      .format(i+1, epochs, train_loss, train_acc, valid_loss, valid_acc))

        self.fitted = True

        return self.results


    def evaluate(self, test_dl: DataLoader):
        if not self.fitted:
            raise ValueError("must fit the mode first. call train method first")

        test_acc, test_loss = self.__test_step(test_dl)

        if self.verbose:
            print("Test Loss : {:8.3f} | Test Acc: {:8.3f}".format(test_loss, test_acc))

        return test_loss, test_acc


    def plot_model_results(self):

        if not self.fitted:
            raise ValueError("must fit the mode first. call train method first")

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

        xdata = range(1, len(self.results["train_acc"])+1)

        ax1.plot(xdata, self.results["train_loss"], label="Train Loss")
        ax1.plot(xdata, self.results["valid_loss"], label="Valid Loss")
        ax1.set_xlabel("Epochs");ax1.set_ylabel("Loss")
        ax1.set_title("Model Loss")

        ax2.plot(xdata, self.results["train_acc"], label="Train Acc")
        ax2.plot(xdata, self.results["valid_acc"], label="Valid Acc")
        ax2.set_xlabel("Epochs");ax2.set_ylabel("Accuracy")
        ax2.set_title("Model Accuracy")

        ax1.legend(); ax2.legend()

        plt.show()

    def save(self, filename: str):
        if not self.fitted:
            raise ValueError("must fit the mode first. call train method first")

        torch.save(self.model, filename)

        print("Model Saved Successfully as [{}]".format(filename))


    def load(self, filename: str):
        model = torch.load(filename)

        print("Model Loaded Successfully from [{}]".format(filename))

        return model

