import copy
import torch


class ModelTrainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
        }

        self.best_val_loss = float("inf")
        self.best_model_state = None


    def train_one_epoch(self, train_loader):

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:

            X = X.to(self.device)
            y = y.float().to(self.device)

            # Clear gradients from previous batch
            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(X)

            # Loss
            loss = self.criterion(pred, y)

            # Backpropagation
            loss.backward()

            # Update weights
            self.optimizer.step()

            # -------------------------
            # Statistics
            # -------------------------

            running_loss += loss.item() * X.size(0)

            probabilities = torch.sigmoid(pred)
            predicted_class = (probabilities >= 0.5).float()

            correct += (predicted_class == y).sum().item()
            total += y.numel()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total

        self.history["train_loss"].append(epoch_loss)
        self.history["train_accuracy"].append(epoch_accuracy)

        return epoch_loss, epoch_accuracy


    def validation_one_epoch(self, val_loader):

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for X, y in val_loader:

                X = X.to(self.device)
                y = y.float().to(self.device)

                # Forward pass
                pred = self.model(X)

                # Loss
                loss = self.criterion(pred, y)

                # -------------------------
                # Statistics
                # -------------------------

                running_loss += loss.item() * X.size(0)

                probabilities = torch.sigmoid(pred)
                predicted_class = (probabilities >= 0.5).float()

                correct += (predicted_class == y).sum().item()
                total += y.numel()

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_accuracy = correct / total

        self.history["val_loss"].append(epoch_loss)
        self.history["val_accuracy"].append(epoch_accuracy)

        return epoch_loss, epoch_accuracy


    def fit(self, train_loader, val_loader, epochs):

        for epoch in range(1, epochs + 1):

            train_loss, train_accuracy = self.train_one_epoch(
                train_loader
            )

            val_loss, val_accuracy = self.validation_one_epoch(
                val_loader
            )

            # -------------------------
            # Keep best model
            # -------------------------

            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss

                self.best_model_state = copy.deepcopy(
                    self.model.state_dict()
                )

            # -------------------------
            # Display progress
            # -------------------------

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

        # Restore model with best validation loss
        if self.best_model_state is not None:
            self.model.load_state_dict(
                self.best_model_state
            )

        print(
            f"\nTraining complete. "
            f"Best validation loss: {self.best_val_loss:.4f}"
        )


    def evaluate(self, test_loader):

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        all_probabilities = []
        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for X, y in test_loader:

                X = X.to(self.device)
                y = y.float().to(self.device)

                pred = self.model(X)

                loss = self.criterion(pred, y)

                running_loss += loss.item() * X.size(0)

                probabilities = torch.sigmoid(pred)
                predicted_class = (probabilities >= 0.5).float()

                correct += (predicted_class == y).sum().item()
                total += y.numel()

                all_probabilities.append(
                    probabilities.cpu()
                )

                all_predictions.append(
                    predicted_class.cpu()
                )

                all_labels.append(
                    y.cpu()
                )

        test_loss = running_loss / len(test_loader.dataset)
        test_accuracy = correct / total

        all_probabilities = torch.cat(all_probabilities)
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        print(
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_accuracy:.4f}"
        )

        return {
            "loss": test_loss,
            "accuracy": test_accuracy,
            "probabilities": all_probabilities,
            "predictions": all_predictions,
            "labels": all_labels
        }