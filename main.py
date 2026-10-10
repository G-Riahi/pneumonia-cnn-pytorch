from PneumoniaCNN import PneumoniaCNNBaseline
from Data_Loader import PneumoniaDataModule
from Model_Trainer import ModelTrainer
from Grad_CAM import Grad_CAM_implementaiton

import torch
import torch.nn as nn

from sklearn.metrics import (precision_score, recall_score,f1_score, classification_report, confusion_matrix)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Preparing data...")
data = PneumoniaDataModule()
data.setup()
distribution = data.get_class_distribution()

print("Creating model...")
model = PneumoniaCNNBaseline().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(distribution["normal"] / distribution["pneumonia"]))

trainer = ModelTrainer(
    model=model,
    optimizer=optimizer,
    criterion=loss,
    device=device
)

print("Starting training...")
trainer.fit(data.train_loader, data.val_loader, 60)

print("Evaluating...")
results = trainer.evaluate(data.test_loader)

y_true = results["labels"].numpy().flatten().astype(int)
y_pred = results["predictions"].numpy().flatten().astype(int)

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\nTest metrics")
print("-------------------------")
print(f"Accuracy:  {results['accuracy']:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Pneumonia"],
        zero_division=0
    )
)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion matrix:")
print(cm)

gradcam = Grad_CAM_implementaiton(model, [model.features[3]], data.test_dataset, data.mean, data.std, device)

gradcam.plot_pneumonia()