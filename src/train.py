import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from .plotting import plot_mdr

import logging
import wandb


def do_training(model, optimizer, metric, train, test, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1} / {epochs}")
        model.train()
        running_loss = 0.

        for inputs, labels in train:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = metric(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_epoch_loss = running_loss / len(train)

        model.eval()
        running_loss = 0.

        true_labels = []
        y_scre = []
        for inputs, labels in test:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = metric(outputs.squeeze(), labels.squeeze())
            running_loss += loss.item()

            true_labels.extend(labels.squeeze())
            y_scre.extend(outputs.squeeze())

        test_epoch_loss = running_loss / len(test)

        y_true = torch.stack(true_labels).cpu().squeeze()
        y_scre = torch.stack(y_scre).cpu().squeeze()
        y_pred = torch.round(y_scre)

        # This is required because wandb has weird requirements for argument
        # ro ROC curve plot.
        y_scre_per_class = torch.hstack(
                [(1 - y_scre).unsqueeze(-1),
                    y_scre.unsqueeze(-1)]
        )
        wandb.log({
            "MDR": plot_mdr(y_true, y_scre),
            "Epoch": epoch,
            "Train Loss": train_epoch_loss,
            "Test Loss": test_epoch_loss,
            "roc": wandb.plot.roc_curve(
                                    y_true=y_true,
                                    y_probas=y_scre_per_class),
            "Test AUC": roc_auc_score(y_true, y_scre),
            "Test Precision": precision_score(y_true, y_pred),
            "Test Recall": recall_score(y_true, y_pred),
        })

        ckpt_filename = f"checkpoints/model_checkpoint_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), ckpt_filename)
        wandb.save(ckpt_filename)

    return model
