import torch 
import logging
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import wandb 

def do_training(model, optimizer, metric, train, test, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1} / {epochs}")
        model.train()
        running_loss = 0.

        for i, (inputs, labels) in enumerate(train):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = metric(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_epoch_loss = running_loss / len(train)
        logging.info(f"Train loss: {train_epoch_loss:.03f}")

        model.eval()
        running_loss = 0.
        
        true_labels = []
        predicted_labels = []

        for i, (inputs, labels) in enumerate(test):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = metric(outputs.squeeze(), labels.squeeze())
            running_loss += loss.item()

            true_labels.append(labels.squeeze())
            predicted_labels.append(outputs.squeeze())

        test_epoch_loss = running_loss / len(test)
        logging.info(f"Test loss: {test_epoch_loss:.03f}")

        true_labels = torch.cat(true_labels).cpu().squeeze()
        print(true_labels)
        predicted_labels = torch.cat(predicted_labels).cpu().squeeze()

        test_auc = roc_auc_score(true_labels, predicted_labels)
        test_precision = precision_score(true_labels, [round(x) for x in predicted_labels])
        test_recall = recall_score(true_labels, [round(x) for x in predicted_labels])
        test_confusion_matrix = confusion_matrix(true_labels, [round(x) for x in predicted_labels])

        wandb.log({
            "Train Loss": epoch_loss,
            "Test Loss": epoch_loss,
            "Test AUC": test_auc,
            "Test Precision": test_precision,
            "Test Recall": test_recall,
            "Test Confusion Matrix": wandb.plot.confusion_matrix(true_labels, [round(x) for x in predicted_labels], ["0", "1"]),
        })

        # Save model checkpoint
        torch.save(model.state_dict(), f"model_checkpoint_epoch_{epoch + 1}.pt")
        wandb.save(f"model_checkpoint_epoch_{epoch + 1}.pt")

    return model

