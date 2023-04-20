import torch 
import logging

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
        epoch_loss = running_loss / len(train)
        logging.info(f"Train loss: {epoch_loss:.03f}")
        model.eval()
        running_loss = 0.
        for i, (inputs, labels) in enumerate(test):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = metric(outputs.squeeze(), labels.squeeze())
            running_loss += loss.item()
        epoch_loss = running_loss / len(test)
        logging.info(f"Test loss: {epoch_loss:.03f}")
    return model
