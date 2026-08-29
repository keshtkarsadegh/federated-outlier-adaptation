
import json

import torch

from torch import optim, nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from federated_outlier_adaptation.logging_utils import NistLogger

base_learning_rate = 1e-3
base_weight_decay = 1e-4

"""
Purpose:
    Implements training of a global model that includes outlier client data
    in federated adaptive learning experiments. The routine trains on combined
    data, tracks accuracy, saves best checkpoints, evaluates on both global
    and outlier test sets, and outputs JSON metrics and plots.

Key features:
    - Optimizer: Adam with configurable base learning rate and weight decay
    - Loss: CrossEntropy
    - Metrics: train/val accuracies across epochs, final test accuracy
    - Outputs: JSON logs + Matplotlib accuracy plots

The loop is dataset agnostic: it consumes loaders that yield ``(inputs,
labels)`` and a model whose ``forward`` accepts those inputs, so the same
routine produces the combined reference model.  The variable name
``images`` is kept from the published code.

Dependencies:
    - torch, tqdm, matplotlib
    - NistLogger (for structured experiment logging)

Outputs saved:
    - `global_clients_global_metrics.json`
    - `global_clients_all_outliers_metrics.json`
    - Training/validation/test accuracy plots (.png)
"""

def global_outliers_training(global_model,train_loader, eval_loader,all_outliers_test_loader,global_test_loader,global_clients_path,results_path,num_epochs,patience=None,min_epochs=0):
    """
       Train a global model using both inlier and outlier client data,
       validate across epochs, and evaluate on separate global and outlier test sets.

       Args:
           global_model (torch.nn.Module): Model to be trained.
           train_loader (DataLoader): Training data (clients + outliers).
           eval_loader (DataLoader): Validation data for monitoring accuracy.
           all_outliers_test_loader (DataLoader): Test set containing outlier clients.
           global_test_loader (DataLoader): Test set containing all global clients.
           global_clients_path (str or Path): File path to save the best model weights.
           results_path (str or Path): Directory to store JSON metrics and plots.
           num_epochs (int): Maximum number of training epochs.
           patience (int, optional): Non-improving epochs tolerated before the
               training stops.  ``None`` (the default) runs every epoch, which
               is the published behaviour.
           min_epochs (int): Epochs that always run before patience may fire.

       Which weights are reported:
           The loop always saved the best-validation checkpoint to disk, but then
           evaluated the *last* epoch's weights - so the model on disk and the
           test accuracy reported next to it were two different models.  The best
           weights are now restored before the final evaluation, which makes the
           checkpoint and its reported numbers the same thing.

       Workflow:
           1. Train model on `train_loader` with Adam optimizer and CrossEntropy loss.
           2. Track training and validation accuracy each epoch.
           3. Save checkpoint when validation accuracy improves.
           4. Evaluate final model on:
               - Global test data
               - Outlier test data
           5. Save accuracies to JSON and plot curves for visualization.

       Returns:
           None. Side effects:
               - Writes model checkpoint
               - Saves JSON metrics
               - Exports PNG plots
               - Logs training/validation/test accuracy with NistLogger
       """
    base_device= 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer=optim.Adam(global_model.parameters(), lr=base_learning_rate, weight_decay=base_weight_decay)
    criterion=nn.CrossEntropyLoss()

    from federated_outlier_adaptation.training.convergence import EarlyStopper

    # === Accuracy Tracking ===
    train_accuracies = []
    val_accuracies = []
    stopper = EarlyStopper(patience=patience, min_epochs=min_epochs)

    # === Training Loop ===
    for epoch in range(num_epochs):
        global_model.train()
        total_loss, correct, total = 0.0, 0, 0
        data_iter = tqdm(train_loader, desc=f"[Global+outliers training] Epoch {epoch + 1}/{num_epochs}")

        for images, labels in data_iter:
            images, labels = images.to(base_device), labels.to(base_device)
            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_accuracies.append(train_acc)

        # === Validation ===
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in eval_loader:
                images, labels = images.to(base_device), labels.to(base_device)
                outputs = global_model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        val_accuracies.append(val_acc)

        NistLogger.info(f"Epoch {epoch + 1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        improved = val_acc > stopper.best_accuracy
        should_stop = stopper.update(epoch, val_acc, global_model)
        if improved:
            torch.save(global_model.state_dict(), global_clients_path)
            NistLogger.info(f"Saved new best model with val_acc={val_acc:.4f} at {global_clients_path}")
        if should_stop:
            NistLogger.info(
                f"Early stopping (patience={patience}, min_epochs={min_epochs}) "
                f"after {stopper.epochs_run} epochs; keeping epoch "
                f"{stopper.best_epoch + 1}."
            )
            break

    # The checkpoint on disk is the best validation epoch; the evaluation below
    # has to be of that same model, not of whatever the last epoch produced.
    best = stopper.best()
    if best is not None:
        global_model.load_state_dict(best.state_dict())

    # === Final Test Evaluation ===
    global_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in global_test_loader:
            images, labels = images.to(base_device), labels.to(base_device)
            outputs = global_model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    NistLogger.info(f"Final Test Accuracy on all global: {test_acc:.4f}")

    # === Save Metrics as JSON ===
    metrics = {
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "test_accuracy": test_acc,
        "convergence": stopper.metrics(),
    }
    with open(results_path/"global_clients_global_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    NistLogger.info(f"Saved metrics JSON to {results_path/'global_clients_metrics.json'}")

    # === Save Accuracy Plot ===
    plt.figure()
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.axhline(y=test_acc, color='r', linestyle='--', label=f"Test Accuracy on Global Data: {test_acc:.4f}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path/"global_trained_by_outliers_onglobaldata_metrics.png")
    NistLogger.info(f"Saved accuracy plot to {results_path/"global_outliers_plot.png"}")


    # === Final Test Evaluation ===
    global_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in all_outliers_test_loader:
            images, labels = images.to(base_device), labels.to(base_device)
            outputs = global_model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    NistLogger.info(f"Final Test Accuracy on all Clients/Outliers: {test_acc:.4f}")

    # === Save Metrics as JSON ===
    metrics = {
        "test_accuracy": test_acc
    }
    with open(results_path/"global_clients_all_outliers_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    NistLogger.info(f"✅ Saved metrics JSON to {results_path/"global_clients_all_outliers_metrics.json"}")

    # === Save Accuracy Plot ===
    plt.figure()
    plt.axhline(y=test_acc, color='r', linestyle='--', label=f"Test Accuracy: {test_acc:.4f}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy on all Clients/Outliers Data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path/"global_trained_by_outliers_onoutliersdata_metrics.png")
    NistLogger.info(f"Saved accuracy plot to {results_path/"global_clients_all_outliers_metrics.png"}")

    plt.figure()
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.axhline(y=test_acc, color='r', linestyle='--', label=f"Test Accuracy on Global Data: {test_acc:.4f}")
    plt.axhline(y=test_acc, color='r', linestyle='--', label=f"Test Accuracy on Ouliers Data: {test_acc:.4f}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path / "global_trained_by_outliers_onglobaldata_metrics_combiend.png")
    NistLogger.info(f"Saved accuracy plot to {results_path / "global_outliers_plot_combined.png"}")


