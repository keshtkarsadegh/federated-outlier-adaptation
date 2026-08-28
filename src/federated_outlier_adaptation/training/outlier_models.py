import copy
import json
from tqdm import tqdm
import torch

from torch import optim, nn
import matplotlib.pyplot as plt

from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.providers import default_provider

# === Paths ===

base_learning_rate = 1e-3
base_weight_decay = 1e-4
"""
outlier_training.py

Purpose:
    Train personalized models for a set of outlier writers in the NIST dataset.
    Each outlier is trained independently, evaluated on global and client-level
    test sets, and the results are aggregated.

Key Features:
    - Builds per-writer CNN models (FlexibleCNN) with independent training loops.
    - Tracks and logs training/validation accuracy across epochs.
    - Selects the best checkpoint per writer based on validation accuracy.
    - Evaluates best models on:
        * Global writer test set
        * All selected outlier clients' test set
    - Saves results to JSON and plots accuracy curves for each writer.

Inputs:
    - provider: dataset provider; defaults to the process-wide NIST one.
    - global_writers: list of IDs for global writers used as test reference.
    - selected_5_writers: list of outlier writer IDs for training.
    - batch_size: mini-batch size for loaders.
    - num_epochs: number of training epochs.
    - outliers_results_path: directory to save JSON results and plots.

Outputs:
    - outlier_training_results.json (per-writer metrics)
    - <writer_id>_accuracy_plot.png (training vs validation + test refs)

Dependencies:
    - torch, tqdm, matplotlib
    - federated_outlier_adaptation.model.FlexibleCNN
    - federated_outlier_adaptation.data.datasets.NistDataset
    - federated_outlier_adaptation.logging_utils.NistLogger
"""


def outliers_train(
    global_writers, selected_5_writers, batch_size, num_epochs, outliers_results_path, provider=None
):
    """
        Train and evaluate CNN models for a group of selected outlier writers.

        Args:
            global_writers (list[str] or list[int]): Writer IDs considered "global",
                used to build the global test loader for evaluation.
            selected_5_writers (list[str] or list[int]): IDs of outlier writers
                to train personalized models for.
            batch_size (int): Mini-batch size for training and evaluation.
            num_epochs (int): Number of epochs to train each outlier model.
            outliers_results_path (Path): Directory where plots and JSON
                metrics will be saved.

        Workflow:
            1. For each writer in `selected_5_writers`:
                - Build a new FlexibleCNN.
                - Train with writer-specific data (60/40 train/val split).
                - Track per-epoch training/validation accuracy.
                - Keep the checkpoint with the best validation accuracy.
            2. Evaluate the best checkpoint on:
                - Global test set (`global_writers`)
                - All selected outlier test set (`selected_5_writers`)
            3. Log results and plot accuracy curves.

        Side Effects:
            - Saves PNG plots per writer: "<writer_id>_accuracy_plot.png"
            - Aggregates all metrics into JSON: "outlier_training_results.json"
            - Logs training/validation/test results via NistLogger.

        Degenerate writers:
            A writer whose 60/40 split yields no training or no validation batch
            is **skipped with a warning** and recorded in the JSON under
            ``skipped``.  This step is a tail of the global-training phase, run
            after theta_g has already been trained and saved, so one writer with
            too little data must not destroy an otherwise finished job.  It is
            not hypothetical: at 62 classes some writers hold as few as 18
            images, and 40 % of a writer that small can round down to an empty
            validation split.

        Returns:
            dict: Per-writer metrics, also persisted as files and logs.
        """
    #`` === Track results ===
    results = {}
    provider = provider or default_provider()

    # === Train per writer ===
    for writer_id in selected_5_writers:
        # === Load personal data ===
        # Built before the model, so a writer that cannot be trained costs
        # nothing and is reported before any work is done on it.
        train_loader, eval_loader,_ = provider.build_dataset(writer_id,train_rate=0.6,eval_rate=0.4,batch_size=batch_size)

        empty = [
            name
            for name, loader in (("train", train_loader), ("validation", eval_loader))
            if loader is None or len(loader.dataset) == 0
        ]
        if empty:
            try:
                held = int(provider.sample_count(writer_id))
            except (AttributeError, KeyError, TypeError):  # pragma: no cover - exotic provider
                held = None
            sizes = {
                name: (0 if loader is None else len(loader.dataset))
                for name, loader in (("train", train_loader), ("validation", eval_loader))
            }
            NistLogger.warning(
                f"Skipping the per-writer model of {writer_id}: its "
                f"{' and '.join(empty)} split is empty "
                f"({sizes}, {held} sample(s) in total). A 60/40 split of a very "
                "small writer can round down to nothing; the writer is left out "
                "of this reference and the remaining writers are unaffected."
            )
            results[writer_id] = {
                "skipped": True,
                "reason": f"empty {'/'.join(empty)} split",
                "split_sizes": sizes,
                "samples": held,
            }
            continue

        best_client_model=None
        best_val_acc=-1
        client_model = provider.make_model()
        client_model = client_model.to('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer = optim.Adam(client_model.parameters(), lr=base_learning_rate, weight_decay=base_weight_decay)
        criterion =  nn.CrossEntropyLoss()

        train_accuracies = []
        val_accuracies = []

        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):

            client_model.train()
            correct, total = 0, 0

            for images, labels in train_loader:
                images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                optimizer.zero_grad()
                outputs = client_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            train_accuracies.append(train_acc)

            client_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in eval_loader:
                    images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                    outputs = client_model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / total
            val_accuracies.append(val_acc)

            NistLogger.info(f"[{writer_id}] Epoch {epoch + 1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_client_model=copy.deepcopy(client_model)
        client_model = copy.deepcopy(best_client_model)
        # (Optional) Move to device
        client_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        client_model.eval()  # If you
        # === Evaluate on full global writer test set ===
        _, _,global_test_loader = provider.build_dataset(global_writers,train_rate=0.0,eval_rate=0.0,batch_size=batch_size)


        global_test_acc = 0.0
        correct, total = 0, 0
        if global_test_loader and len(global_test_loader.dataset) > 0:
            with torch.no_grad():
                for images, labels in global_test_loader:
                    images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                    outputs = client_model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            if total > 0:
                global_test_acc = correct / total
        _, _,all_client_test_loader = provider.build_dataset(selected_5_writers,train_rate=0.0,eval_rate=0.0,batch_size=batch_size)

        client_test_acc = 0.0
        correct, total = 0, 0
        if all_client_test_loader and len(all_client_test_loader.dataset) > 0:
            with torch.no_grad():
                for images, labels in all_client_test_loader:
                    images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                    outputs = client_model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            if total > 0:
                client_test_acc = correct / total
        results[writer_id] = {
            "train_accuracies": train_accuracies,
            "eval_accuracies": val_accuracies,
            "all_client_test_accuracies": client_test_acc,
            "global_test_accuracy": global_test_acc
        }

        # === Plot accuracy ===
        plt.figure()
        plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(val_accuracies, label="Validation Accuracy")
        plt.axhline(y=client_test_acc, color='b', linestyle='-', label=f"Test Accuracy on All Clients Data : {client_test_acc:.2f}")
        plt.axhline(y=global_test_acc, color='r', linestyle='--', label=f"Test Accuracy on Global Data: {global_test_acc:.2f}")
        plt.title(f"Accuracy – {writer_id}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outliers_results_path / f"{writer_id}_accuracy_plot.png")
        plt.close()

    # === Save accuracy data as JSON ===
    json_path =outliers_results_path / "outlier_training_results.json"
    outliers_results_path.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    NistLogger.info(f"\nSaved all accuracy data to: {json_path}")
    skipped = [wid for wid, entry in results.items() if entry.get("skipped")]
    if skipped:
        NistLogger.warning(
            f"{len(skipped)} of {len(selected_5_writers)} writer(s) were skipped "
            f"for want of data: {skipped}"
        )
    return results
