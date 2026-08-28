import contextlib
import os

import torch

"""
Utilities for computing and storing Fisher information matrices and 
corresponding model parameters, typically used in Elastic Weight 
Consolidation (EWC) or similar continual learning regularization methods.

Functions:
    - compute_and_save_fisher_and_params: computes Fisher diagonals and saves them with model params
    - load_fisher_and_params: loads previously saved Fisher information and parameters
"""


def _backward_context(model, device):
    """
    Context in which the Fisher pass may back-propagate.

    The pass deliberately runs the model in evaluation mode, but cuDNN refuses
    to back-propagate through a recurrent layer outside training mode
    ("cudnn RNN backward can only be called in training mode").  For models that
    contain a recurrent layer the cuDNN path is therefore switched off for the
    duration of the pass; the native kernels produce the same gradients.  Purely
    convolutional models - every NIST and CIFAR-10 topology - keep cuDNN
    enabled, so their Fisher information is computed exactly as before.
    """
    is_cuda = str(device).startswith("cuda")
    has_recurrent = any(isinstance(module, torch.nn.RNNBase) for module in model.modules())
    if is_cuda and has_recurrent:
        return torch.backends.cudnn.flags(enabled=False)
    return contextlib.nullcontext()


def compute_and_save_fisher_and_params(
    model, dataloader, criterion, device, fisher_path, max_batches=None
):
    """
    Compute and save Fisher information and model parameters.

    Runs the model in evaluation mode over the provided dataloader, computes
    the diagonal Fisher information matrix (squared gradients of the loss),
    and saves both the Fisher values and a snapshot of the current model
    parameters to disk.

    The routine is dataset agnostic: it only needs a loader that yields
    ``(inputs, labels)`` and a model whose ``forward`` accepts those inputs, so
    it serves NIST, LEAF Shakespeare and CIFAR-10 unchanged.

    Args:
        model (torch.nn.Module): The model to analyze.
        dataloader (torch.utils.data.DataLoader): Data used to compute gradients.
        criterion (torch.nn.Module): Loss function.
        device (torch.device or str): Device to perform computations on.
        fisher_path (str): Directory path where results will be saved.
        max_batches (int, optional): Stop after this many batches and average
            over them.  ``None`` (the default) uses the whole loader, which is
            what every published run did.

    Saves:
        fisher.pt         : dict mapping parameter names -> Fisher diagonals.
        global_params.pt  : dict mapping parameter names -> parameter tensors.
"""


    model.eval()

    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    global_params = {n: p.clone().detach().to(device) for n, p in model.named_parameters() if p.requires_grad}

    batches = 0
    with _backward_context(model, device):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += (p.grad.detach() ** 2)

            batches += 1
            if max_batches is not None and batches >= int(max_batches):
                break

    divisor = batches if max_batches is not None else len(dataloader)
    divisor = max(divisor, 1)
    for n in fisher:
        fisher[n] /= divisor

    # Save both to a file
    os.makedirs(fisher_path, exist_ok=True)
    torch.save(fisher, os.path.join(fisher_path, "fisher.pt"))
    torch.save(global_params, os.path.join(fisher_path, "global_params.pt"))

def load_fisher_and_params(fisher_path, device):
    """
    Load Fisher information and model parameters from disk.

    Args:
        fisher_path (str): Directory containing 'fisher.pt' and 'global_params.pt'.
        device (torch.device or str): Device to map loaded tensors to.

    Returns:
        tuple:
            fisher (dict): Fisher information per parameter.
            global_params (dict): Saved model parameters.
    """

    fisher = torch.load(os.path.join(fisher_path, "fisher.pt"), map_location=device)
    global_params = torch.load(os.path.join(fisher_path, "global_params.pt"), map_location=device)
    return fisher, global_params