import torch


from federated_outlier_adaptation.logging_utils import NistLogger


def client_all_clients_evaluate( model,all_clients_test_loader):
    base_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(base_device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in all_clients_test_loader:
            images, labels = images.to(base_device), labels.to(base_device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    all_clients_acc = correct / total
    NistLogger.info(f"All Clients Test Accuracy: {all_clients_acc:.4f}")
    return all_clients_acc

def client_global_evaluate( model,global_test_loader):
    base_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(base_device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in global_test_loader:
            images, labels = images.to(base_device), labels.to(base_device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    global_test_acc = correct / total
    NistLogger.info(f"Global Test Accuracy: {global_test_acc:.4f}")
    return global_test_acc