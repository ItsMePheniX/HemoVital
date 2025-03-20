def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc="Evaluating"):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos).logits
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct/total:.4f}")


evaluate_model(model, train_loader)  
