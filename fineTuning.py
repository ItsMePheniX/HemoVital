from transformers import TimeSformerForVideoClassification
import torch.optim as optim
import torch.nn as nn

# Load pre-trained TimeSformer
model = TimeSformerForVideoClassification.from_pretrained(
    "facebook/timesformer-base-finetuned-k400", num_labels=7
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Training function
def train_model(model, dataloader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for videos, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {correct/total:.4f}")

# Train the model
train_model(model, train_loader, optimizer, criterion, epochs=5)
