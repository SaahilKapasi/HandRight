import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, load
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

training_data = datasets.EMNIST(
    root="data",
    split='byclass',
    train=True,
    download=True,
    transform=ToTensor(),
)

# Create a dataloader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 62)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

p = transforms.Compose([transforms.Resize((28, 28))])


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def response(canvas_path: str, expected_char: str):
    """
    Predicts the character from the canvas image and compares it to the expected character,
    returning a confidence score for the prediction.
    """
    # Load the trained model weights
    with open('image_recognition.pth', 'rb') as f:
        model.load_state_dict(load(f, map_location=device))

    # Open and preprocess the canvas image
    canvas_img = Image.open(canvas_path).convert('L')
    canvas_img = p(canvas_img)  # Apply the transformations

    # Convert the canvas image to tensor
    canvas_tensor = ToTensor()(canvas_img).unsqueeze(0).to(device)

    # Get the model's prediction for the canvas image
    model.eval()
    with torch.no_grad():
        canvas_logits = model(canvas_tensor)
        canvas_probs = F.softmax(canvas_logits, dim=1)
        canvas_pred_index = torch.argmax(canvas_probs, dim=1).item()
        confidence_score = torch.max(canvas_probs).item()

    # Convert the predicted index to character
    if canvas_pred_index < 10:
        # Map numbers 0-9 to their corresponding ASCII characters
        canvas_pred_char = str(canvas_pred_index)
    elif 10 <= canvas_pred_index < 36:
        # Map uppercase letters A-Z to their corresponding ASCII characters
        canvas_pred_char = chr(canvas_pred_index + 55)
    else:
        # Map lowercase letters a-z to their corresponding ASCII characters
        canvas_pred_char = chr(canvas_pred_index + 61)

    # Print and return the confidence score
    print(f"Canvas predicted character: {canvas_pred_char}")
    print(f"Expected character: {expected_char}")
    print(f"Confidence score of the prediction: {confidence_score:.2f}")

    return [canvas_pred_char, confidence_score]


if __name__ == "__main__":
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
    print("Done!")
    torch.save(model.state_dict(), 'image_recognition.pth')
