import torch
from torchvision.datasets import EMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from cnn import CNN


emnist_compose = transforms.Compose([lambda img: transforms.functional.rotate(img, -90, fill=0),lambda img: transforms.functional.hflip(img),transforms.ToTensor()])
dataset = EMNIST(root="dataset",split="letters",train=False,download=True,transform=emnist_compose)
test_loader = DataLoader(dataset,batch_size=32,shuffle=False)
model = CNN()
model.load_state_dict(torch.load("model/handwriting_model.pth"))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        labels = labels - 1
        outputs = model(images)
        predictions = torch.argmax(outputs,dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")