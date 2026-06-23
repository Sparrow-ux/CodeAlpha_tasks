from torchvision.datasets import EMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn 
import torch.optim as optim 
from cnn import CNN


emnist_compose = transforms.Compose([lambda img: transforms.functional.rotate(img, -90, fill=0),lambda img: transforms.functional.hflip(img),transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),transforms.ToTensor()])
dataset = EMNIST(root="dataset",split="letters",train=True,download=True,transform=emnist_compose)

train_loader = DataLoader(dataset,batch_size=64,shuffle=True)

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

for epoch in range(10):
    for batch_idx, (images, labels) in enumerate(train_loader):
        labels = labels - 1
        outputs = model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1},"f"Batch {batch_idx}, "f"Loss: {loss.item():.4f}")


torch.save(model.state_dict(),"model/handwriting_model.pth")
print("Model Saved Successfully")