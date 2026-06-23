import os
import cv2
import torch
from torchvision import transforms
from cnn import CNN

model = CNN()
model.load_state_dict(torch.load("model/handwriting_model.pth"))
model.eval()
files = os.listdir("characters")
print(files)
files = os.listdir("characters")
transform = transforms.ToTensor()
for file in files:
    img = cv2.imread(
        f"characters/{file}",
        cv2.IMREAD_GRAYSCALE
    )
    print(file, img.shape)

    img = cv2.transpose(img)
    img = cv2.flip(img, 0)
    img = cv2.resize(img, (28, 28))
    img = transform(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        prediction = torch.argmax(
            output,
            dim=1
        )
    letter = chr(
        prediction.item() + ord('A')
    )
    print(file, "->", letter)