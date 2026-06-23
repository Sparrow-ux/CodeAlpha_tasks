import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from cnn import CNN
import sys
img_path = "characters/char_0.png"
if len(sys.argv) > 1:
    img_path = sys.argv[1]

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Could not read image:", img_path)
    sys.exit()
h, w = img.shape
if h > w:
    new_h, new_w = 20, int(w*20/h)
else:
    new_h, new_w = int(h*20/w), 20

if new_w == 0: new_w = 1
if new_h == 0: new_h = 1

resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
pad_top = (28 - new_h) // 2
pad_bottom = 28 - new_h - pad_top
pad_left = (28 - new_w) // 2
pad_right = 28 - new_w - pad_left
img_padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

transform = transforms.ToTensor()
img_tensor = transform(img_padded).unsqueeze(0)

model = CNN()
model.load_state_dict(torch.load("model/handwriting_model.pth"))
model.eval()

with torch.no_grad():
    output = model(img_tensor)
    prediction = torch.argmax(output, dim=1)
letter = chr(prediction.item() + ord('A'))
print(f"Predicted letter for {img_path}: {letter}")