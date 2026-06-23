from flask import Flask, render_template, request
import cv2
import torch
from torchvision import transforms
import sys
from cnn import CNN
sys.path.append('model')
import os


app = Flask(__name__)
model = CNN()
model.load_state_dict(torch.load("model/handwriting_model.pth"))
model.eval()
def predict_character(char_img):
    h, w = char_img.shape
    if h > w:
        new_h, new_w = 20, int(w*20/h)
    else:
        new_h, new_w = int(h*20/w), 20
    if new_w == 0: new_w = 1
    if new_h == 0: new_h = 1
    resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_top = (28 - new_h) // 2
    pad_bottom = 28 - new_h - pad_top
    pad_left = (28 - new_w) // 2
    pad_right = 28 - new_w - pad_left

    img_padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    transform = transforms.ToTensor()
    img_tensor = transform(img_padded).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1)

    return chr(prediction.item() + ord('A'))

@app.route("/", methods=["GET", "POST"])
def home():

    filename = None
    if request.method == "POST":
        image = request.files["image"]
        os.makedirs('static/uploads', exist_ok=True)
        os.makedirs('characters', exist_ok=True)
        image.save("static/uploads/" + image.filename)
        filepath = "static/uploads/" + image.filename
        img = cv2.imread(filepath)
        if img is None:
            return "Failed to load image. Please ensure the upload succeeded.", 400
        print("Step 1")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Step 2")
        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        print("Step 3")
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("Step 4")
        print("Contours Found:", len(contours))

        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= 50:
                valid_contours.append((x, y, w, h))
        valid_contours.sort(key=lambda item: item[0])

        widths = [item[2] for item in valid_contours]
        avg_width = sum(widths) / len(widths) if widths else 0
        space_threshold = avg_width * 0.6

        output = img.copy()
        predicted_text = ""
        prev_x_end = None

        for i, (x, y, w, h) in enumerate(valid_contours):
            if prev_x_end is not None:
                gap = x - prev_x_end
                if gap > space_threshold:
                    predicted_text += " "
            prev_x_end = x + w

            character = thresh[y:y+h, x:x+w]
            cv2.imwrite(f"characters/char_{i}.png",character)

            letter = predict_character(character)
            predicted_text += letter

            print(f"x={x}, y={y}, w={w}, h={h}, letter={letter}") 
            cv2.rectangle(output,(x, y),(x + w, y + h),(0, 255, 0),2)
            cv2.putText(output,letter,(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0, 255, 0),2)
        cv2.imwrite("static/uploads/boxed_" + image.filename,output)
        filename = "boxed_" + image.filename
    return render_template("index.html", filename=filename,predicted_text=predicted_text if 'predicted_text' in locals() else None)

if __name__ == "__main__":
    app.run(debug=True)