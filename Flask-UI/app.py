from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import os
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64*5*5, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

model = CNN()
model.load_state_dict(torch.load('fashion_mnist_cnn_v2.pth', map_location=torch.device('cpu')))
model.eval()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 28, 28]
    return img_tensor

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('file[]')
        predictions = []
        
        for file in files:
            if file:
                # Save file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process and predict
                processed_image = preprocess_image(filepath)
                with torch.no_grad():
                    outputs = model(processed_image)
                    confidence, predicted = torch.max(torch.softmax(outputs, dim=1), 1)
                    
                predictions.append({
                    'filename': filename,
                    'label': class_names[predicted.item()],
                    'confidence': f"{confidence.item()*100:.2f}%"
                })

        return render_template('index.html', predictions=predictions)
    
    return render_template('index.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)