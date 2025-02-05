from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
import os
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load PyTorch model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 128)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN()
model.load_state_dict(torch.load('fashion_mnist_cnn.pth', map_location=torch.device('cpu')))
model.eval()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 28, 28]
    return img_tensor

@app.route('/', methods=['GET', 'POST'])
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
                prediction = model.predict(processed_image)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
                predictions.append({
                    'filename': filename,
                    'label': class_names[predicted_class],
                    'confidence': f"{confidence:.2f}%"
                })

        return render_template('index.html', predictions=predictions)
    
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)