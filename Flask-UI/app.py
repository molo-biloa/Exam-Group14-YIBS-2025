from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import os
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Updated CNN model class to match the enhanced architecture
class EnhancedCNN(torch.nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.network = torch.nn.Sequential(
            # Conv Block 1
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            # Conv Block 2
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            # Conv Block 3
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            # Classifier
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 3 * 3, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)

# Load the new model
model = EnhancedCNN()
model.load_state_dict(torch.load('fashion_mnist_cnn_v3.pth', map_location=torch.device('cpu')))
model.eval()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_descriptions = {
    'T-shirt/top': 'A lightweight, short-sleeved shirt, typically made of cotton.',
    'Trouser': 'A garment worn from the waist to the ankles, covering both legs separately.',
    'Pullover': 'A knitted garment for the upper body, typically put on over the head.',
    'Dress': 'A one-piece garment for women or girls covering the body and extending down over the legs.',
    'Coat': 'An outer garment worn outdoors, typically having sleeves and extending below the hips.',
    'Sandal': 'A light shoe with an openwork upper or straps attaching to the sole.',
    'Shirt': 'A garment for the upper body, typically with a collar, sleeves, and buttons down the front.',
    'Sneaker': 'A soft shoe with a rubber sole, suitable for sports or casual wear.',
    'Bag': 'A container made of flexible material with an opening at the top, used for carrying items.',
    'Ankle boot': 'A boot that covers the foot and extends just above the ankle.'
}


# Updated preprocessing to match training pipeline
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img)
    
    # Normalize to [-1, 1] range (matches training)
    img_array = (img_array / 255.0 - 0.5) / 0.5
    
    # Convert to tensor and add batch + channel dimensions
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img_tensor

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            files = request.files.getlist('file[]')
            if not files:
                raise ValueError("No files uploaded")

            predictions = []
            
            for file in files:
                if file and file.filename != '':
                    # Save the uploaded file
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    # Preprocess and predict
                    processed_image = preprocess_image(filepath)
                    with torch.no_grad():
                        outputs = model(processed_image)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                        predictions.append({
                            'filename': filename,
                            'label': class_names[predicted.item()],
                            'confidence': f"{confidence.item()*100:.2f}%",
                            'description': class_descriptions[class_names[predicted.item()]]
                        })

            return render_template('index.html', predictions=predictions)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('index.html', predictions=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)