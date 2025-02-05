# Exam-Group14-YIBS-2025
This repository contains scripts for training an AI model on datasets like Fashion-MNIST. The project covers data preprocessing, model training, and saving the trained model for inference. Features      Dataset Handling: Load and preprocess Fashion-MNIST or similar datasets.     

Here's a comprehensive guide organized by your requirements:

---

# **Fashion MNIST Classifier - Complete Guide**  
**Folder Structure**  
```bash
fashion-classifier/  
├── model_training/  
│   ├── fashion_mnist_cnn.pth  
│   └── fashion_mnist_training.ipynb  
├── app/  
│   ├── static/  
│   │   └── uploads/  
│   ├── templates/  
│   │   └── index.html  
│   └── app.py  
```

---

## **Part 1: Model Training on Google Colab**  

### **Step 1: Open Google Colab**  
1. Go to [Google Colab](https://colab.research.google.com/)  
2. Create a new notebook: `File > New Notebook`  

### **Step 2: Install Dependencies**  
```python
!pip install torch torchvision matplotlib kagglehub
```

### **Step 3: Upload Dataset**  
```python
import kagglehub
path = kagglehub.dataset_download("zalando-research/fashionmnist")
```

### **Step 4: Full Training Code**  
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*5*5, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.network(x)

# Load data
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Initialize training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1} completed')

# Save model
torch.save(model.state_dict(), 'fashion_mnist_cnn.pth')
```

### **Step 5: Download Trained Model**  
Run this after training completes:  
```python
from google.colab import files
files.download('fashion_mnist_cnn.pth')
```

---

## **Part 2: Running the Web Application**  

### **Prerequisites**  
1. Python 3.8+ installed  
2. Required folder structure (create missing folders):  
   ```bash
   mkdir -p app/static/uploads app/templates
   ```

### **Step 1: Install Dependencies**  
```bash
pip install flask torch torchvision pillow
```

### **Step 2: Organize Files**  
Place these files in their respective folders:  
- `app.py` (root directory)  
- `index.html` (in `templates/`)  
- `fashion_mnist_cnn.pth` (root directory)  

### **Step 3: Run the Application**  
```bash
# Windows
set FLASK_APP=app.py
flask run

# Linux/macOS
export FLASK_APP=app.py
flask run
```

### **Step 4: Access Locally**  
Visit `http://localhost:5000` in your browser.  

---

## **Live Deployment (Optional)**  
For temporary public access:  

### **Using ngrok**  
1. Install ngrok: https://ngrok.com/download  
2. Run:  
```bash
ngrok authtoken YOUR_AUTH_TOKEN
ngrok http 5000
```  
3. Use the generated HTTPS link (e.g., `https://abc123.ngrok.io`)  

### **Heroku Deployment (Permanent)**  
1. Create `requirements.txt`:  
```txt
flask
torch
torchvision
pillow
```
2. Create `Procfile`:  
```txt
web: gunicorn app:app
```
3. Deploy via Heroku CLI:  
```bash
heroku create
git push heroku main
```

---

# **Live Demo**  
Temporary test link (via ngrok):  
`https://your-ngrok-subdomain.ngrok.io`  

---

**Notes**  
- Batch uploads supported (up to 10 images at once)  
- Model will automatically classify all uploaded images  
- Predictions show with confidence scores and previews  

Let me know if you need clarification on any step!