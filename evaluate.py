import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = '.'
TEST_DIR = os.path.join(PROJECT_ROOT, 'test')
MODEL_PATH = 'skin_disease_efficientnet_b0_finetuned.pth'
NUM_CLASSES = 21
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

class SkinDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        class_names = sorted(os.listdir(root_dir))
        for i, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = i
                self.idx_to_class[i] = class_name
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, filename))
                        self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, SyntaxError) as e:
            print(f"\nWarning: Skipping corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            image = self.transform(image)
        return image, label

def get_model(num_classes, model_path):

    model = models.efficientnet_b0(weights=None) 
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f"Model loaded successfully from {model_path}")
    return model

def evaluate_model(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() 

    all_preds = []
    all_labels = []

    with torch.no_grad(): 
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)

def plot_class_predictions(labels, preds, class_names):
    """
    Plots a bar chart showing correct and incorrect predictions for each class.
    """
    correct_counts = {name: 0 for name in class_names}
    incorrect_counts = {name: 0 for name in class_names}

    for label, pred in zip(labels, preds):
        class_name = class_names[label]
        if label == pred:
            correct_counts[class_name] += 1
        else:
            incorrect_counts[class_name] += 1

    data = []
    for name in class_names:
        data.append([name, correct_counts[name], 'Correct'])
        data.append([name, incorrect_counts[name], 'Incorrect'])

    import pandas as pd
    df = pd.DataFrame(data, columns=['Class', 'Count', 'Type'])

    plt.figure(figsize=(12, 10))
    sns.barplot(data=df, x='Count', y='Class', hue='Type', palette={'Correct': 'green', 'Incorrect': 'red'}, dodge=False)
    
    plt.title('Correct vs. Incorrect Predictions per Class')
    plt.xlabel('Number of Predictions')
    plt.ylabel('Disease Class')
    plt.tight_layout()
    plt.savefig('class_prediction_report.png') 
    plt.show()


if __name__ == '__main__':

    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = SkinDiseaseDataset(root_dir=TEST_DIR, transform=eval_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    class_names = [test_dataset.idx_to_class[i] for i in range(len(test_dataset.idx_to_class))]

    model = get_model(num_classes=NUM_CLASSES, model_path=MODEL_PATH)

    true_labels, predicted_labels = evaluate_model(model, test_dataloader)

    print("\n" + "="*50)
    print("           Classification Report")
    print("="*50 + "\n")

    report = classification_report(true_labels, predicted_labels, target_names=class_names, digits=4)
    print(report)

    overall_accuracy = np.sum(true_labels == predicted_labels) / len(true_labels)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print("\n" + "="*50 + "\n")

    plot_class_predictions(true_labels, predicted_labels, class_names)