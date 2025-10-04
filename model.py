import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

# --- 1. Configuration ---
PROJECT_ROOT = '.'
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'train')
TEST_DIR = os.path.join(PROJECT_ROOT, 'test')
MODEL_SAVE_PATH = 'skin_disease_efficientnet_b0_finetuned.pth'

NUM_CLASSES = 21
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

# --- Hyperparameters for the two training stages ---
# Stage 1: Train only the classifier head
EPOCHS_STAGE_1 = 15
LR_STAGE_1 = 0.001

# Stage 2: Fine-tune the entire model
EPOCHS_STAGE_2 = 30
LR_STAGE_2 = 0.00005 # A very small learning rate is crucial for fine-tuning

# --- 2. Custom Dataset (Unchanged) ---
class SkinDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        class_names = sorted(os.listdir(root_dir))
        for i, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = i
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

# --- 3. More Advanced Data Augmentation ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # Add Random Erasing to make the model more robust
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 4. Dataset and DataLoader Creation ---
print("Loading datasets...")
train_dataset = SkinDiseaseDataset(root_dir=TRAIN_DIR, transform=data_transforms['train'])
val_dataset = SkinDiseaseDataset(root_dir=TEST_DIR, transform=data_transforms['val'])
print(f"Classes found: {len(train_dataset.class_to_idx)}")

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# --- NEW: Function to Calculate Class Weights ---
def calculate_class_weights(dataset):
    """Calculates class weights to handle imbalance."""
    labels = np.array(dataset.labels)
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float)

# --- 5. Model Definition (Unchanged) ---
def get_model(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# --- 6. The Optimized Training Loop (same loop, used for both stages) ---
def train_model(model, criterion, optimizer, scheduler, num_epochs, current_stage):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"--- Starting {current_stage} on {device} ---")
    
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_acc = 0.0

    # Load the best accuracy from the previous stage if fine-tuning
    if current_stage.startswith("Stage 2"):
        # We start by checking the accuracy of the loaded model
        model.eval()
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
        best_acc = running_corrects.double() / dataset_sizes['val']
        print(f"Loaded model from Stage 1. Initial validation accuracy: {best_acc:.4f}")


    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"New best model saved with accuracy: {best_acc:.4f}")

    print(f'Best val Acc during {current_stage}: {best_acc:4f}')
    return model

if __name__ == '__main__':
    # --- Get the Model ---
    model = get_model(NUM_CLASSES)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Calculate Class Weights and define weighted loss function ---
    class_weights = calculate_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    print(f"Using class weights: {class_weights}")

    # --- STAGE 1: TRAIN THE CLASSIFIER HEAD ---
    # Freeze the feature extractor
    for param in model.features.parameters():
        param.requires_grad = False
    
    optimizer_stage1 = optim.Adam(model.classifier.parameters(), lr=LR_STAGE_1)
    scheduler_stage1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=EPOCHS_STAGE_1)

    model = train_model(model, criterion, optimizer_stage1, scheduler_stage1, num_epochs=EPOCHS_STAGE_1, current_stage="Stage 1: Feature Extraction")

    # --- STAGE 2: FINE-TUNE THE ENTIRE MODEL ---
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    optimizer_stage2 = optim.Adam(model.parameters(), lr=LR_STAGE_2)
    scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=EPOCHS_STAGE_2)
    
    # Load the best weights from stage 1 before starting stage 2
    print(f"\nLoading best model from Stage 1 saved at: {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    model = train_model(model, criterion, optimizer_stage2, scheduler_stage2, num_epochs=EPOCHS_STAGE_2, current_stage="Stage 2: Fine-Tuning")