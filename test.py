import os
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from model.efficientghost import efficientnet

classes = ('A', 'B')


transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dir ='features/data'
test_dataset = ImageFolder(root=test_dir, transform=transform_test)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
num_classes=2
model = efficientnet(num_classes=num_classes)
num_ftrs = model.classifier[-1].in_features

model.classifier[-1] = nn.Sequential(

    nn.Dropout(0.5),
    nn.Linear(num_ftrs, num_classes)
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('./best_model.pth', map_location=device)
model.load_state_dict(state_dict, strict=True)
model.to(device)
model.eval()

correct = 0
total = 0
y_true = []
y_pred = []
y_scores = []

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)

    correct += (preds == labels).sum().item()
    total += labels.size(0)
    y_true.extend(labels.cpu().numpy())
    y_pred.extend(preds.cpu().numpy())
    y_scores.extend(probs[:, 1].cpu().numpy())

    for i in range(len(labels)):
        print(f"Image Name: {test_dataset.imgs[total - labels.size(0) + i][0].split(os.sep)[-1]}, "
              f"True Label: {classes[labels[i].item()]}, "
              f"Predicted: {classes[preds[i].item()]}")


accuracy = correct / total * 100
print('Accuracy: {:.2f}%'.format(accuracy))


conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


print(classification_report(y_true, y_pred, target_names=classes, digits=4))


fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

print(f"AUC: {roc_auc:.4f}")