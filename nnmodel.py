from torchvision.transforms import ToTensor
from torchvision import datasets

train_data = datasets.MNIST(
    root = 'MNIST_data',
    train = True,
    transform = ToTensor(),
    download = True
)

test_data = datasets.MNIST(
    root = 'MNIST_data',
    train = False,
    transform = ToTensor(),
    download = True
)

print(train_data) # 60,000 datapoints
print(test_data) # 10,000 datapoints

#train_data.data_shape
# returns torch.Size([60000,28,28])

train_data.targets.size()
# returns torch.Size([60000])

# Want to load training data in batches

from torch.utils.data import DataLoader

loaders = {
    'train' : DataLoader(train_data,
                         batch_size=100,
                         shuffle=True,
                         num_workers = 1),
    'test' : DataLoader(test_data,
                         batch_size=100,
                         shuffle=True,
                         num_workers = 1),
}

# Define deep learning architecture

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #optimizer

#convolutional neural network
class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)    
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.conv2_drop = nn.Dropout2d() # deactivates certain neurons in network during training
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # call activation functions manually
    def forward(self, x): 
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x)
    
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    # Put model into training mode
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\t{loss.item():.6f}')

def test():
    # Put model into evaluation mode
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim = 1, keepdim = True)
            # Compare if target and predicted class are the same
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders['test'].dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%\n)')







if __name__ == "__main__":
    for epoch in range(1,11):
        train(epoch)
        test()

    import matplotlib.pyplot as plt

    model.eval()

    data, target = test_data[0]

    data = data.unsqueeze(0).to(device)

    output = model(data)

    prediction = output.argmax(dim=1, keepdim=True).item()

    print(f'Prediction: {prediction}')

    image = data.squeeze(0).squeeze(0).cpu().numpy()

    plt.imshow(image, cmap='gray')
    plt.show()
