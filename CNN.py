import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR
import torchvision
import matplotlib.pyplot as plt
import argparse
import copy
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sb
import pandas as pd
import time

# argument parser
parser = argparse.ArgumentParser()

# -save command line arg
parser.add_argument('-save', action='store_true', help='Saves the model after training')

# -load command line arg
parser.add_argument('-loadVal', action='store_true', help='Loads the saved model on validation data')

# -load command line arg
parser.add_argument('-loadTest', action='store_true', help='Loads the saved model on test data')

# -graph command line arg
parser.add_argument('-graph', action='store_true', help='Displays a training loss vs validation loss graph after training')

# -v command line arg
parser.add_argument('-v', action='store_true', help='Enables visualization of some example data')

# -img command line arg
parser.add_argument('-img', help="Predicts an image's class based on provided image folder path")

# -cm command line arg
parser.add_argument('-cm', action='store_true', help='Creates a confusion matrix for the loaded model (requires loadVal or loadTest)')

args = parser.parse_args()

img_path = args.img

# Define Convolution Neural Network (CNN) model
class CNN(nn.Module):
    ''' Convolutional Neural Network model '''
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False) # Convolution 1 (bias set to False due to batchnorm) 1st 3 because RGB 
        self.bn1 = nn.BatchNorm2d(32) # Normalise 32 feature maps
        self.pool = nn.MaxPool2d(2, 2) # Max pool
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False) # Convolution 2
        self.bn2 = nn.BatchNorm2d(64) # Normalise 64 feature maps
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128) # Normalise 128 feature maps
        # self.conv4 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        # self.bn4 = nn.BatchNorm2d(256) # Normalise 256 feature maps
        # self.conv5 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        # self.bn5 = nn.BatchNorm2d(512) # Normalise 512 feature maps
        self.flatten = nn.Flatten() # convert to 1 dimensional array
        self.fc1 = nn.Linear(128*8*8,1024) # first hidden layer after feature extraction      ( 4*4 if 128 resize )
        self.bn6 = nn.BatchNorm1d(1024) # Normalise the array
        self.dropout = nn.Dropout(p=0.2) # can play with that too
        self.fc2 = nn.Linear(1024, 512) # second hidden layer
        self.bn7 = nn.BatchNorm1d(512) # Normalise the array
        self.fc3 = nn.Linear(512, 5) # third hidden layer 6 = number of classes

    def forward(self, x):
        ''' Forward Pass '''
        x = F.relu(self.bn1(self.conv1(x))) # Perform convolution on input, batch normalise and pass output through ReLU
        x = self.pool(x) # Max pool output from ReLU
        x = F.relu(self.bn2(self.conv2(x))) # Perform convolution on input (max pool output), normalise and pass output through ReLU
        x = self.pool(x) # Max pool output from ReLU
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        # x = F.relu(self.bn4(self.conv4(x)))
        # x = self.pool(x)
        # x = F.relu(self.bn5(self.conv5(x)))
        # x = self.pool(x)
        x = self.flatten(x) # Flatten the output
        x = F.relu(self.bn6(self.fc1(x))) # Pass flattened output into fc1, normalise and apply the activation function to the output
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x))) # Pass output of ReLU'd fc1 into fc2, normalise and apply ReLU
        x = self.dropout(x)
        x = self.fc3(x) # Pass output of ReLU'd fc2 into fc3
        return x



# Define training model
def train(net, trainloader, criterion, optimizer, device):
    net.train() # model set to training mode
    running_loss = 0.0 # used to calculate the loss across batches
    for data in trainloader:
        inputs, labels = data   # retrieves the inputs and labels from data
        inputs, labels = inputs.to(device), labels.to(device) # sends to device
        optimizer.zero_grad() # sets all gradients to 0
        outputs = net(inputs) # Get predictions (calls net.forward)
        loss = criterion(outputs, labels) # calculates loss (compares predicted outputs to true outputs)     
        loss.backward() # propagates the loss backwards through the network
        optimizer.step() # Update the weights in the network
        running_loss += loss.item() # Updates the loss
        #print(labels)
    avg_loss = running_loss/len(trainloader) # calculates average loss of training data
    # exit(0)
    return avg_loss


# Define testing model
def test(net, testloader, criterion, device):
    net.eval() # model set to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad(): # Do not calculate gradients
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Sends to device
            outputs = net(inputs) # Get predictions (calls net.forward)
            loss = criterion(outputs, labels) # calculates loss (compares predicted outputs to true outputs)    
            running_loss += loss.item() # Updates the loss
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item() # calculates how many are correct
    accuracy = correct/total # calculates accuracy of predictions
    avg_loss = running_loss/len(testloader) # calculates average loss of testing data
    return accuracy, avg_loss 


def conf_matrix_percent(cnn, test_loader, criterion, device):
    cnn.eval()
    correct = 0
    total = 0
    n_classes = 5  # Number of gesture classes
    cm = np.zeros((n_classes, n_classes), dtype=int)

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                cm[labels[i]][predicted[i]] += 1

    test_acc = correct / total
    test_loss = 1 - test_acc  # Assuming 1 is 100%
    
    # Convert the confusion matrix to percentages
    cm_percent = cm / cm.sum(axis=1, keepdims=True) 


    return test_acc, test_loss, cm_percent

# Loads the specified model
def load(cnn, test_loader, criterion, device):
    print("Loading params...")
    try:
        cnn.load_state_dict(torch.load('C:/Users/Acer/Desktop/Uni/sem2/EEE4022S/ML/snapshot/cnn.pth')) #
        print("Done!")
        # In your code where you call conf_matrix, modify it to use conf_matrix_percent
        if args.cm:
            test_acc, test_loss, cm = conf_matrix_percent(cnn, test_loader, criterion, device)
            print(f"Test accuracy = {test_acc*100:.2f}%, Test loss = {test_loss:.4f}")
            CATEGORIES = ["Grabbing", "Lifting", "Pulling", "Pushing", "Patting"]
            df_cm = pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES)
            sb.heatmap(df_cm, annot=True, fmt=".1f", cmap="Greens")  # Use "Blues" colormap for blue color
            plt.xlabel("Predicted Labels")
            plt.ylabel("Ground Truth")
            plt.title("Confusion Matrix")
            plt.show()
        
    except:
        print("No model or accuracy file found to load")
    # finally:
        # exit()


# Saves the specified model
def save(cnn, test_loader, criterion, device, best_loss, best_params):
    # Loads in current best model and checks accuracy
    try:
        cnn.load_state_dict(torch.load('C:/Users/Acer/Desktop/Uni/sem2/EEE4022S/ML/snapshot/cnn.pth')) 
        _, saved_loss = test(cnn, test_loader, criterion, device)
    except:
        saved_loss = 1000
    # If accuracy of current model is better than saved model then overwrite saved model
    if (best_loss < saved_loss):
        torch.save(best_params,'C:/Users/Acer/Desktop/Uni/sem2/EEE4022S/ML/snapshot/cnn.pth') # Saves best model
        print("Model saved to snapshot/cnn.pth") 
    else:
        print("Saved model achieves lower validation loss. Discarding current model.")


# Confusion matrix during the test 
def conf_matrix(net, testloader, criterion, device):
    net.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Sends to device
            outputs = net(inputs) # Get predictions (calls net.forward)
            loss = criterion(outputs, labels) # calculates loss (compares predicted outputs to true outputs)  
            running_loss += loss.item() # Updates the loss  
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item() # calculates how many are correct
            y_true += labels.tolist()
            y_pred += pred.tolist()
    accuracy = correct/total # calculates accuracy of predictions
    avg_loss = running_loss/len(testloader) # calculates average loss of testing data
    c_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, avg_loss, c_matrix
    

def main():
    
    # Transform sequence
    transform_normal = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(), # 0-255 to 0-1
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    # Transform sequence for test data
    transform_test = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    BATCH_SIZE = 64 # play with that hyperparam (32,64,128)
    train_path = 'C:/Users/Acer/Desktop/Uni/sem2/EEE4022S/ML/data/train/'
    val_path = 'C:/Users/Acer/Desktop/Uni/sem2/EEE4022S/ML/data/validation/'
    test_path = 'C:/Users/Acer/Desktop/Uni/sem2/EEE4022S/ML/data/test/'

    # Training data
    train_data_1 = torchvision.datasets.ImageFolder(train_path, transform=transform_normal)

    # Validation data
    val_data_1 = torchvision.datasets.ImageFolder(val_path, transform=transform_normal)

    # Test data
    test_data = torchvision.datasets.ImageFolder(test_path, transform=transform_test)

    # Dataloader for training
    train_loader = DataLoader(train_data_1, batch_size=BATCH_SIZE, shuffle=True)

    # Dataloader for validation
    val_loader = DataLoader(val_data_1, batch_size=BATCH_SIZE, shuffle=False)
    
    # Dataloader for testing
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Identify device to use
    device = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available()
           else "cpu")
    
    # Print out device used
    print(f"Using {device} device")
    
    # Initialise CNN and send parameters to the device
    cnn = CNN().to(device)
    
    # Define loss function
    loss_fn = nn.CrossEntropyLoss() # appropriate for classification models
    
    # Loads on the model on validation data
    if args.loadVal: 
        load(cnn, val_loader, loss_fn, device)
        return
    
    # Loads the model on test data
    if args.loadTest: 
        load(cnn, test_loader, loss_fn, device)
        return


    # -------------------------------- TRAIN THE MODEL --------------------------------------


    # optimizer parameters
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-6

    # learning rate decay parameters
    STEP_SIZE=7
    GAMMA=0.1

    # Define optimizer
    optimizer = optim.Adam(cnn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # can try SGD optimizer too (0.01 for learning rate)

    # Define learning rate scheduler
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
            
    # tracks the best accuracy
    best_loss = 1000 
    best_accuracy = 0
    best_params = None
    
    print("\nParameters:")
    print("-----------")
    print("-> Learning Rate: ", LEARNING_RATE)
    print("-> Learning Rate Decay: ", GAMMA)
    print("-> Learning Rate Decay Step: ", STEP_SIZE)
    print("-> Weight Decay: ", WEIGHT_DECAY)
    print("-> Batch Size: ", BATCH_SIZE)
    print("---------------------------------------")
    
    train_loss_array = []
    val_loss_array = []
    
    # Train the model for 20 epochs
    for epoch in range(10):
        train_loss = train(cnn, train_loader, loss_fn, optimizer, device)
        val_acc, val_loss = test(cnn, val_loader, loss_fn, device)
        
        # track losses over the epochs
        train_loss_array.append(train_loss)
        val_loss_array.append(val_loss)
        
        scheduler.step()
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Validation loss = {val_loss:.4f}, Validation accuracy = {val_acc:.4f}")
        
        # Finds the best accuracy and saves the model
        if (val_loss < best_loss):
            best_loss = val_loss
            best_accuracy = val_acc
            best_params = copy.deepcopy(cnn.state_dict()) # makes a deepcopy s.t. if cnn model changes, best_params won't
            
    print("---------------------------------------")
    print (f"Best validation accuracy = {best_accuracy*100:.2f}%  |   Best validation loss = {best_loss:.4f}")
        
    # Saves the model
    if args.save: save(cnn, val_loader, loss_fn, device, best_loss, best_params)

    # Plots the training loss vs validation loss
    plt.plot(train_loss_array, label='Training loss')
    plt.plot(val_loss_array, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Validation Loss')
    plt.legend()
    plt.show()

# Call the main method if executed properly
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time = {execution_time:.3f} seconds")
    # plt.show()