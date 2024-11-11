# Common Libnary Files 
# Reference by "Deep Learning with PyTorch" by M. Akaishi

#ReadMe
README = 'Common LIbrary for PyTorch'

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from tqdm.notebook import tqdm

#Calculate Loss 
def eval_loss(loader, device, net, criterion):

    #Get the first set from the DataLoader
    for images, labels in loader:
        break

    #Allocate GPU
    inputs = images.to(device)
    labels = labels.to(device)

    #Calculate prediction
    outputs = net(inputs)

    #Calculate loss
    loss = criterion(outputs, labels)

    return loss

#Main training loop for adjusting parameters
def my_training_loop(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):

    base_epochs = len(history)

    for epoch in range(base_epochs, num_epochs+base_epochs):
        #Numer of correct results in 1 epoch for accuracy
        n_train_acc, n_val_acc = 0, 0
        #Total loss for 1 epoch(Before Average)
        train_loss, val_loss = 0, 0
        # Total data in 1 epoch
        n_train, n_test = 0, 0

        # Training phase
        net.train()

        for inputs, labels in tqdm(train_loader):
            # Total number of data in 1 batch
            train_batch_size = len(labels)
            # Total data in 1 epoch
            n_train += train_batch_size

            # Send data to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Initialize gradient
            optimizer.zero_grad()

            # Calculate prediction
            outputs = net(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Calculate gradient
            loss.backward()

            # Adjust parameter
            optimizer.step()

            # Get predicted labels
            predicted = torch.max(outputs, 1)[1]

            # Calculte loss before average and number of correct answer
            # since loss is calculated for average, make it before average and add
            train_loss += loss.item() * train_batch_size
            n_train_acc += (predicted == labels).sum().item()

        # Prediction phase
        net.eval()
        
        for inputs_test, labels_test in test_loader:
            # Number of data in 1 batch
            test_batch_size = len(labels_test)
            # Acculated number of data in 1 epoch
            n_test += test_batch_size
            
            # Send data to GPU 
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            # Calculate predictin
            outputs_test = net(inputs_test)

            # Calculate loss
            loss_test = criterion(outputs_test, labels_test)

            # Get predicted labels
            predicted_test = torch.max(outputs_test, 1)[1]

            # Calculte loss before averaging and cont correct answer
            # since loss is calculated for average, make it before average and add
            val_loss += loss_test.item() * test_batch_size
            n_val_acc += (predicted_test == labels_test).sum().item()
        
        # Calculate accuracy
        train_acc = n_train_acc / n_train if n_train > 0 else 0
        val_acc = n_val_acc / n_test if n_test > 0 else 0
        # Calculate loss 
        avg_train_loss = train_loss / n_train if n_train > 0 else 0
        avg_val_loss = val_loss / n_test if n_test > 0 else 0
        
        # Print result
        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {val_acc:.5f}')        # Record
        
        #Record history
        item = np.array([epoch+1, avg_train_loss, train_acc, avg_val_loss, val_acc])
        history = np.vstack((history, item))
    
    return history



# Learning curve chart for loss and accuracy
def evaluate_history(history):
    # Display initial and final loss and accuracy
    print(f'Initial state: Loss: {history[0,3]:.5f} Accuracy: {history[0, 4]:.5f}')
    print(f'Final state: Loss: {history[-1,3]: .5f} Accuracy: {history[-1,4]:.5f}')

    num_epochs = len(history)
    if num_epochs < 10:
        unit = 1
    else:
        unit = num_epochs / 10
    
    # Loss chart
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='Training')
    plt.plot(history[:,0], history[:,3],  color='orange', label='Validate')
    plt.xticks(np.arange(0, num_epochs+1, unit))
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.title('Learning Curve for Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Accuracy chart
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='Training')
    plt.plot(history[:,0], history[:,4],  color='orange', label='Validate')
    plt.xticks(np.arange(0, num_epochs+1, unit))
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve for Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# Display image and lable with trained model
def show_images_labels(loader, classes, net, device):

    # Get the first set from the DataLoader
    for images, labels in loader:
        break
    # Decide number of display, number of batch or 50.
    n_size = min(len(images), 50)

    if net is not None:
        #Allocate GPU
        inputs = images.to(device)
        labels = labels.to(device)

        # Calculate loss
        outputs = net(inputs)
        predicted = torch.max(outputs, 1)[1]

    # Display the fist n_size
    plt.figure(figsize=(20, 15)) 
    for i in range(n_size):
        ax = plt.subplot(5, 10, i+1)
        label_name = classes[labels[i]]
        # Incase net != Noe, display result on the title
        if net is not None:
            predicted_name = classes[predicted[i]]
            # Change color if it gets false answer
            if label_name == predicted_name:
                c = 'k'# Correct prediction, black color
            else:
                c = 'r'# Incorrect prediction, red color
            ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        #If net == none, display only correct label
        else:
            ax.set_title(label_name, fontsize=20)
        # Convert Tensor to NumPy
        # image_np = images[i].numpy().copy()
        image_np = images[i].cpu().detach().numpy().copy()
        # Change index as follow
        # (channel, row, column) -> (row, column, channel)
        img = np.transpose(image_np, (1, 2, 0))
        # Set range [-1, 1] -> [0, 1]
        img = (img+1)/2
        # Display result
        plt.imshow(img)
        ax.set_axis_off()
    plt.show()


#PorTorch fixed random seed
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True




            



