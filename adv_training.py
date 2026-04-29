from models import ResNet, ConvNet
import torch.nn as nn
import argparse
from utils import UcrDataset, UCR_dataloader
import torch.optim as optim
import torch.utils.data
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Argument parser to handle command line inputs
parser = argparse.ArgumentParser()

# Add arguments for various configurations
parser.add_argument('--test', action='store_true', help='Run testing mode only')
parser.add_argument('--query_one', action='store_true', help='Query the probability of a specific target index sample')
parser.add_argument('--idx', type=int, help='The index of the test sample to query')
parser.add_argument('--gpu', type=str, default='0', help='GPU index to use')
parser.add_argument('--channel_last', type=bool, default=True, help='Indicates if the channel of data is last')
parser.add_argument('--n_class', type=int, default=2, help='Number of classes in the dataset')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train')
parser.add_argument('--e', default=1499, help='Epochs number for saving the model')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
parser.add_argument('--cuda', action='store_false', help='Disable CUDA if set')
parser.add_argument('--checkpoints_folder', default='model_checkpoints', help='Folder to save model checkpoints')
parser.add_argument('--manualSeed', type=int, help='Manual seed for reproducibility')
parser.add_argument('--run_tag', default='ECG200', help='Tag for the current run (e.g., dataset name)')
parser.add_argument('--model', default='f', help='Model type to use (ResNet or FCN)')
parser.add_argument('--normalize', action='store_true', help='Normalize the data')
parser.add_argument('--checkpoint_every', default=5, help='Save checkpoints after every N epochs')
opt = parser.parse_args()

# Print the parsed options
print(opt)

# Configure CUDA
if torch.cuda.is_available() and not opt.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    print("You have a CUDA device, consider running with --cuda as an option.")

# Set the device (GPU or CPU)
device = torch.device("cuda:0" if opt.cuda else "cpu")

# Set the manual seed if not provided
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


def train(l, e):
    """
    Train the model.

    Args:
        l (list): List to store the loss values.
        e (list): List to store the epoch numbers.
    """
    # Create directory for checkpoints if it does not exist
    os.makedirs(opt.checkpoints_folder, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_folder, opt.run_tag), exist_ok=True)

    # Load training dataset
    dataset_path = 'data/' + opt.run_tag + '/' + opt.run_tag + '_TRAIN.txt'
    dataset = UcrDataset(dataset_path, channel_last=opt.channel_last, normalize=True)
    batch_size = int(min(len(dataset) / 10, 16))

    print('Dataset length: ', len(dataset))
    print('Batch size:', batch_size)
    dataloader = UCR_dataloader(dataset, batch_size)

    seq_len = dataset.get_seq_len()
    n_class = opt.n_class
    print('Sequence length:', seq_len)

    # Initialize the model
    if opt.model == 'r':
        net = ResNet(n_in=seq_len, n_classes=n_class).to(device)
    elif opt.model == 'f':
        net = ConvNet(n_in=seq_len, n_classes=n_class).to(device)

    net.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    print('############# Start Training ###############')
    for epoch in range(opt.epochs):
        for i, (data, label) in enumerate(dataloader):
            if data.size(0) != batch_size:
                break
            data = data.float().to(device)
            label = label.long().to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, label.view(label.size(0)))
            loss.backward()
            optimizer.step()

            print('Training Progress:')
            print('[%d/%d][%d/%d] Loss: %.4f ' % (epoch, opt.epochs, i + 1, len(dataloader), loss.item()))

        # Save loss and epoch info every 10 epochs
        if (epoch % 10 == 0):
            l.append(loss.item())
            e.append(epoch)

    print('Saving the model.....')
    torch.save(net, f'model_checkpoints/{opt.run_tag}/pre_{opt.model}Trained.pth')


def te():
    """
    Test the model.
    """
    data_path = 'data/' + opt.run_tag + '/' + opt.run_tag + '_TEST.txt'
    dataset = UcrDataset(data_path, channel_last=opt.channel_last, normalize=opt.normalize)
    batch_size = int(min(len(dataset) / 10, 16))
    print('Dataset length: ', len(dataset))
    print('Batch size:', batch_size)
    dataloader = UCR_dataloader(dataset, batch_size)

    model_path = f'model_checkpoints/{opt.run_tag}/pre_{opt.model}Trained.pth'
    model = torch.load(model_path, map_location='cuda:0')

    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0

        for i, (data, label) in enumerate(dataloader):
            data = data.float().to(device)
            label = label.long().to(device).view(label.size(0))
            total += label.size(0)
            out = model(data).cuda()
            softmax = nn.Softmax(dim=-1)
            prob = softmax(out)
            pred_label = torch.argmax(prob, dim=1)

            correct += (pred_label == label).sum().item()

        print('The TEST Accuracy of %s is : %.2f %%' % (data_path, correct / total * 100))


def query_one(idx):
    """
    Query the probability of the true class for a specific test sample.

    Args:
        idx (int): Index of the test sample.
    """
    data_path = 'data/' + opt.run_tag + '/' + opt.run_tag + '_TEST.txt'
    test_data = np.loadtxt(data_path)
    test_data = torch.from_numpy(test_data)

    test_one = test_data[idx]

    X = test_one[1:].float().to(device)
    y = test_one[0].long() - 1
    y = y.to(device)
    if y < 0:
        y = opt.n_class - 1
    print('Ground truth:', y)

    model_path = 'model_checkpoints/' + opt.run_tag + '/pre_' + opt.model + str(opt.e) + 'epoch.pth'
    model = torch.load(model_path, map_location='cpu')
    model.eval()

    out = model(X)
    softmax = nn.Softmax(dim=-1)
    prob_vector = softmax(out)
    print('Probability vector:', prob_vector)
    prob = prob_vector.view(opt.n_class)[y].item()

    print('Confidence in true class of sample %d is %.4f ' % (idx, prob))


def plot1(model, loss, epoch):
    """
    Plot the training loss over epochs.

    Args:
        model (str): Model type (ResNet or FCN).
        loss (list): List of loss values.
        epoch (list): List of epoch numbers.
    """
    plt.title('Training Loss of ' + opt.run_tag + ' Model', fontstyle='italic')
    plt.figure(figsize=(6, 4))
    plt.plot(epoch, loss, color='b', label='Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(loc='upper right', fontsize=8)
    plt.show()


if __name__ == '__main__':
    l = []
    e = []
    train(l, e)
    plot1(opt.model, l, e)
    te()
