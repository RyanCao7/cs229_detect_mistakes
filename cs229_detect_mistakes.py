import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, Dataset

import torchvision
from torchvision import transforms, utils

import librosa
import librosa.display

import os
import numpy as np
import warnings
import json
import math
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from networks.py import Inception3, ResNet, Bottleneck

DATASET_PATH_LOCAL = '/Volumes/Ryan_Cao/cs229_dataset/added/07-12-2018_at_01:55:02_with_50_examples'
DATASET_PATH_REMOTE = 'cs229_dataset/added/07-12-2018_at_10:06:18_with_10000_examples'

class NetThree(nn.Module):
    def __init__(self):
        super(NetThree, self).__init__()
        # Conv layer: (input, output, 3x8 rectangular conv)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 10, (4, 8)), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(10, 24, (8, 4)), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(nn.Conv2d(24, 48, (4, 8)), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(48, 96, (8, 4)), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(96, 120, (6, 6)), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(120, 96, (6, 6)), nn.ReLU())
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(96, 48, (8, 4)), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(48, 24, (4, 8)), nn.ReLU())
        # FC layer: (input, output)
        self.rnn = nn.LSTM(feature_dim, 360, 1, batch_first=True)
        self.fc2 = nn.Linear(720, 360)

    def forward(self, x):
        # print('input:', x.shape)
        # Conv, then ReLU, then max pool (2x2 pool)
        x = self.conv1(x)
        # print('after first conv:', x.shape)
        # You can also specify a single number for a square pool
        x = self.conv2(x)
        # print('after second conv:', x.shape)
        x = self.pool(x)
        # print('after first pool:', x.shape)
        # Flatten feature maps for the linear layer. -1 param says "compute this for me pls"
        x = self.conv3(x)
        # print('after third conv:', x.shape)
        # You can also specify a single number for a square pool
        x = self.conv4(x)
        # print('after fourth conv:', x.shape)
        x = self.pool(x)
        # print('after second pool:', x.shape)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        # Pass through fc layers
        x = F.relu(self.fc1(x))
        # print('after first fc/relu:', x.shape)
        x = F.sigmoid((self.fc2(x)))
        # print('after second fc/relu:', x.shape)
        # NO ACTIVATION FUNCTION?!?!?!?
        # print('final:', x.shape)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # All dims minus the batch dim
        num_features = 1
        for dim in size:
            num_features *= dim
        return num_features


class NetTwo(nn.Module):
    def __init__(self):
        super(NetTwo, self).__init__()
        # Conv layer: (input, output, 3x8 rectangular conv)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 10, (4, 8)), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(10, 24, (8, 4)), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(nn.Conv2d(24, 48, (4, 8)), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(48, 24, (8, 4)), nn.ReLU())
        # FC layer: (input, output)
        self.fc1 = nn.Linear(110208, 720)
        self.fc2 = nn.Linear(720, 360)

    def forward(self, x):
        # print('input:', x.shape)
        # Conv, then ReLU, then max pool (2x2 pool)
        x = self.conv1(x)
        # print('after first conv:', x.shape)
        # You can also specify a single number for a square pool
        x = self.conv2(x)
        # print('after second conv:', x.shape)
        x = self.pool(x)
        # print('after first pool:', x.shape)
        # Flatten feature maps for the linear layer. -1 param says "compute this for me pls"
        x = self.conv3(x)
        # print('after third conv:', x.shape)
        # You can also specify a single number for a square pool
        x = self.conv4(x)
        # print('after fourth conv:', x.shape)
        x = self.pool(x)
        # print('after second pool:', x.shape)
        x = x.view(-1, self.num_flat_features(x))
        # Pass through fc layers
        x = F.relu(self.fc1(x))
        # print('after first fc/relu:', x.shape)
        x = F.sigmoid((self.fc2(x)))
        # print('after second fc/relu:', x.shape)
        # NO ACTIVATION FUNCTION?!?!?!?
        # print('final:', x.shape)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # All dims minus the batch dim
        num_features = 1
        for dim in size:
            num_features *= dim
        return num_features

class NetOne(nn.Module):
    def __init__(self):
        super(NetOne, self).__init__()
        # Conv layer: (input, output, 2x10 rectangular conv)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 10, (2, 10)), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(10, 10, (10, 2)), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(nn.Conv2d(10, 24, (2, 10)), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(24, 10, (10, 2)), nn.ReLU())
        # FC layer: (input, output)
        # Note that conv layer gives 16 feature maps of size 5x5
        self.fc1 = nn.Linear(10 * 82 * 56, 720)
        self.fc2 = nn.Linear(720, 360)

    def forward(self, x):
        # print('input:', x.shape)
        # Conv, then ReLU, then max pool (2x2 pool)
        x = self.conv1(x)
        # print('after first conv:', x.shape)
        # You can also specify a single number for a square pool
        x = self.conv2(x)
        # print('after second conv:', x.shape)
        x = self.pool(x)
        # print('after first pool:', x.shape)
        # Flatten feature maps for the linear layer. -1 param says "compute this for me pls"
        x = self.conv3(x)
        # print('after third conv:', x.shape)
        # You can also specify a single number for a square pool
        x = self.conv4(x)
        # print('after fourth conv:', x.shape)
        x = self.pool(x)
        # print('after second pool:', x.shape)
        x = x.view(-1, self.num_flat_features(x))
        # Pass through fc layers
        x = F.relu(self.fc1(x))
        # print('after first fc/relu:', x.shape)
        x = F.sigmoid((self.fc2(x)))
        # print('after second fc/relu:', x.shape)
        # NO ACTIVATION FUNCTION?!?!?!?
        # print('final:', x.shape)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # All dims minus the batch dim
        num_features = 1
        for dim in size:
            num_features *= dim
        return num_features

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv layer: (input, output, 6x6 square conv)
        self.conv1 = nn.Conv2d(1, 10, 6, stride=(3, 2))
        self.conv2 = nn.Conv2d(10, 4, 5)
        self.pool = nn.MaxPool2d(3, 3)
        # FC layer: (input, output)
        # Note that conv layer gives 16 feature maps of size 5x5
        self.fc1 = nn.Linear(4 * 38 * 40, 720)
        self.fc2 = nn.Linear(720, 360)

    def forward(self, x):
        # print('input:', x.shape)
        # Conv, then ReLU, then max pool (2x2 pool)
        x = self.conv1(x)
        # print('after first conv:', x.shape)
        x = F.relu(x)
        # print('after first relu:', x.shape)
        # You can also specify a single number for a square pool
        x = self.pool(F.relu(self.conv2(x)))
        # print('after first pool:', x.shape)
        # Flatten feature maps for the linear layer. -1 param says "compute this for me pls"
        x = x.view(-1, self.num_flat_features(x))
        # Pass through fc layers
        x = F.relu(self.fc1(x))
        # print('after first fc/relu:', x.shape)
        x = F.sigmoid((self.fc2(x)))
        # print('after second fc/relu:', x.shape)
        # NO ACTIVATION FUNCTION?!?!?!?
        # print('final:', x.shape)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # All dims minus the batch dim
        num_features = 1
        for dim in size:
            num_features *= dim
        return num_features

# Normalizes for a single spectrogram across time
class NormalizeTransform(object):
    def __init__(self):
        pass

    # PRE: spectrogram is a 2D np array
    def __call__(self, spectrogram, label):
        means = np.mean(spectrogram, axis=1)
        stds = np.std(spectrogram, axis=1)
        for i in range(len(spectrogram)):
            spectrogram[i] = (spectrogram[i] - means[i]) / stds[i]

        return spectrogram, label

# Hard crops the spectrogram to be 256 x 360
class ThreeSixtyCrop(object):

    def __init__(self):
        self.output_size = (256, 360)

    # Resizes spectrogram to be 256 x 360, and label to be 1x360
    def __call__(self, spectrogram, label):
        spectrogram = spectrogram[:, :360]
        label = label[:360]
        assert(spectrogram.shape == (256, 360))
        return spectrogram, label

# Literally just a wrapper so we can use DataLoader
class DummySpectrogramDataset(Dataset):
    def __init__(self, examples, labels):
        self.examples = examples
        self.labels = labels

    # Grabs the length of the entire dataset
    def __len__(self):
        assert(len(self.labels) == len(self.examples))
        return len(self.labels)

    # Grabs the idx'th item in tuple form (spectrogram, labels)
    def __getitem__(self, idx):
        return (self.examples[idx], self.labels[idx])

def count_gt_proportion(labels):
    total = 1.
    for dim in labels.shape:
        total *= dim
    return np.sum(labels) / total

def load_dataset(data_path, batch_size=100, num_threads=2):

    try:
        dataset = np.load(data_path + '/dataset.npy', encoding="bytes")
        labels = np.load(data_path + '/labels.npy', encoding="bytes")
    except IOError:

        # Crops to size 256 x 360 first, then normalizes across the time axis
        crop = ThreeSixtyCrop()
        normalize = NormalizeTransform()

        dataset = []
        labels = []
        tqdm.write('Loading dataset...')
        for folder in tqdm(natsorted(glob(data_path + "*/*/*/"))):
            try:
                spectrogram = np.load(glob(folder + "*mel*.npy")[0])
                label = np.load(glob(folder + "*dense*.npy")[0])
            except IndexError:
                continue

            assert(label.shape[-1] == spectrogram.shape[-1])
            assert(label.shape[-1] >= 360)
            # Crop each spectrogram
            spectrogram, label = crop(spectrogram, label)
            spectrogram, label = normalize(spectrogram, label)
            dataset.append(np.transpose(spectrogram))
            labels.append(label)
            # tqdm.write(str(spectrogram.shape))
            # tqdm.write(str(label.shape))

        np.save(data_path + '/dataset.npy', dataset)
        np.save(data_path + '/labels.npy', labels)

    num_data = np.shape(labels)[0]
    tqdm.write('num_data: ' + str(num_data))

    # Convert to torch tensors
    dataset = np.asarray(dataset)
    labels = np.asarray(labels)

    # Takes a random set of the dataset as training data
    all_indices = np.random.permutation(np.arange(num_data))
    train_indices = all_indices[:int(0.8 * num_data)]
    validation_indices = all_indices[int(0.8 * num_data):]

    # Picks out the corresponding examples and labels for training/validation sets.
    training_examples = dataset[train_indices]
    training_labels = labels[train_indices]
    validation_examples = dataset[validation_indices]
    validation_labels = labels[validation_indices]

    for i, example in enumerate(training_examples):
        training_examples[i] = torch.FloatTensor(training_examples[i])
        training_labels[i] = torch.FloatTensor(training_labels[i])

    for i, example in enumerate(validation_examples):
        training_examples[i] = torch.FloatTensor(validation_examples[i])
        training_labels[i] = torch.FloatTensor(training_labels[i])

    # Creates data loaders for training and validation datasets
    train_loader = DataLoader(DummySpectrogramDataset(training_examples, training_labels), batch_size = batch_size, shuffle = True, num_workers = num_threads)
    validation_loader = DataLoader(DummySpectrogramDataset(validation_examples, validation_labels), batch_size = batch_size, shuffle = False, num_workers = num_threads)

    return train_loader, validation_loader

def get_dataset_path():
    path = input('What is your dataset path? ')
    while not os.path.exists(path):
        path = input('What is your dataset path? ')
    return path

# Plots a single mel spectrogram to store_path, given the np.ndarray with the
# spectrogram data and the labels.
def plot_mel_spectrogram(store_path, spectrogram, record_len, labels, fmax):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', fmax=fmax, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    for i, label in enumerate(labels):
        if label > 0:
            plt.axvline(x = record_len * i / len(labels), color = 'g', linewidth = 3, alpha = 0.4)
    plt.savefig(store_path)
    plt.close()

def plot_all(path):
    for folder in tqdm(glob(path + "*/*/*/")):
        try:
            # tqdm.write("folder: " + folder)
            # Loads the recording length, spectrogram, and labels.
            recording_len_file = open(folder + 'duration.txt')
            recording_len = float(recording_len_file.readlines()[1].rstrip('\n'))
            recording_len_file.close()
            crop = ThreeSixtyCrop()
            spectrogram = np.load(glob(folder + "*mel*.npy")[0])
            labels = np.load(glob(folder + "*dense*.npy")[0])
            predictions = np.load(glob(folder + '*pred*.npy')[0])
            spectrogram, labels = crop(spectrogram, labels)
            plot_mel_spectrogram(folder + '_mel_spectrogram_gt' + '.png', spectrogram, recording_len, labels, fmax=4096)
            plot_mel_spectrogram(folder + '_mel_spectrogram_pred' + '.png', spectrogram, recording_len, predictions, fmax=4096)
        except IndexError:
            tqdm.write('error')
            pass

# Now deprecated.
def visualize_predictions(folder_path, spectrogram, predictions, ground_truth):
    duration_file = open(folder_path + 'duration.txt')
    duration = float(duration_file.readlines()[0].rstrip('\n'))
    duration_file.close()
    # spectrogram = np.load(glob(folder_path + "*mel*.npy")[0])
    # labels = np.load(glob(folder_path + "*dense*.npy")[0])
    if ground_truth:
        plot_mel_spectrogram(folder_path + '_mel_spectrogram_gt' + '.png', spectrogram, duration, predictions, fmax=4096)
    else:
        plot_mel_spectrogram(folder_path + '_mel_spectrogram_pred' + '.png', spectrogram, duration, predictions, fmax=4096)

# Gets predictions based on sigmoid function outputs
def get_pred(outputs):
    preds = torch.zeros(outputs.shape)
    preds[outputs >= 0.5] = 1
    return preds

# We have to load up the optimizer too???!!
# TODO: Make it so we can load up from any epoch, and not just the last one.
def loadup(master_path):
    # Loads loss data
    loss_file = open(master_path + 'loss_data', 'r')
    loss_data = json.load(loss_file)

    # Loads saved network weights
    if loss_data['net_type'] == 0:
        net = Net()
    elif loss_data['net_type'] == 1:
        net = NetOne()
    elif loss_data['net_type'] == 2:
        net = NetTwo()
    elif loss_data['net_type'] == 3:
        net = NetThree()
    elif loss_data['net_type'] == 4:
        net = ResNet(Bottleneck, [3, 4, 23, 3])
    elif loss_data['net_type'] == 5:
        net = Inception3()
    weight_files = natsorted(glob(master_path + 'weights/epoch*'))
    net.load_state_dict(torch.load(weight_files[-1],  map_location='cpu'))

    # Loads epoch data
    indicator_file = open(master_path + 'indicator.txt', 'r')
    epoch_data = indicator_file.readlines()
    epoch_number = int(epoch_data[0].strip('epoch: ').strip('\n'))
    total_epochs = int(epoch_data[1].strip('total epochs: ').strip('\n'))

    return net, loss_data, epoch_number, total_epochs

# Helper plot functions
def plot_something(save_path, list_one, list_two):
    plt.plot(list_two, list_one)
    plt.savefig(save_path)
    plt.close()

def plot_two_things(save_path, list_one, list_one_times, list_two, list_two_times, list_one_title, list_two_title):
    plt.plot(list_one_times, list_one, '-r', label=list_one_title)
    plt.plot(list_two_times, list_two, '-b', label=list_two_title)
    plt.legend(loc='upper left')
    plt.savefig(save_path)
    plt.close()

def plot_everything(master_path, loss_data, net):
    # for param in net.parameters():
    #     print(np.linalg.norm(param.data), param.size())
    #     plt.plot()
    # exit()
    plot_two_things(master_path + 'losses.png', loss_data['training_losses'], np.arange(1, len(loss_data['training_losses']) + 1), loss_data['validation_losses'], np.arange(1, len(loss_data['validation_losses']) + 1), 'training_loss', 'validation_loss')
    plot_two_things(master_path + 'accuracies.png', loss_data['training_accuracies'], np.arange(1, len(loss_data['training_accuracies']) + 1), loss_data['validation_accuracies'], np.arange(1, len(loss_data['validation_accuracies']) + 1), 'training_accuracy', 'validation_accuracy')

# Make sure this saves an 'indicator.txt', which will tell us if we've started training or not.
def save_everything(master_path, net, loss_data, epoch, total_epochs):
    # Saves loss data
    loss_data_file = open(master_path + 'loss_data', 'w')
    loss_data_file.write(json.dumps(loss_data, indent=4))
    loss_data_file.close()

    if not os.path.exists(master_path + 'weights'):
        os.makedirs(master_path + 'weights')

    # Saves network weights learned
    torch.save(net.state_dict(), master_path + 'weights/epoch' + str(epoch))

    # Save epoch data
    indicator_file = open(master_path + 'indicator.txt', 'w')
    indicator_file.write('epoch: ' + str(epoch) + '\n')
    indicator_file.write('total epochs: ' + str(total_epochs) + '\n')
    indicator_file.close()

def validate_network(net, validation_loader, epoch):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    total_loss = 0
    num_items = 0
    correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(validation_loader)):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            inputs = torch.unsqueeze(inputs, 1)

            # Pushes to cuda (GPU) if available
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + backward + optimize
            outputs = net(inputs)
            predictions = get_pred(outputs)
            loss = F.binary_cross_entropy(outputs, labels)

            # Converts back to cpu to compute stats
            outputs = outputs.cpu()
            loss = loss.cpu()
            labels = labels.cpu()
            predictions = predictions.cpu()

            # Adds up total losses and correctness
            total_loss += loss.sum().item()
            num_items += labels.shape[0] * labels.shape[1]
            correct += (predictions == labels).sum().item()

        avg_loss = total_loss / num_items
        accuracy = correct / num_items
        tqdm.write('VALIDATION: Epoch %4d | Accuracy: %.5f | Loss: %.5f' % (epoch, accuracy, avg_loss))
        return accuracy, avg_loss

def get_net_type():
    net_type = -1
    while(net_type <= 0 or net_type > 5):
        net_type = int(input('Which type of net do we want? (0 - 5 for now) -> '))
    return net_type

# TODO: Also make this actually read in from a dataset LOL
# TODO: Make us able to visualize the predictions of the neural net
def train_network(run_id, local=False, batch_size=100, num_epochs=40, learning_rate=0.001):

    master_path = None
    if local:
        master_path = '/Volumes/Ryan_Cao/mistake_detect_all_runs/' + run_id + '/'
    else:
        master_path = 'mistake_detect_all_runs/' + run_id + '/'
    if not os.path.exists(master_path):
        os.makedirs(master_path)

    # Loads up net from previously trained state, if it exists.
    if os.path.isfile(master_path + 'indicator.txt'):
        net, loss_data, epoch_number, total_epochs = loadup(master_path)
    else:
        net_type = get_net_type()
        if net_type == 0:
            net = Net()
        elif net_type == 1:
            net = NetOne()
        elif net_type == 2:
            net = NetTwo()
        elif net_type == 3:
            net = NetThree()
        elif net_type == 4:
            net = ResNet(Bottleneck, [3, 4, 23, 3])
        elif net_type == 5:
            net = Inception3()
        loss_data = {
            'training_losses': [],
            'training_accuracies': [],
            'validation_losses': [],
            'validation_accuracies': [],
            'weight_summaries': [],
            'net_type': net_type
        }
        epoch_number = 1
        total_epochs = num_epochs
        loss_data['dataset_path'] = get_dataset_path()

    # Loads up dataset and net from saved state, if any exists. Also uses GPU if possible.
    train_loader, validation_loader = load_dataset(loss_data['dataset_path'], batch_size = batch_size, num_threads = 2)

    # Puts network on GPU when possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # Let's actually train our nn!
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)

    for epoch in tqdm(range(epoch_number, total_epochs + 1)):
        total_loss = 0
        num_items = 0
        correct = 0

        batch_loss = 0
        batch_num_items = 0
        batch_correct = 0
        for i, data in tqdm(enumerate(train_loader)):
            # Get the inputs
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()
            inputs = torch.unsqueeze(inputs, 1)

            # Pushes to cuda (GPU) if available
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradient buffers
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            predictions = get_pred(outputs)
            loss = F.binary_cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            # Adds up total losses and correctness for epoch
            outputs = outputs.cpu()
            loss = loss.cpu()
            labels = labels.cpu()
            predictions = predictions.cpu()
            total_loss += loss.sum().item()
            num_items += labels.shape[0] * labels.shape[1]
            correct += (predictions == labels).sum().item()

            # For batch statistics (just to be printed for now)
            batch_loss += loss.sum().item()
            batch_num_items += labels.shape[0] * labels.shape[1]
            batch_correct += (predictions == labels).sum().item()

            if i % (batch_size / 5) == (batch_size / 5) - 1:
                tqdm.write('TRAINING: Epoch %4d | Batch: %2d | Accuracy: %.5f | Loss: %.5f' % (epoch, i, batch_correct / batch_num_items, batch_loss / batch_num_items))
                batch_loss = 0
                batch_num_items = 0
                batch_correct = 0

        avg_loss = total_loss / num_items
        accuracy = correct / num_items
        tqdm.write('---------------------------------------------------')
        tqdm.write('TRAINING: Epoch %4d | Accuracy: %.5f | Loss: %.5f' % (epoch, accuracy, avg_loss))
        validation_acc, validation_loss = validate_network(net, validation_loader, epoch)
        tqdm.write('---------------------------------------------------')
        loss_data['training_losses'].append(avg_loss)
        loss_data['training_accuracies'].append(accuracy)
        loss_data['validation_losses'].append(validation_loss)
        loss_data['validation_accuracies'].append(validation_acc)
        plot_everything(master_path, loss_data, net)
        save_everything(master_path, net, loss_data, epoch, total_epochs)

    print('finished training')

def test_network(run_id, test_folder_path, local=False):

    master_path = None
    if local:
        master_path = '/Volumes/Ryan_Cao/mistake_detect_all_runs/' + run_id + '/'
    else:
        master_path = 'mistake_detect_all_runs/' + run_id + '/'
    # Loads up net from previously trained state, if it exists.
    if os.path.isfile(master_path + 'indicator.txt'):
        net, loss_data, epoch_number, total_epochs = loadup(master_path)
    else:
        print('Sorry, that run id isn\'t recognized.')
        return

    crop = ThreeSixtyCrop()
    normalize = NormalizeTransform()

    with torch.no_grad():
        for folder in tqdm(natsorted(glob(test_folder_path + "*/*/*/"))):
            # tqdm.write(folder)
            try:
                spectrogram = np.load(glob(folder + "*mel*.npy")[0])
                label = np.load(glob(folder + "*dense*.npy")[0])
            except IndexError:
                continue

            assert(label.shape[-1] == spectrogram.shape[-1])
            assert(label.shape[-1] >= 360)
            # Crop each spectrogram
            spectrogram, label = crop(spectrogram, label)
            # spectrogram, label = normalize(spectrogram, label)
            input = torch.FloatTensor(spectrogram)
            # Predict on the spectrogram
            input = np.transpose(input)
            input = torch.unsqueeze(input, 0) # For input channels
            input = torch.unsqueeze(input, 0) # For batch size
            output = net(input)
            predictions = get_pred(output)
            predictions = torch.squeeze(predictions).numpy()
            if not os.path.exists(folder + run_id):
                os.makedirs(folder + run_id)
            np.save(folder + run_id + '/predictions.npy', predictions)

        print('Done predicting!')
        # plot_all(test_folder_path)

warnings.filterwarnings('ignore')
plt.ion() # I don't think this helps me... :(

def get_batch_size():
    batch_sz = -1
    while batch_sz <= 0:
        batch_sz = int(input('What is your batch size? '))
    return batch_sz

def get_num_epochs():
    num_epochs = -1
    while num_epochs <= 0:
        num_epochs = int(input('How many epochs to train? '))
    return num_epochs

def get_lr():
    lr = -1
    while lr <= 0:
        lr = float(input('Learning rate? '))
    return lr

mode = int(input('Which mode would you like? (0 for train; 1 for test) -> '))
id = input('Which run ID would you like? ')
if mode == 0:
    batch_size = get_batch_size()
    num_epochs = get_num_epochs()
    lr = get_lr()
    train_network(id, batch_size = batch_size, num_epochs = num_epochs, learning_rate = lr)
elif mode == 1:
    test_folder = get_dataset_path()
    test_network(id, test_folder)
else:
    print('Please provide a valid mode!')
