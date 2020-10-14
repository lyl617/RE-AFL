import argparse
import torch
import _thread
import os
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.distributed as dist
import matplotlib.pyplot as plt
from fl_utils import printer, time_since

import fl_datasets
import fl_utils
import fl_models
from fl_train_test import train, test
import threading
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--world_size', type=int, default=2, metavar='N',
                        help='number of working devices (default: 2)')
parser.add_argument('--rank', type=int, default=0, metavar='N',
                        help='device index (default: 0)')
parser.add_argument('--addr', type=str, default='192.168.0.100', metavar='N',
                        help='master  ip address')
parser.add_argument('--port', type=str, default='23333',metavar='N',
                        help='port number')
parser.add_argument('--enable_vm_test', action="store_true", default=False)
parser.add_argument('--dataset_type', type=str, default='MNIST',metavar='N',
                        help='dataset type, default: MNIST')
parser.add_argument('--alpha', type=float, default=1.0,metavar='N',
                        help='The value of alpha')
parser.add_argument('--model_type', type=str, default='LR',metavar='N',
                        help='model type, default: Linear Regression')
parser.add_argument('--pattern_idx', type=int, default= 0, metavar='N',
                        help='0: IID, 1: Low-Non-IID, 2: Mid-Non-IID, 3: High-Non-IID')
parser.add_argument('--datanum_idx', type=int, default= 0, metavar='N',
                        help='0: 6000, 1: 4,000-8,000, 2: 1,000-11,000')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epoch_start', type=int, default=0,
                    help='')
parser.add_argument('--log_save_interval', type=int, default=10, metavar='N',
                        help='the interval of log figure')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--local_iters', type=int, default=1, metavar='N',
                        help='input local interval for training (default: 1)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='For Saving the current Model')

args = parser.parse_args()
world_size = args.world_size
rank = args.rank
print("rank is: ", rank)
dist.init_process_group(backend="gloo",
                        init_method="tcp://"+args.addr+":"+args.port,
                        world_size=world_size+1,
                        rank=rank) 
# NUM_RECV_PARAS = 0
test_loss_plot = []
test_acc_plot = []
used_paras = []
recv_queue = []

# exit conditions
exit_loss_threshold = 1.3024 # loss threshold 1.3024  0.8222 0.5108
loss_interval = 10 # the mean of last loss_iterval loss is calculated

# unused_paras = []
lock = threading.Lock()

def split_data(dataset):
    num_samples = len(dataset.data)
    temp_data = []
    temp_target = []
    dataset.data=torch.div(dataset.data/255.0-0.1307, 0.3081)
    for i in range(num_samples):
        temp_data .append( dataset.data[i,:,:])
        temp_target.append(dataset.targets[i])
    return torch.utils.data.TensorDataset(torch.stack(temp_data), torch.stack(temp_target))

def load_test_data():
    global_test_dataset =  datasets.MNIST('./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
    return split_data(global_test_dataset)


def apply_global_para(model, global_para):
    para_dict = model.state_dict()
    keys=list(model.state_dict())
    for i in range(len(keys)):
        para_dict.update({keys[i]: global_para[i]})
    model.load_state_dict(para_dict)
    del para_dict

def aggregate_nomalization(global_para, local_paras, num_aggre, alpha):
    # print("Collected paras from %d devices"%len(local_paras))
    for i in range(len(global_para)):
        len_aggre = 0
        first = True
        for local_para in local_paras:
            # if len_aggre >= num_aggre : break
            if first:
                new_para_i = local_para[i]
            else:
                new_para_i = torch.add(global_para[i], local_para[i])#update the i-th para of global model
            global_para[i] = new_para_i
            len_aggre += 1
            if len_aggre == num_aggre: break
            first = False
        global_para[i] = torch.div(global_para[i], alpha * world_size + 0.0)
    # local_paras.clear()
    return global_para

def remove_alpha_paras(local_paras, alpha):
    num_paras = int(alpha * world_size)
    for i in range(num_paras):
        local_paras.remove(local_paras[i])

def recv_para(local_para, src):
    # print(src,": ", end="")
    for j in range(len(local_para)):
        # print(local_para[j].size(), end=" ")
        temp = local_para[j].to('cpu')
        dist.recv(temp, src=src)
        local_para[j]=temp.to('cuda')
        # del temp
    # if(len(used_paras) < num_paras):
        # lock.acquire()
    global recv_queue
    lock.acquire()
    recv_queue.append(src)
    global used_paras
    used_paras.append(local_para)
    del local_para
    lock.release()
    # else:
    #     unused_paras.append(local_para)
    # local_paras.append(local_para)
    # print(src, end = " ")

def send_para(global_para, epoch_index, dst):
    for j in range(len(global_para)):
        dist.send(global_para[j].to('cpu'), dst=dst)
    dist.send(torch.tensor(epoch_index), dst=dst)

class recv_paras_thread(threading.Thread):
    def __init__(self, src, local_para):
        threading.Thread.__init__(self)
        self.src = src
        self.local_para = local_para
    def run(self):
        recv_para(self.local_para, self.src)

def gen_chunk_sizes(num_clients, datanum_idx):
    if datanum_idx == 0:
        low = 6000
        high = 6000
    else:
        if datanum_idx == 1:
            low = 4000
            high = 8000
        else:
            low = 1000
            high = 11000
    
    raw_sizes = (torch.rand(size=[num_clients])*(high-low)+low).int()

    selected_low_index = (torch.rand([1])*num_clients).long()
    selected_high_index = (torch.rand([1])*num_clients).long()
    while selected_low_index == selected_high_index:
        selected_high_index = (torch.rand([1])*num_clients).long()
    raw_sizes[selected_low_index] = low
    raw_sizes[selected_high_index] = high
    return raw_sizes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    datanum_idx = args.datanum_idx

    tx2_chunk_sizes = gen_chunk_sizes(world_size, datanum_idx)
    for i in range(world_size):
        dist.send(tx2_chunk_sizes[i], dst=i+1)

    #model = Net().to(device)

    is_train = False

    alpha = args.alpha

    log_save_interval = args.log_save_interval

    args.save_model = True

    pattern_list = ['random', 'lowbias', 'midbias', 'highbias']

    datanum_list = ['balance', 'lowimbalance', 'highimbalance']

    checkpoint_dir = 'server_result/server_' + str(args.rank) + '/'
    fl_utils.create_dir(checkpoint_dir)

    fig_dir = checkpoint_dir + 'figure/'
    fl_utils.create_dir(fig_dir)

    FIG_ROOT_PATH = fig_dir + 'alpha_' + str(alpha) + 'model-type_' + args.model_type + '_dataset-type' + args.dataset_type + \
        '_batch-size' + str(args.batch_size) + '_tx2nums' + str(args.world_size) + '_' + pattern_list[args.pattern_idx] + '_data-pattern' + \
            datanum_list[args.datanum_idx] + '_data' + '_exit-loss' + str(exit_loss_threshold) + '_lr' + str(args.lr) + '_epoch' + str(args.epochs) + '_local' + str(args.local_iters) + '_FIGTYPE.png'

    MODEL_PATH = checkpoint_dir + 'model/'
    fl_utils.create_dir(MODEL_PATH)

    LOAD_MODEL_PATH = MODEL_PATH + 'alpha_' + str(alpha) + '_model-type_' + args.model_type + '_dataset-type' + args.dataset_type + \
        '_batch-size' + str(args.batch_size) + '_tx2nums' + str(args.world_size) + '_' + pattern_list[args.pattern_idx] + '_data-pattern' + \
            datanum_list[args.datanum_idx] + '_data' + '_exit-loss' + str(exit_loss_threshold)  + '_lr' + str(args.lr) + '_epoch' + str(args.epochs) + '_local' + str(args.local_iters) + '.pth'

    SAVE_MODEL_PATH = MODEL_PATH + 'alpha_' + str(alpha) + '_model-type_' + args.model_type + '_dataset-type' + args.dataset_type + \
        '_batch-size' + str(args.batch_size) + '_tx2nums' + str(args.world_size) + '_' + pattern_list[args.pattern_idx] + '_data-pattern' + \
            datanum_list[args.datanum_idx] + '_data' + '_exit-loss' + str(exit_loss_threshold) + '_lr' + str(args.lr) + '_epoch' + str(args.epochs) + '_local' + str(args.local_iters) + '.pth'
    

    LOG_ROOT_PATH = checkpoint_dir +  'log/' + 'alpha_' + str(alpha) + '/model-type_' + args.model_type + '_dataset-type' + args.dataset_type + \
        '_batch-size' + str(args.batch_size) + '_tx2nums' + str(args.world_size) + '_' + pattern_list[args.pattern_idx] + '_data-pattern' + \
            datanum_list[args.datanum_idx] + '_data' + '_exit-loss' + str(exit_loss_threshold) + '_lr' + str(args.lr) + '_epoch' + str(args.epochs) + '_local' + str(args.local_iters) +'/' 

    fl_utils.create_dir(LOG_ROOT_PATH)

    LOG_PATH = LOG_ROOT_PATH + 'model_acc_loss.txt'

    log_out = open(LOG_PATH, 'w+')
    # if args.epoch_start == 0:
    #     log_out.write("%s\n" % LOG_PATH)

    # log_out = dict()
    # log_out["model_acc_loss"] = open(os.path.join(LOG_ROOT_PATH, "model_acc_loss.txt"), 'w+')

    # <--Load datasets
    train_dataset, test_dataset = fl_datasets.load_datasets(
        args.dataset_type)

    test_loader = fl_utils.create_server_test_loader(args, kwargs, test_dataset)
    
    #test_dataset = load_test_data()
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # <--Create Neural Network model instance
    if args.dataset_type == 'FashionMNIST':
        if args.model_type == 'LR':
            model = fl_models.MNIST_LR_Net().to(device)
        else:
            model = fl_models.MNIST_Net().to(device)

    elif args.dataset_type == 'MNIST':
        if args.model_type == 'LR':
            model = fl_models.MNIST_LR_Net().to(device)
        else:
            model = fl_models.MNIST_Small_Net().to(device)

    elif args.dataset_type == 'CIFAR10':

        if args.model_type == 'Deep':
            model = fl_models.CIFAR10_Deep_Net().to(device)
            args.decay_rate = 0.98
        else:
            model = fl_models.CIFAR10_Net().to(device)
            args.decay_rate = 0.98

    elif args.dataset_type == 'Sent140':

        if args.model_type == 'LSTM':
            model = fl_models.Sent140_Net().to(device)
            args.decay_rate = 0.99
        else:
            model = fl_models.Sent140_Net().to(device)
            args.decay_rate = 0.99
    else:
        pass

    if not args.epoch_start == 0:
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    global_para = [para[1].data for para in model.named_parameters()]
        # for j in range(len(global_para)):
        #     dist.send(global_para[j].to('cpu'), dst=i)


    for i in range(1, world_size+1):
        _thread.start_new_thread(send_para, (global_para, 0, i))
    

    global recv_queue
    global used_paras
    for i in range(1, world_size+1):
        local_para = [para[1].data for para in model.named_parameters()]
        recv_queue.append(i)
        used_paras.append(local_para)
        del local_para
    num_paras = world_size
    
    start = time.time()
    test(args, start, model, device, test_loader, 0, log_out)
    for epoch in range(1, args.epochs + 1):
        for i in range(num_paras):
            recv_src = recv_queue[0]
            recv_queue.remove(recv_src)
            used_paras.remove(used_paras[0])

            local_para = [para[1].data for para in model.named_parameters()]
            recv_thread = recv_paras_thread(src=recv_src, local_para=local_para)
            recv_thread.start()
            # _thread.start_new_thread(recv_para, (local_para, i, int(alpha * world_size)))
            # for j in range(len(global_para)):
            #     temp = local_para[j].to('cpu')
            #     dist.recv(temp, src=i)
            #     local_para[j]=temp.to('cuda')
            #     del temp
            # .append(local_para)
        while True:
            if len(recv_queue) >= int(alpha * world_size):
            # if len(used_paras) == world_size:
                break
        num_paras = int(alpha * world_size)
        global_para = aggregate_nomalization(global_para, used_paras, num_paras, alpha)
        apply_global_para(model, global_para)

        test_loss, test_acc = test(args, start, model, device, test_loader, epoch, log_out)
        test_loss_plot.append(test_loss)
        test_acc_plot.append(test_acc)
        if epoch % log_save_interval == 0:
            fl_utils.plot_learning_curve(LOG_PATH, FIG_ROOT_PATH)
            if (args.save_model):
                # pass
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
            # for j in range(len(global_para)):
            #     dist.send(global_para[j].to('cpu'), dst=i)
        # # plt.close('all')
        for i in range(num_paras):
            _thread.start_new_thread(send_para, (global_para, epoch, recv_queue[i]))
        
        ## exit when loss is lower than pre-defined threshold
        if len(test_loss_plot) < loss_interval:
            # print("1",np.mean(test_loss_plot))
            # print(type(np.mean(test_loss_plot) ))
            # print(type(np.float64(exit_loss_threshold)))
            if np.mean(test_loss_plot) <= np.float64(exit_loss_threshold):
                # print("exit ok")
                break
        else:
            # print("2",np.mean(test_loss_plot[-loss_interval::]))
            # print(type(np.mean(test_loss_plot[-loss_interval::])))
            if np.mean(test_loss_plot[-loss_interval::]) <= np.float64(exit_loss_threshold):
                # print("exit ok") 
                break
    # print("escape from the for loop")
    time.sleep(5)

    local_para = [para[1].data for para in model.named_parameters()]
    if(epoch != args.epochs):
        for i in range(num_paras):
            for j in range(len(local_para)):
                # print(local_para[j].size(), end=" ")
                temp = local_para[j].to('cpu')
                dist.recv(temp, src=recv_queue[i])
        num_paras = 0
    # print(recv_queue)
    for i in recv_queue:
        if i not in recv_queue[0:num_paras]:
            # print("send stop to: ", i)
            for j in range(len(local_para)):
                dist.send(local_para[j].to('cpu'), dst=i)
            dist.send(torch.tensor(args.epochs), dst=i)


    
    plt.ioff()
    plt.figure()
    plt.plot(test_loss_plot)
    plt.title('Test loss')
    plt.xlabel('epochs')
    plt.ylabel('test loss')
    plt.savefig(fig_dir + './test_loss.png')
    plt.figure()
    plt.plot(test_acc_plot)
    plt.title('Test accuracy')
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.savefig(fig_dir + './test_acc.png')
    plt.ioff()
    # plt.show()
    # if (args.save_model):
    #     torch.save(model.state_dict(), "mnist_cnn.pt")
main()
