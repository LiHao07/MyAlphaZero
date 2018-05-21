import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

N=8

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*N*N,N*N+1)
        
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*N*N, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*N*N)
        x_act = F.log_softmax(self.act_fc1(x_act),dim=1)

        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*N*N)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    def __init__(self , model_file=None, use_gpu=False):#输入：是否从已有model中载入，是否使用gpu
        self.use_gpu = use_gpu
        self.l2_const = 1e-4  # coef of l2 penalty
        
        if self.use_gpu:
            self.policy_value_net = Net().cuda()
        else:
            self.policy_value_net = Net()
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),weight_decay=self.l2_const)

        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file))

    def policy_value(self, state_batch):#得到所有位置下棋的概率
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, Board):#得到所有可行位置下棋的概率
        #Board.graphic()
        legal_positions = Board.availables
        #print(legal_positions)
        current_state = np.ascontiguousarray(Board.board.reshape(-1, 1, N, N))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(Variable(torch
                .from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        #输入：棋盘，策略，该方是否胜利，lr 
        #输出：loss & policy entropy
        #print(mcts_probs)
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)
        log_act_probs, value = self.policy_value_net(state_batch)
        
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.data[0], entropy.data[0]

    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)