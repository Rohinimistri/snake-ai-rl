import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert to numpy arrays first (fix warning too)
        state = np.array(state)
        next_state = np.array(next_state)
        action = np.array(action)
        reward = np.array(reward)

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Handle single input (short memory)
        if len(state.shape) == 1:
          state = torch.unsqueeze(state, 0)
          next_state = torch.unsqueeze(next_state, 0)
          action = torch.unsqueeze(action, 0)
          reward = torch.unsqueeze(reward, 0)
          done = (done,)

          # Predict Q values
          pred = self.model(state)

          # Clone predictions as target
          target = pred.clone().detach()

          # Update Q values
          for idx in range(len(done)):
              Q_new = reward[idx]
              if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

                target[idx][torch.argmax(action[idx]).item()] = Q_new

                # Train
                self.optimizer.zero_grad()
                loss = self.criterion(target, pred)
                loss.backward()
                self.optimizer.step()