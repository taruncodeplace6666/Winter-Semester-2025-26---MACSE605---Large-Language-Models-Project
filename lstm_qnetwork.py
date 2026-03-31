# src/lstm_qnetwork.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImprovedLSTMQNetwork(nn.Module):
    def __init__(self, state_size=8, action_size=2, hidden_size=64, lstm_layers=2, seq_len=10):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.seq_len = seq_len
        
        self.lstm = nn.LSTM(
            input_size=state_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2 if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            batch_first=True,
            dropout=0.2
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size // 2)
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        
        self.dropout = nn.Dropout(0.2)
        
        self.value_fc = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.value = nn.Linear(hidden_size // 4, 1)
        
        self.advantage_fc = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.advantage = nn.Linear(hidden_size // 4, action_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=np.sqrt(2))
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = self.layer_norm1(lstm_out)
        
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out)
        
        pooled = torch.mean(attn_out, dim=1)
        
        fc1_out = F.relu(self.layer_norm2(self.fc1(pooled)))
        fc1_out = self.dropout(fc1_out)
        
        fc2_out = F.relu(self.layer_norm3(self.fc2(fc1_out)))
        fc2_out = self.dropout(fc2_out)
        
        value = F.relu(self.value_fc(fc2_out))
        value = self.value(value)
        
        advantage = F.relu(self.advantage_fc(fc2_out))
        advantage = self.advantage(advantage)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values, hidden
    
    def get_q_values(self, x):
        q_values, _ = self.forward(x)
        return q_values


class LSTMSequenceBuffer:
    def __init__(self, seq_len=10, state_size=8):
        self.seq_len = seq_len
        self.state_size = state_size
        self.buffer = []
        self.hidden = None
    
    def add(self, state):
        self.buffer.append(state)
        if len(self.buffer) > self.seq_len:
            self.buffer = self.buffer[-self.seq_len:]
    
    def get_sequence(self):
        if len(self.buffer) == 0:
            return np.zeros((self.seq_len, self.state_size))
        
        if len(self.buffer) < self.seq_len:
            padding = [np.zeros(self.state_size) for _ in range(self.seq_len - len(self.buffer))]
            seq = padding + self.buffer
        else:
            seq = self.buffer[-self.seq_len:]
        
        return np.array(seq)
    
    def reset(self):
        self.buffer = []
        self.hidden = None


class LSTMAgent:
    def __init__(self, state_size=8, action_size=2, seq_len=10):
        self.state_size = state_size
        self.action_size = action_size
        self.seq_len = seq_len
        
        self.q_network = ImprovedLSTMQNetwork(state_size, action_size, seq_len=seq_len).to(DEVICE)
        self.target_network = ImprovedLSTMQNetwork(state_size, action_size, seq_len=seq_len).to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.buffer = LSTMSequenceBuffer(seq_len, state_size)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.tau = 0.005
        
        self.step_count = 0
    
    def act(self, state, training=True):
        self.buffer.add(state)
        sequence = self.buffer.get_sequence()
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
            q_values = self.q_network.get_q_values(seq_tensor)
            return torch.argmax(q_values).item()
    
    def update(self):
        self.step_count += 1
        if self.step_count % 100 == 0:
            for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.tau * q_param.data + (1 - self.tau) * target_param.data)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, path)
        print(f"✅ LSTM Agent saved to {path}")
    
    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.step_count = checkpoint.get('step_count', 0)
            print(f"✅ LSTM Agent loaded from {path}")
            return True
        return False