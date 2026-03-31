# src/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from datetime import datetime
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleQNetwork(nn.Module):
    def __init__(self, state_size=8, action_size=2, hidden_size=128):
        super().__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.layer_norm1(x)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.layer_norm2(x)
        return self.fc4(x)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, state_size=8, action_size=2, model_path="models/agent.pth"):
        self.state_size = state_size
        self.action_size = action_size
        self.model_path = model_path
        
        # Optimized hyperparameters
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.tau = 0.001
        self.learning_rate = 0.0005
        
        self.policy_net = SimpleQNetwork(state_size, action_size).to(DEVICE)
        self.target_net = SimpleQNetwork(state_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        self.memory = ReplayBuffer(capacity=20000)
        
        self.step_count = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.losses = []
        
        self.stats = {
            'correct_alerts': 0,
            'false_alerts': 0,
            'correct_no_alerts': 0,
            'missed_alerts': 0
        }
        
        self.pseudo_labels = {}
        self.llm_advice_history = []
        self.confidence_history = []
        
        if os.path.exists(model_path):
            self.load_model()
    
    def act(self, state, training=True):
        """Original act method (kept for compatibility)"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()
    
    def act_with_confidence(self, state, training=True):
        """
        NEW: Action selection with confidence threshold for high precision
        Only alerts when VERY confident
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_tensor)
            probabilities = torch.softmax(q_values, dim=1)
            
            # Get best action and confidence
            confidence, action = torch.max(probabilities, dim=1)
            confidence = confidence.item()
            action = action.item()
            
            # Track confidence
            self.confidence_history.append(confidence)
            if len(self.confidence_history) > 1000:
                self.confidence_history = self.confidence_history[-1000:]
            
            # PRECISION FILTER: Only alert if VERY confident (>85%)
            if action == 1:
                if confidence > 0.85:  # High threshold for precision
                    return 1
                else:
                    # When uncertain, default to no alert
                    return 0
            else:
                return 0
    
    def calculate_reward(self, action, data):
        """
        NEW: Optimized reward function for maximum precision and recall
        """
        is_anomaly = False
        severity = 0
        confidence = 1.0
        
        # === BALANCED ANOMALY DETECTION ===
        # Temperature thresholds (optimized for your data)
        if data['temperature'] > 40:
            is_anomaly = True
            severity += min((data['temperature'] - 40) * 0.12, 0.8)
        elif data['temperature'] > 37:
            severity += 0.2
        
        # Humidity thresholds
        if data['humidity'] > 72:
            is_anomaly = True
            severity += min((data['humidity'] - 72) * 0.06, 0.6)
        elif data['humidity'] > 62:
            severity += 0.1
        
        # Status - weighted appropriately
        if data['status'].lower() in ['critical', 'danger']:
            is_anomaly = True
            severity += 0.9
        elif data['status'].lower() == 'warning':
            if data['temperature'] > 38 or data['humidity'] > 68:
                is_anomaly = True
                severity += 0.5
            else:
                severity += 0.2
        
        severity = min(severity, 1.0)
        alert = (action == 1)
        
        # === AGGRESSIVE REWARDS for high precision ===
        if alert:
            if is_anomaly:
                # Correct alert - good reward
                reward = 20 * severity
            else:
                # FALSE ALARM - EXTREME PENALTY (fixes precision)
                reward = -100
        else:
            if not is_anomaly:
                # Correct no-alert - small positive
                reward = 1
            else:
                # MISSED ALERT - High penalty (maintains recall)
                reward = -30 * severity
        
        return reward, is_anomaly
    
    def remember(self, state, action, reward, next_state, done):
        """Enhanced memory with priority for mistakes"""
        # Store bad experiences multiple times to learn faster
        if reward < -50:  # False alarms
            for _ in range(5):  # Store 5 times
                self.memory.push(state, action, reward, next_state, done)
        elif reward < -20:  # Missed alerts
            for _ in range(3):  # Store 3 times
                self.memory.push(state, action, reward, next_state, done)
        else:
            self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Enhanced training with prioritized learning"""
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)
        
        current_q = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            # Double DQN for stability
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target = rewards + (1 - dones) * self.gamma * next_q
        
        # Huber loss for robustness
        loss = nn.SmoothL1Loss()(current_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        self.losses.append(loss.item())
        if len(self.losses) > 100:
            self.losses = self.losses[-100:]
        
        self.step_count += 1
        if self.step_count % 100 == 0:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def apply_pseudo_labels(self, pseudo_labels):
        self.pseudo_labels.update(pseudo_labels)
    
    def apply_llm_advice(self, advice):
        if not advice:
            return
        
        if 'epsilon_adjustment' in advice:
            adj = advice['epsilon_adjustment']
            confidence = advice.get('confidence', 0.5)
            self.epsilon += adj * confidence
            self.epsilon = np.clip(self.epsilon, self.epsilon_min, 1.0)
            print(f"  📈 LLM adjusted epsilon to {self.epsilon:.3f}")
        
        if advice.get('focus') == 'exploitation' and advice.get('confidence', 0) > 0.7:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(0.0001, param_group['lr'] * 0.99)
        
        self.llm_advice_history.append(advice)
    
    def process(self, data):
        """Main processing method - uses confidence-based action"""
        state = self._extract_state(data)
        
        # Use the new confidence-based action selection
        action = self.act_with_confidence(state, training=True)
        
        reward, is_anomaly = self.calculate_reward(action, data)
        
        next_state = self._extract_state(data, shift=True)
        self.remember(state, action, reward, next_state, False)
        
        self.total_reward += reward
        self.episode_rewards.append(reward)
        if len(self.episode_rewards) > 1000:
            self.episode_rewards = self.episode_rewards[-1000:]
        
        self._update_stats(action, is_anomaly)
        
        return {
            'action': action,
            'reward': reward,
            'state': state,
            'is_anomaly': is_anomaly
        }
    
    def _update_stats(self, action, is_anomaly):
        if action == 1:
            if is_anomaly:
                self.stats['correct_alerts'] += 1
            else:
                self.stats['false_alerts'] += 1
        else:
            if is_anomaly:
                self.stats['missed_alerts'] += 1
            else:
                self.stats['correct_no_alerts'] += 1
    
    def _extract_state(self, data, shift=False):
        status_map = {'normal': 0, 'warning': 1, 'critical': 2, 'danger': 2}
        status = status_map.get(data['status'].lower(), 0)
        
        temp_norm = np.clip(data['temperature'] / 100.0, 0, 1)
        humidity_norm = np.clip(data['humidity'] / 100.0, 0, 1)
        status_norm = status / 2.0
        hour = datetime.now().hour / 24.0
        
        if shift:
            temp_norm += np.random.normal(0, 0.01)
            humidity_norm += np.random.normal(0, 0.01)
            temp_norm = np.clip(temp_norm, 0, 1)
            humidity_norm = np.clip(humidity_norm, 0, 1)
        
        return np.array([
            temp_norm,
            humidity_norm,
            status_norm,
            hour,
            temp_norm * humidity_norm,
            abs(temp_norm - 0.5),
            abs(humidity_norm - 0.5),
            (temp_norm + humidity_norm) / 2
        ], dtype=np.float32)
    
    def get_stats(self):
        """Enhanced stats with confidence tracking"""
        total = sum(self.stats.values())
        
        accuracy = (self.stats['correct_alerts'] + self.stats['correct_no_alerts']) / max(1, total)
        precision = self.stats['correct_alerts'] / max(1, self.stats['correct_alerts'] + self.stats['false_alerts'])
        recall = self.stats['correct_alerts'] / max(1, self.stats['correct_alerts'] + self.stats['missed_alerts'])
        f1 = 2 * (precision * recall) / max(1, precision + recall)
        
        avg_confidence = np.mean(self.confidence_history[-100:]) if self.confidence_history else 0
        
        return {
            **self.stats,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'epsilon': self.epsilon,
            'total_reward': self.total_reward,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_loss': np.mean(self.losses) if self.losses else 0,
            'memory_size': len(self.memory),
            'step_count': self.step_count,
            'avg_confidence': avg_confidence
        }
    
    def save_model(self, path=None):
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'policy': self.policy_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'stats': self.stats,
            'confidence_history': self.confidence_history[-1000:]
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path=None):
        path = path or self.model_path
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.policy_net.load_state_dict(checkpoint['policy'])
            self.target_net.load_state_dict(checkpoint['target'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.step_count = checkpoint.get('step_count', 0)
            self.stats = checkpoint.get('stats', self.stats)
            self.confidence_history = checkpoint.get('confidence_history', [])
            print(f"✅ Model loaded from {path}")
            return True
        return False