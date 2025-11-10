import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MDNLayer(nn.Module):
    """
    MDN Layer
    """
    def __init__(self, input_dim, output_dim, num_mixtures):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_mixtures = num_mixtures

        self.pi_layer = nn.Linear(input_dim, num_mixtures)
        self.sigma_layer = nn.Linear(input_dim, num_mixtures * output_dim)
        self.mu_layer = nn.Linear(input_dim, num_mixtures * output_dim)
    
    def forward(self, h):
        pi = self.pi_layer(h)
        pi = F.softmax(pi, dim=-1)
        
        sigma = torch.exp(self.sigma_layer(h))
        sigma = sigma.view(h.size(0), h.size(1), self.num_mixtures, self.output_dim)

        mu = self.mu_layer(h)
        mu = mu.view(h.size(0), h.size(1), self.num_mixtures, self.output_dim)

        return pi, sigma, mu
    
def mdn_loss(pi, sigma, mu, target):
    """Computing the MDN loss"""
    target= target.unsqueeze(2)
    norm = (target - mu) / sigma
    prob = -0.5 * (norm ** 2) - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
    prob = prob.sum(dim=-1)

    log_pi = torch.log(pi + 1e-8)
    log_likelihood = torch.logsumexp(log_pi + prob, dim=-1)

    return -torch.mean(log_likelihood)

class HandwritingRNN(nn.Module):
    """Main RNN (LSTM) model using MDN."""
    def __init__(self, input_dim, hidden_dim, num_mixtures, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.mdn = MDNLayer(hidden_dim, output_dim=2, num_mixtures=num_mixtures)
        self.pen_up_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, h_c=None):
        lstm_out = F.relu(self.fc(lstm_out))
        pi, sigma, mu = self.mdn(lstm_out)
        pen_up = torch.sigmoid(self.pen_up_layer(lstm_out))

        return pi, sigma, mu, pen_up, (h, c)
    
    def calculate_loss(self, pi, sigma, mu, pen_up, target_sequence):
        """
        Calculating total loss
        """
        target_dx_dy = target_sequence[:, :, :2]
        target_p1 = target_sequence[:, :, 2].unsqueeze(-1)

        loss_mdn = mdn_loss(pi, sigma, mu, target_dx_dy)
        loss_p1 = F.binary_cross_entropy(pen_up, target_p1, reduction='mean')
        total_loss = loss_mdn + loss_p1

        return total_loss, loss_mdn, loss_p1