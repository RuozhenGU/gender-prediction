import csv
import numpy as np
import pandas as pd
from numpy.random import randn

class RNN:
    # A Recurrent Neural Network.

    def __init__(self, input_size, output_size, hidden_size=64, norm=1000):

        self.Whh = randn(hidden_size, hidden_size) / norm
        self.Wxh = randn(hidden_size, input_size) / norm
        self.Why = randn(output_size, hidden_size) / norm

        # bias
        self.bh, self.by = np.zeros((hidden_size, 1)), np.zeros((output_size, 1))

    def forward(self, inputs):
        """
        forward pass of the RNN
        """
        h = np.zeros((self.Whh.shape[0], 1))

        self.prev_input = inputs
        self.last_hs = {0: h}

        # exec one step
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        return self.Why @ h + self.by, h

    def backprop(self, d_y, lr=2e-2):
        """
        backward pass of the RNN
        """

        n = len(self.prev_input)
        
        d_Why, d_by = d_y @ self.last_hs[n].T, d_y

        # Initialize as zero
        d_Whh, d_Wxh = np.zeros(self.Whh.shape), np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        d_h = self.Why.T @ d_y

        # Backpropagate through time.
        for t in reversed(range(n)):

            d_bh += (1 - self.last_hs[t + 1] ** 2) * d_h

            d_Whh += (1 - self.last_hs[t + 1] ** 2) * d_h @ self.last_hs[t].T

            d_Wxh += (1 - self.last_hs[t + 1] ** 2) * d_h @ self.prev_input[t].T

            d_h = self.Whh @ (1 - self.last_hs[t + 1] ** 2) * d_h

        # clip it so no exploding gradients
        np.clip(d_Wxh, -1, 1, out=d_Wxh)
        np.clip(d_Whh, -1, 1, out=d_Whh)
        np.clip(d_Why, -1, 1, out=d_Why)
        np.clip(d_bh, -1, 1, out=d_bh)
        np.clip(d_by, -1, 1, out=d_by)

        # Update
        self.bh = self.bh - lr * d_bh
        self.by = self.by - lr * d_by

        self.Whh = self.Whh - lr * d_Whh
        self.Wxh = self.Wxh - lr * d_Wxh
        self.Why = self.Why - lr * d_Why