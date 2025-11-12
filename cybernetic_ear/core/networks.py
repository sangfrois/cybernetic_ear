import torch
import torch.nn as nn

class GrooveGRU(nn.Module):
    """
    A GRU-based model to analyze and predict microtiming deviations.
    """
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, output_size=1):
        """
        Initializes the GrooveGRU model.

        Parameters
        ----------
        input_size : int
            The number of input features (should be 1 for microtiming deviations).
        hidden_size : int
            The number of features in the hidden state.
        num_layers : int
            The number of recurrent layers.
        output_size : int
            The number of output features.
        """
        super(GrooveGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        """
        Performs a forward pass through the GRU model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, seq_length, input_size).
        h : torch.Tensor, optional
            The initial hidden state, by default None.

        Returns
        -------
        out : torch.Tensor
            The output tensor of shape (batch_size, output_size).
        h : torch.Tensor
            The final hidden state.
        """
        if h is None:
            h = self.init_hidden(x.size(0))
            
        out, h = self.gru(x, h)
        
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state of the GRU.

        Parameters
        ----------
        batch_size : int
            The batch size.

        Returns
        -------
        torch.Tensor
            The initial hidden state.
        """
        # The hidden state is a tensor of shape (num_layers, batch_size, hidden_size)
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.gru.weight_ih_l0.device)

    def process(self, deviations):
        """
        Processes a sequence of microtiming deviations and returns a groove pattern.

        Parameters
        ----------
        deviations : list or np.ndarray
            A sequence of microtiming deviations.

        Returns
        -------
        float
            The calculated groove pattern.
        """
        if not isinstance(deviations, torch.Tensor):
            deviations = torch.tensor(deviations, dtype=torch.float32)
        
        # Reshape the input for the GRU: (batch_size, seq_length, input_size)
        deviations = deviations.view(1, -1, 1)
        
        # We don't need to train the model here, just use it for inference
        self.eval()
        with torch.no_grad():
            output = self.forward(deviations)
            
        return output.item()