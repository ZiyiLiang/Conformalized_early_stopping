import torch.nn as nn
import torch as th
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_size=32):
        super(SimpleNN, self).__init__()
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_size = hidden_size
    
        self.build_model()
        self.init_weights()
        
    def init_weights(self):
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                th.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
    
    def build_model(self):
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.out_shape),
        )
        
    def forward(self, x):
        """
        Forward pass
        """
        return self.base_model(x)
    
    def predict_prob(self, inputs):
        """
        Predict probabilities given any input data
        """
        self.eval()
       
        get_prob = nn.Softmax(dim = 1)
        with th.no_grad():
            logits = self(inputs)    # Assume inputs are already on the correct device
            prob = get_prob(logits).cpu().numpy()
        return prob
    
    # Let the anomaly score be the probability of being an outlier
    def get_anomaly_scores(self,inputs):
        if len(inputs.shape) == 1:
            inputs = inputs[None,]
        full_prob = self.predict_prob(inputs)
        return list(full_prob[:,1])