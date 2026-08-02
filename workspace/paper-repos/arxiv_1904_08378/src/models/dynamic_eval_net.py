import torch
import torch.nn as nn

class DynamicEvaluator:
    """Test-time adaptation via gradient descent."""
    def __init__(self, model, optimizer_type='rmsprop', learning_rate=0.01):
        self.model = model
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.mems = None
    
    def forward(self, segment_ids, num_steps=1):
        """Forward with gradient descent adaptation."""
        if self.optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        for _ in range(num_steps):
            optimizer.zero_grad()
            logits, self.mems = self.model(segment_ids, mems=self.mems)
            loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), 
                                               segment_ids[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
        
        return logits, loss.item()
