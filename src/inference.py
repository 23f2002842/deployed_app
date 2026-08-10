import torch
import torch.nn as nn
from transformers import AutoTokenizer
import numpy as np

# 1. Define the exact architecture so PyTorch can load the weights
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(128, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded) 
        
        attn_scores = self.attention_weights(lstm_out).squeeze(-1) 
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9) 
        attn_probs = torch.softmax(attn_scores, dim=-1).unsqueeze(-1) 
        context_vector = torch.sum(attn_probs * lstm_out, dim=1) 
        
        logits = self.classifier(context_vector).view(batch_size, 5)      
        return logits

# 2. Function to load the model (forces CPU for Streamlit Cloud)
def load_bilstm(model_path="models/bilstm_model.pt"):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BiLSTMAttention(vocab_size=tokenizer.vocab_size)
    # map_location='cpu' is critical here!
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, tokenizer

# 3. Function to process a single user input
def predict_top3(prompt, options, model, tokenizer):
    options_list = ['A', 'B', 'C', 'D', 'E']
    
    encodings = tokenizer(
        [prompt] * 5, options, truncation=True, padding='max_length', 
        max_length=512, return_tensors='pt'
    )
    
    with torch.no_grad():
        # Add a batch dimension for a single instance -> shape (1, 5, 512)
        input_ids = encodings['input_ids'].unsqueeze(0)
        attention_mask = encodings['attention_mask'].unsqueeze(0)
        
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
        
    # Get top 3 indices
    top3_indices = np.argsort(probs)[::-1][:3]
    
    # Return a list of tuples: (Letter, Text, Probability)
    results = [(options_list[i], options[i], probs[i]) for i in top3_indices]
    return results
