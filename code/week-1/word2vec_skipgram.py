

"""
word2vec_skipgram.py

Implementation of the Word2Vec skip-gram model with negative sampling.

Contains:
  - Word2VecSkipGram class for embedding lookup and scoring.
  - train_skipgram function to generate training data and train the model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random

class Word2VecSkipGram(nn.Module):
    """
    Word2Vec Skip-Gram model.

    Attributes:
        in_embeddings (nn.Embedding): Embedding lookup for target words.
        out_embeddings (nn.Embedding): Embedding lookup for context words.
    """
    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize in- and out- embedding layers.

        Args:
            vocab_size (int): Number of unique words in the vocabulary.
            embedding_dim (int): Dimensionality of the embedding vectors.
        """
        super(Word2VecSkipGram, self).__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target_word_idx, context_word_idx):
        """
        Compute the dot-product score for target and context word embeddings.

        Args:
            target_word_idx (Tensor): Tensor of target word indices, shape (batch_size,).
            context_word_idx (Tensor): Tensor of context word indices, shape (batch_size,).

        Returns:
            Tensor: Dot-product scores, shape (batch_size,).
        """
        # Lookup embedding for target words
        target_embedding = self.in_embeddings(target_word_idx)
        # Lookup embedding for context words
        context_embedding = self.out_embeddings(context_word_idx)
        # Compute dot-product score between target and context embeddings
        score = torch.sum(target_embedding * context_embedding, dim=1)
        return score

def train_skipgram(corpus, vocab_size, embedding_dim, window_size, num_epochs, learning_rate):
    """
    Train the Word2Vec skip-gram model with negative sampling.

    Args:
        corpus (list of str): Sequence of words from the training text.
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimensionality of word embeddings.
        window_size (int): Context window size on each side of target.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        model (Word2VecSkipGram): Trained skip-gram model.
        word_to_idx (dict): Mapping from word to its index in the vocabulary.
    """
    # Build mapping from words to integer indices
    word_to_idx = {word: i for i, word in enumerate(set(corpus))}
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    model = Word2VecSkipGram(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Prepare positive context pairs
    positive_pairs = [
        (i, j)
        for i in range(len(corpus))
        for j in range(max(0, i - window_size), min(len(corpus), i + window_size + 1))
        if i != j
    ]
    # Generate training samples with negative sampling
    data = []
    for target_idx, context_idx in positive_pairs:
        # Positive sample
        data.append((target_idx, context_idx, 1))
        # Negative sample: pick a random index not equal to target or context
        negative_idx = random.choice([
            idx for idx in range(vocab_size)
            if idx not in (target_idx, context_idx)
        ])
        data.append((target_idx, negative_idx, 0))

    # Training loop over epochs
    for epoch in range(num_epochs):
        total_loss = 0
        for target_idx, context_idx, label in data:
            # Convert indices and label to tensors
            target_idx_tensor = torch.tensor(target_idx).unsqueeze(0)
            context_idx_tensor = torch.tensor(context_idx).unsqueeze(0)
            label_tensor = torch.tensor(label, dtype=torch.float).unsqueeze(0)

            # Zero gradients
            optimizer.zero_grad()
            # Forward pass to compute score
            score = model(target_idx_tensor, context_idx_tensor)
            loss = criterion(score, label_tensor)
            # Backpropagate loss
            loss.backward()
            # Update model parameters
            optimizer.step()
            total_loss += loss.item()
        # Log average loss per epoch
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss / len(data):.4f}")

    return model, word_to_idx

if __name__ == '__main__':
    corpus = ["king", "queen", "man", "woman", "royal", "throne", "prince", "princess"]
    vocab_size = len(set(corpus))
    embedding_dim = 10
    window_size = 2
    num_epochs = 100
    learning_rate = 0.01

    model, word_to_idx = train_skipgram(corpus, vocab_size, embedding_dim, window_size, num_epochs, learning_rate)

    # Example usage: find similarity
    word1 = "king"
    word2 = "queen"
    word3 = "man"

    if word1 in word_to_idx and word2 in word_to_idx and word3 in word_to_idx:
        embedding1 = model.in_embeddings(torch.tensor([word_to_idx[word1]]))
        embedding2 = model.in_embeddings(torch.tensor([word_to_idx[word2]]))
        embedding3 = model.in_embeddings(torch.tensor([word_to_idx[word3]]))

        # Simple cosine similarity
        cosine_similarity = nn.CosineSimilarity(dim=1)
        sim_king_queen = cosine_similarity(embedding1, embedding2)
        sim_king_man = cosine_similarity(embedding1, embedding3)

        print(f"Similarity between {word1} and {word2}: {sim_king_queen.item()}")
        print(f"Similarity between {word1} and {word3}: {sim_king_man.item()}")
    else:
        print("One or more words not found in vocabulary.")


