"""
finetune_bert_classification.py

Fine-tune BERT for sequence classification tasks.

Contains:
  - TextClassificationDataset: PyTorch Dataset for tokenized text and labels.
  - train_bert_classification: Function to train and evaluate a BERT classifier.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Data Preparation
class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset for text classification.

    Attributes:
        texts (List[str]): List of input text samples.
        labels (List[int]): Corresponding integer labels.
        tokenizer (PreTrainedTokenizer): Tokenizer for converting text to tokens.
        max_len (int): Maximum token sequence length.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        """
        Initialize the dataset with texts, labels, and tokenizer.

        Args:
            texts (List[str]): Input text samples.
            labels (List[int]): Classification labels for each sample.
            tokenizer (PreTrainedTokenizer): Tokenizer instance.
            max_len (int): Maximum sequence length for padding/truncation.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Dataset size.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieve a tokenized sample and its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'text': original text string
                - 'input_ids': Tensor of token ids
                - 'attention_mask': Tensor of attention mask
                - 'labels': Tensor of the label
        """
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',  # Encode the text with padding and attention mask
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_bert_classification(data, num_labels, model_name='bert-base-uncased', max_len=128, batch_size=16, num_epochs=3, learning_rate=2e-5):
    """
    Fine-tune BERT for sequence classification.

    Args:
        data (List[Tuple[str, int]]): List of (text, label) pairs.
        num_labels (int): Number of target classes.
        model_name (str): Hugging Face model identifier.
        max_len (int): Maximum token sequence length.
        batch_size (int): Batch size for training and evaluation.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        BertForSequenceClassification: Trained BERT model.
    """
    # Split data into texts and labels
    texts, labels = zip(*data)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Prepare device and move model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Prepare DataLoaders
    train_dataloader = DataLoader(
        TextClassificationDataset(train_texts, train_labels, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        TextClassificationDataset(val_texts, val_labels, tokenizer, max_len),
        batch_size=batch_size
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch}/{num_epochs} - Average training loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)

                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

        for name, metric in [
            ("Accuracy", accuracy),
            ("Precision", precision),
            ("Recall", recall),
            ("F1-score", f1),
        ]:
            print(f"  Validation {name}: {metric:.4f}")

    # Return the fine-tuned model
    return model

if __name__ == '__main__':
    # Example Usage: Dummy data for sentiment classification (positive/negative)
    sample_data = [
        ("This movie is fantastic and I love it!", 1),
        ("I hate this film, it's terrible.", 0),
        ("What a wonderful experience, highly recommend.", 1),
        ("Never again, completely disappointed.", 0),
        ("It was okay, not great but not bad either.", 1), # Neutral, but for binary, assign to one class
        ("Absolutely brilliant performance.", 1),
        ("Worst book I've ever read.", 0),
        ("So happy with the results.", 1),
        ("A complete waste of time and money.", 0),
        ("Enjoyed every minute of it.", 1),
    ]
    num_classes = 2 # 0 for negative, 1 for positive

    print("Starting BERT fine-tuning example...")
    trained_model = train_bert_classification(sample_data, num_classes)
    print("BERT fine-tuning example completed.")

    # You can save the trained model:
    # trained_model.save_pretrained("./my_bert_classifier")
    # tokenizer.save_pretrained("./my_bert_classifier")
