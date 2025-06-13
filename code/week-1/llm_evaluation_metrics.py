
"""
llm_evaluation_metrics.py

Google-style implementation of evaluation metrics for LLMs,
including perplexity, statistical, and embedding-based metrics.

Defines:
  - PerplexityCalculator: Basic, sliding, and batch perplexity methods.
  - StatisticalMetrics: BLEU and ROUGE-L calculations.
  - BERTScoreCalculator: Embedding-based BERTScore.
  - EvaluationPipeline: Orchestrates comprehensive evaluation.
  - Demonstration function for healthcare LLM evaluation.
"""

import math
import warnings

from rouge_score import rouge_scorer

from typing import List, Dict, Tuple, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset

from tqdm import tqdm

class PerplexityCalculator:
    """
    PerplexityCalculator computes different perplexity metrics for causal LMs.

    Args:
        model (nn.Module): Pre-trained causal language model.
        tokenizer (PreTrainedTokenizer): Corresponding tokenizer.
        device (str): 'cuda' or 'cpu'.
        max_length (int): Maximum context length.
    """
    
    def __init__(self, model, tokenizer, device='cuda', max_length=1024):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()
        
    def calculate_perplexity_basic(self, text: str) -> float:
        """
        Compute perplexity for a single short text.

        Args:
            text (str): Input text string.

        Returns:
            float: Perplexity score.
        """
        # Tokenize the input text
        encodings = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length)
        input_ids = encodings.input_ids.to(self.device)
        # Forward pass with labels for loss computation
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            # Convert negative log-likelihood loss to perplexity
            perplexity = torch.exp(loss)
        return perplexity.item()

    def calculate_perplexity_sliding_window(self, text: str, stride: int = 512) -> float:
        """
        Compute perplexity over long text using sliding window.

        Args:
            text (str): Input text string.
            stride (int): Stride for moving window.

        Returns:
            float: Perplexity score.
        """
        # Tokenize the full text
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids.squeeze(0)
        seq_len = input_ids.size(0)
        if seq_len <= self.max_length:
            # Use basic approach for short sequences
            return self.calculate_perplexity_basic(text)
        # Sliding window calculation for long text
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            # Prepare input and target ids for window
            input_ids_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(self.device)
            target_ids = input_ids_chunk.clone()
            target_ids[:, :-trg_len] = -100  # Ignore tokens without full context
            with torch.no_grad():
                outputs = self.model(input_ids_chunk, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        # Aggregate negative log-likelihood and convert to perplexity
        avg_nll = torch.stack(nlls).sum() / (seq_len - 1)
        perplexity = torch.exp(avg_nll)
        return perplexity.item()

    def calculate_perplexity_batch(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """
        Compute perplexities for a batch of texts.

        Args:
            texts (List[str]): List of input texts.
            batch_size (int): Batch size.

        Returns:
            List[float]: Perplexity per text.
        """
        perplexities = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # Tokenize batch with padding
            encodings = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
            input_ids = encodings.input_ids.to(self.device)
            attention_mask = encodings.attention_mask.to(self.device)
            # Create labels (input_ids, but pad tokens set to -100)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                # Calculate perplexity for each sequence in the batch
                for j in range(len(batch_texts)):
                    seq_labels = labels[j]
                    valid_tokens = (seq_labels != -100).sum().item()
                    if valid_tokens > 0:
                        # Calculate sequence-specific cross-entropy loss
                        seq_logits = outputs.logits[j]
                        seq_loss = F.cross_entropy(
                            seq_logits[:-1].view(-1, seq_logits.size(-1)),
                            seq_labels[1:].view(-1),
                            ignore_index=-100,
                            reduction='sum'
                        ) / (valid_tokens - 1)
                        perplexity = torch.exp(seq_loss).item()
                        perplexities.append(perplexity)
                    else:
                        perplexities.append(float('inf'))
        return perplexities

def _lcs_length(x: List[str], y: List[str]) -> int:
    """
    Compute the length of the longest common subsequence between two token lists.
    """
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

class StatisticalMetrics:
    """
    StatisticalMetrics computes BLEU and ROUGE-L scores.

    Methods:
        calculate_bleu_score: BLEU-n score.
        calculate_rouge_l_score: ROUGE-L F1 score.
    """

    @staticmethod
    def calculate_bleu_score(candidate: str, reference: str, n_gram: int = 4) -> float:
        """
        Compute BLEU-n score between candidate and reference.

        Args:
            candidate (str): Generated text.
            reference (str): Reference text.
            n_gram (int): Maximum n-gram order (default: 4).

        Returns:
            float: BLEU score.
        """
        # Tokenize and extract n-grams
        def get_ngrams(text: str, n: int) -> List[Tuple]:
            tokens = text.lower().split()
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        candidate_tokens = candidate.lower().split()
        reference_tokens = reference.lower().split()
        if len(candidate_tokens) == 0:
            return 0.0
        precisions = []
        for n in range(1, n_gram + 1):
            candidate_ngrams = get_ngrams(candidate, n)
            reference_ngrams = get_ngrams(reference, n)
            if len(candidate_ngrams) == 0:
                precisions.append(0.0)
                continue
            # Count n-grams
            from collections import Counter
            candidate_counts = Counter(candidate_ngrams)
            reference_counts = Counter(reference_ngrams)
            # Clipped count for BLEU
            clipped_counts = 0
            total_counts = 0
            for ngram, count in candidate_counts.items():
                clipped_counts += min(count, reference_counts.get(ngram, 0))
                total_counts += count
            precision = clipped_counts / total_counts if total_counts > 0 else 0.0
            precisions.append(precision)
        # Geometric mean of n-gram precisions
        if all(p > 0 for p in precisions):
            bleu_score = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            bleu_score = 0.0
        # Brevity penalty for short candidates
        candidate_length = len(candidate_tokens)
        reference_length = len(reference_tokens)
        if candidate_length > reference_length:
            brevity_penalty = 1.0
        else:
            brevity_penalty = math.exp(1 - reference_length / candidate_length)
        return bleu_score * brevity_penalty

    @staticmethod
    def calculate_rouge_l_score(candidate: str, reference: str) -> Dict[str, float]:
        """
        Compute ROUGE-L F1 score between candidate and reference using Google's rouge_scorer.

        Args:
            candidate (str): Generated text.
            reference (str): Reference text.

        Returns:
            Dict[str, float]: {'precision': float, 'recall': float, 'f1': float}.
        """
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        score = scorer.score(reference, candidate)['rougeL']
        return {
            'precision': score.precision,
            'recall': score.recall,
            'f1': score.fmeasure
        }

class BERTScoreCalculator:
    """
    BERTScoreCalculator uses BERT embeddings to compute precision, recall, F1.

    Args:
        model_name (str): Hugging Face model name.
        device (str): 'cuda' or 'cpu'.
    """

    def __init__(self, model_name: str = 'bert-base-uncased', device: str = 'cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def calculate_bertscore(self, candidates: List[str], references: List[str]) -> Dict[str, List[float]]:
        """
        Compute BERTScore for candidate-reference pairs.

        Args:
            candidates (List[str]): Generated texts.
            references (List[str]): Reference texts.

        Returns:
            Dict[str, List[float]]: Precision, recall, F1 per pair.
        """
        assert len(candidates) == len(references), "Candidates and references must have same length"
        precisions, recalls, f1s = [], [], []
        for candidate, reference in zip(candidates, references):
            # Tokenize and encode texts
            cand_tokens = self.tokenizer(candidate, return_tensors='pt', padding=True, truncation=True).to(self.device)
            ref_tokens = self.tokenizer(reference, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                # Get contextual embeddings
                cand_embeddings = self.model(**cand_tokens).last_hidden_state.squeeze(0)
                ref_embeddings = self.model(**ref_tokens).last_hidden_state.squeeze(0)
                # Normalize and compute pairwise cosine similarity
                cand_norm = F.normalize(cand_embeddings, p=2, dim=1)
                ref_norm = F.normalize(ref_embeddings, p=2, dim=1)
                similarity_matrix = torch.mm(cand_norm, ref_norm.transpose(0, 1))
                # Precision: each candidate token's max similarity to any reference token
                precision = similarity_matrix.max(dim=1)[0].mean().item()
                # Recall: each reference token's max similarity to any candidate token
                recall = similarity_matrix.max(dim=0)[0].mean().item()
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
        return {
            'precision': precisions,
            'recall': recalls,
            'f1': f1s
        }

class EvaluationPipeline:
    """
    EvaluationPipeline aggregates multiple metrics for LLMs.

    Args:
        model (nn.Module): Causal LM.
        tokenizer (PreTrainedTokenizer): Corresponding tokenizer.
        device (str): 'cuda' or 'cpu'.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.perplexity_calc = PerplexityCalculator(model, tokenizer, device)
        self.bertscore_calc = BERTScoreCalculator(device=device)
        self.statistical_metrics = StatisticalMetrics()

    def evaluate_comprehensive(
        self,
        generated_texts: List[str],
        reference_texts: List[str],
        source_texts: Optional[List[str]] = None
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Run perplexity, BERTScore, BLEU, and ROUGE-L on texts.

        Args:
            generated_texts (List[str]): Model outputs.
            reference_texts (List[str]): Ground truth texts.
            source_texts (Optional[List[str]]): Texts for perplexity.

        Returns:
            Dict[str, Any]: Aggregated metric results.
        """
        results = {}
        # Compute perplexity for generated or provided source texts
        if source_texts is None:
            source_texts = generated_texts
        perplexities = self.perplexity_calc.calculate_perplexity_batch(source_texts)
        results['perplexity'] = {
            'mean': np.mean(perplexities),
            'std': np.std(perplexities),
            'values': perplexities
        }
        # Compute BERTScore metrics
        bertscore_results = self.bertscore_calc.calculate_bertscore(generated_texts, reference_texts)
        results['bertscore'] = {
            'precision': {
                'mean': np.mean(bertscore_results['precision']),
                'std': np.std(bertscore_results['precision']),
                'values': bertscore_results['precision']
            },
            'recall': {
                'mean': np.mean(bertscore_results['recall']),
                'std': np.std(bertscore_results['recall']),
                'values': bertscore_results['recall']
            },
            'f1': {
                'mean': np.mean(bertscore_results['f1']),
                'std': np.std(bertscore_results['f1']),
                'values': bertscore_results['f1']
            }
        }
        # Compute BLEU and ROUGE-L metrics for each pair
        bleu_scores = []
        rouge_scores = {'precision': [], 'recall': [], 'f1': []}
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            bleu = self.statistical_metrics.calculate_bleu_score(gen_text, ref_text)
            rouge = self.statistical_metrics.calculate_rouge_l_score(gen_text, ref_text)
            bleu_scores.append(bleu)
            rouge_scores['precision'].append(rouge['precision'])
            rouge_scores['recall'].append(rouge['recall'])
            rouge_scores['f1'].append(rouge['f1'])
        results['bleu'] = {
            'mean': np.mean(bleu_scores),
            'std': np.std(bleu_scores),
            'values': bleu_scores
        }
        results['rouge_l'] = {
            'precision': {
                'mean': np.mean(rouge_scores['precision']),
                'std': np.std(rouge_scores['precision']),
                'values': rouge_scores['precision']
            },
            'recall': {
                'mean': np.mean(rouge_scores['recall']),
                'std': np.std(rouge_scores['recall']),
                'values': rouge_scores['recall']
            },
            'f1': {
                'mean': np.mean(rouge_scores['f1']),
                'std': np.std(rouge_scores['f1']),
                'values': rouge_scores['f1']
            }
        }
        return results

# Example usage and demonstration
def demonstrate_evaluation_metrics():
    """
    Demonstrate EvaluationPipeline on healthcare example texts.

    Returns:
        None
    """
    # Sample healthcare texts for demonstration
    reference_texts = [
        "Patient presents with acute chest pain radiating to left arm. ECG shows ST elevation in leads II, III, aVF consistent with inferior STEMI.",
        "Blood pressure 140/90 mmHg, heart rate 85 bpm. Patient reports intermittent palpitations over past week.",
        "Laboratory results show elevated troponin levels at 15.2 ng/mL, indicating myocardial injury."
    ]
    generated_texts = [
        "Patient has severe chest pain going to left arm. Heart test shows changes suggesting heart attack in lower wall.",
        "BP is 140/90, pulse 85. Patient says heart racing sometimes in last week.",
        "Lab tests show high troponin at 15.2, which means heart muscle damage."
    ]
    print("Healthcare LLM Evaluation Demonstration")
    print("=" * 50)
    # Initialize with a small model for demonstration
    model_name = "gpt2"  # Replace with your model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Initialize evaluation pipeline
        pipeline = EvaluationPipeline(model, tokenizer, device='cpu')  # Use CPU for demo
        # Perform comprehensive evaluation
        results = pipeline.evaluate_comprehensive(generated_texts, reference_texts)
        # Display results
        print(f"Perplexity: {results['perplexity']['mean']:.2f} ± {results['perplexity']['std']:.2f}")
        print(f"BERTScore F1: {results['bertscore']['f1']['mean']:.3f} ± {results['bertscore']['f1']['std']:.3f}")
        print(f"BLEU Score: {results['bleu']['mean']:.3f} ± {results['bleu']['std']:.3f}")
        print(f"ROUGE-L F1: {results['rouge_l']['f1']['mean']:.3f} ± {results['rouge_l']['f1']['std']:.3f}")
    except ImportError:
        print("Transformers library not available. Install with: pip install transformers")
    except Exception as e:
        print(f"Error in demonstration: {e}")


if __name__ == "__main__":
    demonstrate_evaluation_metrics()

