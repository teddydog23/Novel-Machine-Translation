from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
import sacrebleu

def calculate_bleu(references, hypotheses):
    """
    references: List of list of tokens (ground truth)
    hypotheses: List of list of tokens (predicted)
    """
    smoothie = SmoothingFunction().method4
    scores = []
    for ref, hyp in zip(references, hypotheses):
        score = sentence_bleu([ref], hyp, smoothing_function=smoothie)
        scores.append(score)
    return sum(scores) / len(scores)