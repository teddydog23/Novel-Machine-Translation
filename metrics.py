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

def calculate_corpus_bleu(predictions, references):
    chencherry = SmoothingFunction()
    # References phải là dạng List[List[List[str]]], predictions là List[List[str]]
    formatted_refs = [[ref] for ref in references]
    return corpus_bleu(formatted_refs, predictions, smoothing_function=chencherry.method1)

def calculate_meteor(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        score = single_meteor_score([ref], pred)  # ✅ fix: truyền list[str], list[str]
        scores.append(score)
    return sum(scores) / len(scores)

def calculate_chrf(references, hypotheses):
    refs = [" ".join(ref) for ref in references]
    hyps = [" ".join(hyp) for hyp in hypotheses]
    return sacrebleu.corpus_chrf(refs, hyps).score


# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score

# def calculate_bleu(predictions, references):
#     chencherry = SmoothingFunction()
#     scores = [
#         sentence_bleu([ref], pred, smoothing_function=chencherry.method1)
#         for pred, ref in zip(predictions, references)
#     ]
#     return sum(scores) / len(scores)

# def calculate_meteor(predictions, references):
#     scores = [
#         meteor_score([' '.join(ref)], ' '.join(pred))
#         for pred, ref in zip(predictions, references)
#     ]
#     return sum(scores) / len(scores)