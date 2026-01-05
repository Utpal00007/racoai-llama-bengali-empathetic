import math
import evaluate

class Evaluator:
    def compute_perplexity(self, eval_loss):
        return math.exp(eval_loss)

    def compute_metrics(self, preds, refs):
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")

        return {
            "bleu": bleu.compute(predictions=preds, references=refs),
            "rouge": rouge.compute(predictions=preds, references=refs)
        }
