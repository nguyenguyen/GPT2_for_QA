import evaluate
from config import EVALUATION_CACHE_DIR


if __name__ == '__main__':
    bleu = evaluate.load("blue", cache_dir=EVALUATION_CACHE_DIR)
    predictions = ["hello there general kenobi", "foo bar foobar"]
    references = [
        ["hello there general kenobi", "hello there !"],
        ["foo bar foobar"]
    ]
    results = bleu.compute(predictions=predictions, references=references)
