import evaluate
from config import EVALUATION_CACHE_DIR


if __name__ == '__main__':
    bleu = evaluate.load("bleu", cache_dir=EVALUATION_CACHE_DIR)
    predictions = ["hello there general kenobi", "foo bar foobar"]
    references = [
        ["hello there !"],
        ["foo bar foobar"]
    ]
    results = bleu.compute(predictions=predictions, references=references, max_order=1)
    print(results)
