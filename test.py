import evaluate
from config import EVALUATION_CACHE_DIR


if __name__ == '__main__':
    # bleu = evaluate.load("bleu", cache_dir=EVALUATION_CACHE_DIR)
    # predictions = ["hello there general kenobi", "foo bar foobar"]
    # references = [
    #     ["hello there !"],
    #     ["foo bar foobar"]
    # ]
    # results = bleu.compute(predictions=predictions, references=references, max_order=1)
    # print(results)

    import statistics
    from nltk.translate.bleu_score import sentence_bleu

    predictions = ["hello there general kenobi"]
    references = [
        ["hello there !"]
    ]
    print(sentence_bleu(references[0], predictions[0][0]))

    # scores = []
    #
    # for i in range(len(test_set)):
    #     reference = test_set['True_end_lyrics'][i]
    #     candidate = test_set['Generated_lyrics'][i]
    #     scores.append(sentence_bleu(reference, candidate))
    #
    # statistics.mean(scores)