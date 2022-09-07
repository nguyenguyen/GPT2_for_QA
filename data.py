class NarrativeQADataset(object):
    """
    A single training/test example for the QA dataset.
    """

    def __init__(
        self,
        qa_id,
        question_text,
        answer_text,
        context_text,
        context_tokens,
    ):
        self.qa_id = qa_id
        self.question_text = question_text
        self.answer_text = answer_text
        self.context_text = context_text
        self.context_tokens = context_tokens

    def __str__(self):
        return f"qa_id: {self.qa_id}, question_text: {self.question_text}, context_text: {self.context_text}"


class NarrativeQAInputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        # unique_id,
        # example_index,
        # doc_span_index,
        # tokens,
        # token_to_orig_map,
        # token_is_max_context,
        input_ids,
        input_mask,
        segment_ids,
    ):
        # self.unique_id = unique_id
        # self.example_index = example_index
        # self.doc_span_index = doc_span_index
        # self.tokens = tokens
        # self.token_to_orig_map = token_to_orig_map
        # self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

    def __str__(self):
        return f"\n input_ids: {self.input_ids} \n\n input_mask: {self.input_mask} \n\n segment_ids: {self.segment_ids}"
