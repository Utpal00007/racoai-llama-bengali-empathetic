class DatasetProcessor:
    """
    Prepares Bengali empathetic conversation data
    for LLaMA fine-tuning.
    """

    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess(self, data):
        """
        data: list of dicts with keys: 'input', 'response'
        """

        texts = [
            f"<|user|>{item['input']}<|assistant|>{item['response']}"
            for item in data
        ]

        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=False,  # IMPORTANT: sequence length not reduced
            max_length=self.max_length,
            return_tensors="pt"
        )

        return encodings
