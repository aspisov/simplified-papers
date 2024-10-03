from transformers import pipeline
from tqdm import tqdm
import nltk

nltk.download("punkt")


class Summarizer:
    """Summarizes text using a specified model and device."""

    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6", device="cpu"):
        """Initializes the Summarizer with a specified model and device.

        Args:
            model_name (str): The name of the model to use for summarization.
            device (str): The device to run the model on, e.g., 'cpu' or 'cuda'.
        """
        self.model = pipeline("summarization", model=model_name, device=device)

    def summarize(self, text, max_length=3000):
        """Summarizes the input text.

        Args:
            text (str): The text to summarize.
            max_length (int): The maximum length of the text.

        Returns:
            str: The summarized text.
        """
        while len(text) > max_length:
            print(f"Text length: {len(text)}")
            text = self._chunked_summarize(
                text, max_chunk_size=256, max_summary_size=150
            )
        return text

    def _chunked_summarize(self, text, max_chunk_size, max_summary_size):
        """Summarizes the text in chunks.

        Args:
            text (str): The text to summarize.
            max_chunk_size (int): The maximum size of each chunk.
            max_summary_size (int): The maximum size of each summary.

        Returns:
            str: The summarized text.
        """
        sentences = nltk.sent_tokenize(text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_size + sentence_length > max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_length
            else:
                current_chunk.append(sentence)
                current_size += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        summaries = []
        batch_size = 8
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch_chunks = chunks[i:i + batch_size]
            batch_summaries = self.model(
                batch_chunks,
                max_length=max_summary_size,
                min_length=30,
                do_sample=False,
            )
            summaries.extend(
                [summary["summary_text"] for summary in batch_summaries]
            )

        return " ".join(summaries)
