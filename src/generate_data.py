import json
import os
import re

from pdf_parser import parse_pdf_to_txt


class Dataset:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.initialize()

    def initialize(self) -> None:
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as file:
                json.dump([], file)

    def load(self):
        if os.path.getsize(self.filepath) > 0:
            with open(self.filepath, "r") as file:
                return json.load(file)
        else:
            return []

    def save(self, dataset: list) -> None:
        with open(self.filepath, "w") as file:
            json.dump(dataset, file, indent=4)

    def create_new_examples(
        self,
        paper_url: str,
        snapshot: str,
        key_findings: str,
        objectives: str,
        methods: str,
        results: str,
        conclusions: str,
    ) -> list:
        paper_text = parse_pdf_to_txt(paper_url)

        snapshot_example = {
            "instruction": "Provide a brief snapshot (1-2 sentences) of the paper. Summarize the main result or contribution of the study in concise, non-technical language.",
            "context": paper_text,
            "target": snapshot,
        }

        key_findings_example = {
            "instruction": "Summarize the key findings of the study in detail. Include important metrics, comparisons to baseline methods, and any standout results. Mention comparisons to other techniques if provided.",
            "context": paper_text,
            "target": key_findings,
        }

        objectives_example = {
            "instruction": "Summarize the objectives or goals of the study. Focus on what the authors aimed to achieve through their research.",
            "context": paper_text,
            "target": objectives,
        }

        methods_example = {
            "instruction": "Summarize the methods used in the study, including any techniques, models, or datasets employed. Focus on describing the new techniques and their implementation.",
            "context": paper_text,
            "target": methods,
        }

        results_example = {
            "instruction": "Summarize the results of the study. Include key metrics, performance outcomes, and any significant improvements over prior work. Be concise but cover the core results.",
            "context": paper_text,
            "target": results,
        }

        conclusions_example = {
            "instruction": "Summarize the conclusions of the paper, including how the results support the authors' arguments. Mention any broader implications or potential applications.",
            "context": paper_text,
            "target": conclusions,
        }

        full_example = {
            "instruction": f"Summarize the following paper using the structured format: Snapshot, Key Findings, Objectives, Methods, Results, and Conclusions.",
            "context": paper_text,
            "target": {
                "Snapshot": snapshot,
                "Key Findings": key_findings,
                "Objectives": objectives,
                "Methods": methods,
                "Results": results,
                "Conclusions": conclusions,
            },
        }
        return [
            snapshot_example,
            key_findings_example,
            objectives_example,
            methods_example,
            results_example,
            conclusions_example,
            full_example,
        ]

    def add_new_examples(
        self,
        paper_url: str,
        snapshot: str,
        key_findings: str,
        objectives: str,
        methods: str,
        results: str,
        conclusions: str,
    ) -> list:
        dataset = self.load()
        new_examples = self.create_new_examples(
            paper_url,
            snapshot,
            key_findings,
            objectives,
            methods,
            results,
            conclusions,
        )
        dataset.extend(new_examples)
        self.save(dataset)
        return dataset


if __name__ == "__main__":
    url = "https://arxiv.org/pdf/1706.03762"
    snapshot = """## Snapshot
The Transformer model, a sequence transduction model based entirely on attention, achieves state-of-the-art results in machine translation tasks and outperforms traditional recurrent and convolutional layers in terms of computational complexity and parallelization.
"""
    key_findings = """## Key findings
The Transformer model achieves state-of-the-art results in machine translation tasks, outperforming existing models, including ensembles, by over 2 BLEU. It also generalizes well to other tasks, such as English constituency parsing. ↓
In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length $n$ is smaller than the representation dimensionality $d$, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [38] and byte-pair [31] representations
We presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention
The Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers
On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art ↓
"""
    objectives = """## Objectives
The objective of the study is to propose a new simple network architecture, the Transformer, that relies solely on attention mechanisms, dispensing with recurrence and convolutions entirely, and to evaluate its performance on machine translation tasks.
"""
    methods = """## Methods
The Transformer model uses stacked self-attention and point-wise, fully connected layers for both the encoder and decoder. It employs a residual connection around each of the two sub-layers, followed by layer normalization. The model also uses multi-head attention, which allows it to jointly attend to information from different representation subspaces at different positions.
The Transformer model uses multi-headed self-attention layers to connect all positions in the input and output sequences. The model uses sinusoidal positional encoding and is trained using the Adam optimizer with a learning rate schedule. The model is also regularized using dropout, label smoothing, and residual dropout.
"""
    results = """## Results
The Transformer model achieves state-of-the-art results in machine translation tasks, outperforming existing models, including ensembles, by over 2 BLEU. It also generalizes well to other tasks, such as English constituency parsing.
"""
    conclusions = """Conclusions
The Transformer model is a new simple network architecture that relies solely on attention mechanisms, dispensing with recurrence and convolutions entirely, and achieves state-of-the-art results in machine translation tasks.
The Transformer model is a promising approach to sequence transduction tasks, offering improved computational complexity, parallelization, and interpretability. The model achieves state-of-the-art results in machine translation tasks and performs well on English constituency parsing tasks.
"""

    # removing square brackets and their content
    snapshot = re.sub(r"\[.*?\]", "", snapshot)
    key_findings = re.sub(r"\[.*?\]", "", key_findings)
    objectives = re.sub(r"\[.*?\]", "", objectives)
    methods = re.sub(r"\[.*?\]", "", methods)
    results = re.sub(r"\[.*?\]", "", results)
    conclusions = re.sub(r"\[.*?\]", "", conclusions)

    dataset = Dataset("structured_summaries.json")
    dataset.add_new_examples(
        paper_url=url,
        snapshot=snapshot,
        key_findings=key_findings,
        objectives=objectives,
        methods=methods,
        results=results,
        conclusions=conclusions,
    )
