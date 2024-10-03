# Simplified Papers
This project implements an end-to-end pipeline for extracting and summarizing scientific papers.

![simplified-papers](image.jpg)

## Implementation

### PDF Parsing
The first step is to extract text from the PDF. There are multiple ways to do
 this: algorithmic parsing and OCR. While OCR can handle LaTeX equations,
  existing open-source OCR models like
   [Nougat](https://github.com/facebookresearch/nougat) aren't significantl
   y better than algorithmic parsing and are much slower. Therefore, I am opting
    for algorithmic parsing for now.

### Text Summarization
The best results can be achieved by using a model with a large context window
 like Gemini. However, due to limited compute and VRAM, I am using a smaller
  model. The current approach involves chunked summarization, which continues
   until the summary size is satisfactory.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.