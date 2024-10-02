# simplified-papers
In this project, I am implementing an end-to-end pipeline (potentially service) for extracting and summarizing scientific papers.

## Implementation

### PDF Parsing
The first step is to extract the text from the PDF. There are multiple ways to do this: algorithmic parsing and OCR. The advantage of OCR is that it can handle LaTeX equations. However, existing open-source OCR models like [Nougat](https://github.com/facebookresearch/nougat) aren't significantly better than algorithmic parsing and are much slower. Therefore, I am opting for algorithmic parsing for now.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.