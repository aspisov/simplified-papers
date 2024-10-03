import os
from functools import wraps
import pymupdf4llm
import requests


def download_file_from_url(func):
    """Decorator that downloads a file from a URL and passes the file path to the function.

    Args:
        func: The function to wrap.

    Returns:
        The wrapped function.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        """Wrapper function that handles downloading and cleanup.

        Args:
            url: The URL to download the file from.
            *args: Additional arguments to pass to the wrapped function.
            **kwargs: Additional keyword arguments to pass to the wrapped function.

        Returns:
            The result of the wrapped function.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            temp_file_path = "temp.pdf"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(response.content)
            result = func(temp_file_path, *args, **kwargs)
            return result

        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            return None

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    return wrapper


@download_file_from_url
def parse_pdf_to_markdown(file_path):
    """Parses a PDF file and converts it to markdown text.

    Args:
        file_path: The path to the PDF file.

    Returns:
        The markdown text.
    """
    markdown_text = pymupdf4llm.to_markdown(file_path)
    return markdown_text
