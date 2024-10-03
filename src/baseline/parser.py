import os
import requests
from pypdf import PdfReader


class Parser:
    def __init__(self):
        pass

    @staticmethod
    def parse(url):
        """Downloads and parses a PDF from a given URL.

        Args:
             url (str): The URL of the PDF to download.

        Returns:
             str: The extracted text from the PDF, or an empty string if an error occurs.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            if response.headers.get("content-type") != "application/pdf":
                raise ValueError("URL does not point to a PDF file")
            with open("file.pdf", "wb") as f:
                f.write(response.content)

            text = ""
            reader = PdfReader("file.pdf")
            print(f"Number of pages: {len(reader.pages)}")
            for page in reader.pages:
                text += page.extract_text()

            return text
        except requests.RequestException as e:
            print(f"Failed to download the PDF: {e}")
            return ""
        except Exception as e:
            print(f"An error occurred while reading the PDF: {e}")
            return ""
        finally:
            if os.path.exists("file.pdf"):
                os.remove("file.pdf")
