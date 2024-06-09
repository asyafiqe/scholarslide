# Standard library imports
import asyncio
import base64
import datetime
import io
import json
import os
import re
import requests
import subprocess
import time
import unicodedata
import zipfile
from typing import Dict, List, Optional, Tuple

# Third-party imports
import aiohttp
import fitz
import pandas as pd
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

# Local imports
import pdf_extractor


def get_datetime() -> str:
    # Get the current datetime
    now = datetime.datetime.now()

    # Format the datetime as "yymmdd_HHMMSS"
    formatted_datetime = now.strftime("%y%m%d_%H%M%S")

    return formatted_datetime


class APICallError(Exception):
    def __init__(self, message):
        super().__init__(message)
        print(message)


def ask_llm(
    prompt: str,
    model: str,
    base64_image: str = None,
    api_key: str = None,
    temperature: float = 0,
    max_retries: int = 2,
    expertise_field: str = "biomedical",
    extra_message: dict = None,
    seed: int = 23,
    backoff_factor: int = 2,
):
    """
    Sends a prompt to the OpenRouter API and retrieves a response.

    Args:
        prompt (str): The prompt to send to the API.
        model (str): The model to use for the API call.
        base64_image (str, optional): Base64-encoded image to include in the request. Defaults to None.
        api_key (str, optional): API key for authentication. If not provided, it will be retrieved from the
                                 `OPENROUTER_API_KEY` environment variable.
        temperature (float, optional): Temperature for the API call. Defaults to 0.
        max_retries (int, optional): Maximum retries for failed API calls. Defaults to 2.
        expertise_field (str, optional): Field of expertise for the system message. If not provided, it will be retrieved from the
                                 `EXPERTISE_FIELD` environment variable.
        extra_message (dict, optional): Additional message to include in the API request. Defaults to None.
        seed (int, optional): Random seed for the API call. Defaults to 23.
        backoff_factor (int, optional): Exponential backoff factor for retries. Defaults to 2.

    Returns:
        dict: The response from the OpenRouter API.

    Raises:
        ValueError: If no API key is provided and the `OPENROUTER_API_KEY` environment variable is not set.
        APICallError: If the API call fails after the maximum number of retries.
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key is required. Please provide it as an argument or set the OPENROUTER_API_KEY environment variable."
            )

    if expertise_field is None:
        expertise_field = os.getenv("EXPERTISE_FIELD")
        if expertise_field is None:
            expertise_field = "biomedical"

    retries = 0
    while retries < max_retries:
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"You are an expert in {expertise_field} field tasked to read scientific paper thoroughly.",
                },
                {"role": "user", "content": prompt},
            ]
            if base64_image:
                messages[1]["content"] = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ]
            if extra_message:
                messages.append(extra_message)

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                data=json.dumps(
                    {
                        "model": model,  # Optional
                        "messages": messages,
                        "temperature": temperature,
                        "seed": seed,
                    }
                ),
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error making API call: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying API call ({retries}/{max_retries})")
                time.sleep(backoff_factor**retries)  # Exponential backoff
            else:
                error_message = "Failed to get a response from the API."
                raise APICallError(error_message)


async def ask_llm_async(
    prompt: str,
    model: str,
    base64_image: str = None,
    api_key: str = None,
    temperature: float = 0,
    max_retries: int = 2,
    expertise_field: str = None,
    extra_message: Optional[Dict] = None,
    seed: int = 23,
    backoff_factor: int = 2,
):
    """
    Send a prompt to the OpenRouter API asynchronously and get a response.

    Args:
        prompt (str): The prompt to send to the API.
        model (str): The model to use for the API call.
        base64_image (str, optional): The base64-encoded image to include in the API request. Defaults to None.
        api_key (str, optional): API key for authentication. If not provided, it will be retrieved from the
                                 `OPENROUTER_API_KEY` environment variable.
        temperature (float, optional): The temperature to use for the API call. Defaults to 0.
        max_retries (int, optional): The maximum number of retries to attempt if the API call fails. Defaults to 2.
        expertise_field (str, optional): Field of expertise for the system message. If not provided, it will be retrieved from the
                                 `EXPERTISE_FIELD` environment variable.
        extra_message (dict, optional): An additional message to include in the API request. Defaults to None.
        seed (int, optional): Random seed for the API call. Defaults to 23.
        backoff_factor (int, optional): How long to wait after a failed API call before making next API call.

    Returns:
        dict: The response from the API.

    Raises:
        APICallError: If the API call fails after the maximum number of retries.

    Examples:
    # For the original ask_llm function
    response = await ask_llm_async(prompt, model)

    # For the original ask_claude_json function
    extra_message = {"role": "assistant", "content": "{"}
    response = await ask_llm_async(prompt, model, extra_message=extra_message)

    # Image-based prompt/vision
    base64_image = "..."  # Replace with your base64-encoded image
    response = await ask_llm_async(prompt="Describe the key information in this figure.", model="openai/gpt-4o", base64_image=base64_image)

    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None:
        raise ValueError(
            "API key is required. Please provide it as an argument or set the OPENROUTER_API_KEY environment variable."
        )

    if expertise_field is None:
        expertise_field = os.getenv("EXPERTISE_FIELD")
        if expertise_field is None:
            expertise_field = "biomedical"

    retries = 0
    while retries < max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                messages = [
                    {
                        "role": "system",
                        "content": f"You are an expert in {expertise_field} field tasked to read scientific paper thoroughly.",
                    },
                    {"role": "user", "content": prompt},
                ]
                if base64_image:
                    messages[1]["content"] = [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ]
                if extra_message:
                    messages.append(extra_message)

                async with session.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,  # Optional
                        "messages": messages,
                        "temperature": temperature,
                        "seed": seed,
                    },
                ) as response:
                    response.raise_for_status()
                    return await response.json()

        except aiohttp.ClientError as e:
            print(f"Error making API call: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying API call ({retries}/{max_retries})")
                await asyncio.sleep(backoff_factor**retries)  # Exponential backoff
            else:
                error_message = "Failed to get a response from the API."
                raise APICallError(error_message)


def extract_response(response_data: dict):
    content = response_data["choices"][0]["message"]["content"]

    return content


def extract_response_json(response_data: dict):
    content = response_data["choices"][0]["message"]["content"]
    content = "{" + content
    content = json.loads(content)

    return content


def handle_response(responses: list, response_type: str) -> list:
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            print(f"Error generating {response_type} for row {i}: {response}")
            responses[i] = " "
        elif response is None:
            responses[i] = " "
        else:
            responses[i] = response
    return responses


async def ask_llm_and_extract_async(
    prompt: str, model: str, base64_image: str = None, seed: int = 23
) -> Optional[str]:
    try:
        response_data = await ask_llm_async(
            prompt, model=model, base64_image=base64_image, seed=seed
        )
        return extract_response(response_data)
    except APICallError as e:
        print(e)
        return None


def run_pdf_extraction(
    input_file: str, output_file: str, output_extracted_dir: str, max_retries: int = 3
):
    """
    Extracts the content from a PDF file and saves the extracted text, tables, and figures to files in output_extracted_dir.

    This function uses the Adobe PDF Services API to extract content from a PDF file. It retries the API call up to
    `max_retries` times if an error occurs.

    Parameters:
    input_file (str): The path to the input PDF file.
    output_file (str): The path to the output file where the extracted content zip will be saved.
    output_extracted_dir (str): The directory where the extracted files from the PDF will be saved.
    max_retries (int, optional): The maximum number of retries for the API call. Defaults to 3.

    Returns:
    None

    Raises:
    APICallError: If the API call fails after `max_retries` attempts.

    """
    retries = 0
    while retries < max_retries:
        try:
            pdf_extractor.extract_pdf(input_file, output_file)
            break  # Exit the loop if successful
        except Exception as e:
            print(f"Error making pdf extraction API call: {e}")
            retries += 1
            print(f"Retrying pdf extraction API call ({retries}/{max_retries})")

    if retries == max_retries:
        error_message = "Failed to get a response from the pdf extraction API."
        raise APICallError(error_message)

    with zipfile.ZipFile(output_file, "r") as zip_ref:
        # Extract all files to a specific directory
        zip_ref.extractall(output_extracted_dir)


def extract_pdf_text(input_file: str) -> Tuple[list, str]:
    """
    Extracts the text content from a PDF file.

    Parameters:
    input_file (str): The path to the input PDF file.

    Returns:
    tuple: A tuple containing two elements:
        - cleaned_texts (list): A list of cleaned text strings, where each string represents the text content of a page in the PDF.
        - cleaned_text (str): A single string containing the concatenated and cleaned text content of all pages in the PDF, separated by two newline characters.

    Raises:
    None

    """
    texts = []
    with fitz.open(input_file) as doc:
        for page in doc:
            page_text = page.get_text()
            texts.append(page_text)

    cleaned_texts = [unicodedata.normalize("NFKD", txt) for txt in texts]
    cleaned_text = "\n\n".join(cleaned_texts)

    return cleaned_texts, cleaned_text


async def classify_page_async(texts: list, model: str) -> pd.DataFrame:
    """
    Classifies the sections of a scientific paper page.

    This function takes a list of text passages, where each passage represents the content of a single page of a scientific paper. It then sends each passage (excluding the first page, which is assumed to be the cover) to the specified AI model to classify the sections of the page.

    The function uses a predefined prompt to guide the AI model in identifying the sections of the page, such as Background, Results, Method, Discussion, and Conclusion. The response from the AI model is then extracted and stored in a pandas DataFrame, with each row representing a page and the corresponding classification.

    Parameters:
    texts (list): A list of text passages, where each passage represents the content of a single page of a scientific paper.
    model (str): The name of the AI model to use for the classification task.

    Returns:
    pandas.DataFrame: A DataFrame containing the page number and the raw classification result for each page.
    """
    # first pass
    tasks = []
    for i, txt in enumerate(texts):
        # exclude first page (cover)
        if i != 0:
            prompt = "The following passage is from one page of a scientific paper.\n"
            prompt += "The page may contain Background, Results, Method, Discussion, and/or Conclusion. Usually only one or two of the above. In some case, three might be possible.\n"
            # prompt += 'Passage can have more than one of above.\n'
            prompt += "Which of the section (Background, Results, Method, Discussion, and/or Conclusion) does this page contain?\n\n"
            prompt += "Answer in markdown format.\n"
            prompt += "For example, if the page contain discussion and conclusion, write ## DISCUSSION\n ## CONCLUSION \n"
            # prompt += 'I will give you the title and abstract of the paper and then the passage to check.\n'
            # prompt += 'Give short answer. Only answer Background, Results, Method, Discussion, Conclusion, and/or References.\n'
            # prompt += 'Do not give additional information\n'
            # prompt += 'Answer in JSON format with keys "background", "results", "method", "discussion", "conclusion".\n'
            # prompt += 'Value is boolean.\n'
            # prompt += abstract + '\n'
            prompt += "Here is the passage from one page of a paper:\n\n"
            prompt += txt
            tasks.append(ask_llm_and_extract_async(prompt, model))

    # Use asyncio.gather with 'return_exceptions=True' to handle potential errors
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    clf_raw = handle_response(responses, "classification")

    page_clf_df = pd.DataFrame(
        {"page": [i + 1 for i in range(len(clf_raw))], "clf_raw": clf_raw}
    )

    # Parse page classification output
    sections = ["background", "result", "discussion", "conclusion", "method"]

    for section in sections:
        page_clf_df[section] = page_clf_df["clf_raw"].str.contains(
            f"## {section}", case=False
        )

    # Detect page edge/boundary
    # True if on the page, one page after, one page before
    # bd = bidirectional
    for section in sections:
        page_clf_df[f"{section}_bd"] = (
            page_clf_df[section]
            | page_clf_df[section].shift(1)
            | page_clf_df[section].shift(-1)
        )

    return page_clf_df


def extract_title_abstract(texts: list, model: str) -> Tuple[str, str]:
    """
    Extract the title and abstract of a scientific paper from a given text.

    Args:
        texts (list): A list of strings, where the first element is the main text of the paper and the second element is the abstract.
        model (str): The name of the language model to use for the extraction.

    Returns:
        title (str): The extracted title of the paper.
        title_abstract (str): The full title and abstract in markdown format.
    """

    prompt = "Extract title and abstract of this scientific paper. Put into markdown format. Text:\n"
    prompt += texts[0] + "\n\n" + texts[1]
    response = ask_llm(prompt, model=model)
    title_abstract = extract_response(response)

    # Extract title
    prompt = "Give me the title of this paper. Give response in JSON format with key 'title'. \n"
    prompt += title_abstract
    extra_message = {"role": "assistant", "content": "{"}
    response = ask_llm(prompt, model, extra_message=extra_message)
    extracted_response = extract_response_json(response)
    title = extracted_response["title"]

    return title, title_abstract


def resize_image(image_path: str, max_size: int = 512) -> Image.Image:
    """
    Resizes an image while maintaining the aspect ratio, with a maximum width or height of the specified size.
    If the original image size is already smaller than the maximum size, the original image is returned.

    Args:
        image_path (str): The file path of the image to be resized.
        max_size (int): The maximum width or height of the resized image (default is 512).

    Returns:
        PIL.Image: The resized image.
    """
    # Open the image
    image = Image.open(image_path)

    # Get the original size of the image
    original_width, original_height = image.size

    # Check if the original image size is already smaller than the maximum size
    if original_width <= max_size and original_height <= max_size:
        return image

    # Calculate the new size while maintaining the aspect ratio
    if original_width > original_height:
        new_width = max_size
        new_height = int(original_height * (max_size / original_width))
    else:
        new_height = max_size
        new_width = int(original_width * (max_size / original_height))

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    return resized_image


def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image file into a base64 string.

    Args:
      image_path: The path to the image file.

    Returns:
      A base64 encoded string representing the image data.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def convert_pil_image_to_base64(image: Image.Image) -> str:
    """
    Converts a PIL Image object to a base64 encoded string.

    Args:
        image (PIL.Image): The image to be converted.

    Returns:
        str: The image as a base64 encoded string.
    """
    # Convert the image to bytes
    byte_io = io.BytesIO()
    image.save(byte_io, format="JPEG")
    image_bytes = byte_io.getvalue()

    # Encode the bytes to a base64 string
    base64_string = base64.b64encode(image_bytes).decode("utf-8")

    return base64_string


def get_figtables_info(extracted_zip_dir: str) -> dict:
    """
    Extracts information about figures and tables from a JSON file in the extracted ZIP directory.

    Args:
        extracted_zip_dir (str): The path to the directory containing the extracted ZIP file.

    Returns:
        dict: A dictionary where the keys are the full paths to the figure/table PNG files, and the values are the corresponding page numbers.
    """
    json_file = os.path.join(extracted_zip_dir, "structuredData.json")

    # Reading from a JSON file
    with open(json_file, "r") as file:
        json_data = json.load(file)

    # put figure and table path with page info to a dict
    figtables_info = {}

    for element in json_data["elements"]:
        file_paths = element.get("filePaths")
        if file_paths is not None:
            for file_path in file_paths:
                if "png" in file_path:
                    fullpath = os.path.join(extracted_zip_dir, file_path)
                    page = element["Page"]
                    figtables_info[fullpath] = page

    return figtables_info


async def classify_figtables_async(
    figtables: dict, model: str = "google/gemini-flash-1.5"
) -> dict[str, str]:
    """
    Classifies the figures and tables in the provided dictionary as being part of a scientific paper or not.

    Args:
        figtables (dict): A dictionary where the keys are the full paths to the figure/table PNG files, and the values are the corresponding page numbers.
        model (str, optional): The name of the language model to use for the classification. Defaults to 'google/gemini-flash-1.5'.

    Returns:
        dict: A dictionary where the keys are the full paths to the figure/table PNG files, and the values are the classification results ('yes' or 'no').
    """
    prompt = "Is this picture part of a scientific paper figure? Answer yes or no."

    tasks = []
    for path in figtables.keys():
        resize_imaged = resize_image(path)
        base64_image = convert_pil_image_to_base64(resize_imaged)
        tasks.append(
            ask_llm_and_extract_async(prompt, model=model, base64_image=base64_image)
        )

    # Use asyncio.gather with 'return_exceptions=True' to handle potential errors
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    classifications = handle_response(responses, "classification")
    figtables_classification = {
        key: classifications[index] for index, key in enumerate(figtables)
    }

    return figtables_classification


def merge_figtables_info_classification(
    figtables_info: dict, figtables_classification: dict
) -> pd.DataFrame:
    """
    Merges and processes information from two dictionaries to create a pandas DataFrame.

    Args:
        figtables_info (dict): A dictionary containing information about figures and tables, where the keys are file paths
            and the values are page numbers.
        figtables_classification (dict): A dictionary containing classification information about figures and tables, where
            the keys are file paths and the values are classification labels (e.g., 'yes' or 'no').

    Returns:
        pd.DataFrame: A pandas DataFrame containing the merged and processed information, including the file path, page
            number, classification, type (figure or table), and figure/table number.
    """

    # make a dataframe from figtables_info
    figtables_df = pd.DataFrame.from_dict(figtables_info, orient="index")
    figtables_df.reset_index(inplace=True)
    figtables_df.columns = ["path", "page"]

    # make a dataframe from figtables_classification
    figtables_classification_df = pd.DataFrame.from_dict(
        figtables_classification, orient="index"
    )
    figtables_classification_df.reset_index(inplace=True)
    figtables_classification_df.columns = ["path", "classification_raw"]

    # merge both dataframes
    figtables_df = pd.merge(figtables_df, figtables_classification_df, how="inner")

    # create a 'classification' column parsing 'yes' from 'classification_raw' column
    figtables_df["classification"] = (
        figtables_df["classification_raw"].str.lower().str.contains("yes")
    )

    # get type of image (figures or tables)
    figtables_df["type"] = figtables_df["path"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )

    # create figtables numbering
    figtables_df["figure"] = (figtables_df["type"] == "figures") & (
        figtables_df["classification"] == True
    )

    figtables_df["figure_num"] = np.where(
        figtables_df["figure"], figtables_df["figure"].cumsum(), 0
    )

    figtables_df["table"] = (figtables_df["type"] == "tables") & (
        figtables_df["classification"] == True
    )
    figtables_df["table_num"] = np.where(
        figtables_df["table"], figtables_df["table"].cumsum(), 0
    )

    return figtables_df


def get_caption_text(caption_page: int, cleaned_texts: list) -> str:
    return "\n\n".join(
        [
            cleaned_texts[caption_page - 1],
            cleaned_texts[caption_page],
            cleaned_texts[caption_page + 1],
        ]
    )


def get_caption_prompt(figure_num: int, caption_text: str, model: str) -> str:
    if model.startswith("google"):
        prompt = f"Find caption of Figure {figure_num} in the following text. \n\n"
    else:
        prompt = f"Find figure legend/caption of Figure {figure_num} in the following text. \n\n"
    prompt += caption_text
    return prompt


def get_classification_prompt(caption: str) -> str:
    return (
        f"Is this a figure legend/caption of a scientific paper?\n\nText:\n {caption}"
    )


def ask_llm_and_extract(prompt: str, model: str, seed: int = 23) -> Optional[str]:
    """Asks an LLM a question and extracts the response.

    Args:
        prompt (str): The question to ask the LLM.
        model (str): The name of the LLM model to use.
        seed (int, optional): The random seed to use for the LLM. Defaults to 23.

    Returns:
        Optional[str]: The extracted response from the LLM, or None if there was an error.
    """
    try:
        response_data = ask_llm(prompt, model=model, seed=seed)
        return extract_response(response_data)
    except APICallError as e:
        print(e)
        return None


def correct_responses_index(
    responses: list, original_length: int, skip_indices: list
) -> list:
    """Corrects the indices of responses after skipping some indices.

    Args:
        responses (list): The list of responses to be re-indexed.
        original_length (int): The original length of the list before skipping indices.
        skip_indices (list): A set of indices that were skipped.

    Returns:
        list: A new list with the responses re-indexed to match the original length,
              taking into account the skipped indices.
    """
    # Create a list to store the results
    responses_index_corrected = [None] * original_length

    # Populate the results list with the correct indices
    index = 0
    for i, result in enumerate(responses):
        while index in skip_indices:
            index += 1
        responses_index_corrected[index] = result
        index += 1
    return responses_index_corrected


async def get_caption_async(
    figtables_df: pd.DataFrame,
    cleaned_texts: list,
    model_main: str = "google/gemini-pro",
    model_fallback: str = "anthropic/claude-3-haiku",
):
    """Generates captions asynchronously for figures in a DataFrame.

    This function iterates through a DataFrame containing figure information and generates captions for each figure using
    a large language model (LLM). It uses a two-step process:

    1. **First Pass:** Generates captions using the primary LLM (`model_main`) and checks the quality of the generated
       captions using a secondary LLM (`model_fallback`).
    2. **Second Pass:** If the initial caption is not classified as a valid caption, it generates a new caption using
       the primary LLM with a fixed random seed.
    3. **Third Pass:** If the second pass still fails to generate a valid caption, it uses the secondary LLM to generate
       a caption.
    4. **Post-processing:** Clean output to only get the caption without any  preamble.

    Args:
        figtables_df (pd.DataFrame): DataFrame containing figure information, including page number, figure number, and
            existing captions (if any).
        cleaned_texts (list): List of cleaned text from the document.
        model_main (str, optional): Name of the primary LLM to use for caption generation. Defaults to
            'google/gemini-pro'.
        model_fallback (str, optional): Name of the secondary LLM to use for caption classification and fallback
            caption generation. Defaults to 'anthropic/claude-3-haiku'.

    Returns:
        pd.DataFrame: The input DataFrame with generated captions added to the 'caption' column and caption classification
            results in the 'caption_classification' and 'caption_clf' columns.
    """
    # first pass
    tasks = []
    skip_indices = []
    for index, row in figtables_df.iterrows():
        if row["figure"]:
            caption_text = get_caption_text(int(row["page"]), cleaned_texts)
            prompt = get_caption_prompt(
                int(row["figure_num"]), caption_text, model_main
            )
            tasks.append(ask_llm_and_extract_async(prompt, model_main))
        else:
            # If 'figure' is False, append None to the tasks list
            skip_indices.append(index)

    # Use asyncio.gather with 'return_exceptions=True' to handle potential errors
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Correct indices of response. Accounting for skipped indices.
    responses_index_corrected = correct_responses_index(
        responses, len(figtables_df), skip_indices
    )

    # Handle errors and None values
    figtables_df["caption"] = handle_response(responses_index_corrected, "caption")

    # check caption result
    tasks = []
    skip_indices = []
    for index, row in figtables_df.iterrows():
        if row["figure"]:
            prompt = get_classification_prompt(row["caption"])
            tasks.append(ask_llm_and_extract_async(prompt, model_main))
        else:
            # If 'figure' is False, append None to the tasks list
            skip_indices.append(index)

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Correct indices of response. Accounting for skipped indices.
    responses_index_corrected = correct_responses_index(
        responses, len(figtables_df), skip_indices
    )

    # Handle errors and None values
    figtables_df["caption_classification"] = handle_response(
        responses_index_corrected, "caption_classification"
    )

    # parse check caption response (this output is better than directly asking for json format)
    figtables_df["caption_clf"] = (
        figtables_df["caption_classification"].str.lower().str.contains("yes")
    )

    # second pass
    tasks = []
    skip_indices = []
    for index, row in figtables_df.iterrows():
        if row["figure"] and not row["caption_clf"]:
            caption_text = get_caption_text(int(row["page"]), cleaned_texts)
            prompt = get_caption_prompt(
                int(row["figure_num"]), caption_text, model_main
            )
            tasks.append(ask_llm_and_extract_async(prompt, model_main, seed=34))
        else:
            # If 'figure' is False, append None to the tasks list
            skip_indices.append(index)

    # Use asyncio.gather with 'return_exceptions=True' to handle potential errors
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Correct indices of response. Accounting for skipped indices.
    responses_index_corrected = correct_responses_index(
        responses, len(figtables_df), skip_indices
    )

    # Handle errors and None values
    figtables_df["caption"] = handle_response(responses_index_corrected, "caption")

    # check caption result
    tasks = []
    skip_indices = []
    for index, row in figtables_df.iterrows():
        if row["figure"]:
            prompt = get_classification_prompt(row["caption"])
            tasks.append(ask_llm_and_extract_async(prompt, model_main))
        else:
            # If 'figure' is False, append None to the tasks list
            skip_indices.append(index)

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Correct indices of response. Accounting for skipped indices.
    responses_index_corrected = correct_responses_index(
        responses, len(figtables_df), skip_indices
    )

    # Handle errors and None values
    figtables_df["caption_classification"] = handle_response(
        responses_index_corrected, "caption_classification"
    )

    # parse check caption response (this output is better than directly asking for json format)
    figtables_df["caption_clf"] = (
        figtables_df["caption_classification"].str.lower().str.contains("yes")
    )

    # third pass
    tasks = []
    skip_indices = []
    for index, row in figtables_df.iterrows():
        if row["figure"] and not row["caption_clf"]:
            caption_text = get_caption_text(int(row["page"]), cleaned_texts)
            prompt = get_caption_prompt(
                int(row["figure_num"]), caption_text, model_main
            )
            tasks.append(ask_llm_and_extract_async(prompt, model_fallback))
        else:
            # If 'figure' is False, append None to the tasks list
            skip_indices.append(index)

    # Use asyncio.gather with 'return_exceptions=True' to handle potential errors
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Correct indices of response. Accounting for skipped indices.
    responses_index_corrected = correct_responses_index(
        responses, len(figtables_df), skip_indices
    )

    # Handle errors and None values
    figtables_df["caption"] = handle_response(responses_index_corrected, "caption")

    # post-processing
    # remove unnecessary text
    tasks = []
    skip_indices = []
    for index, row in figtables_df.iterrows():
        # json mode is better, but every model has its own format. Use prompt instead.
        if row["figure"]:
            prompt = "Following is a passage containing figure legend/caption with some intro/preamble/commentary from a scientific paper.\n"
            prompt += "Give me the figure legend/caption only, remove intro/preamble/commentary.\n"
            prompt += "Do not add explanation or anything. Just the figure legend/caption. Figure legend/caption only.\n"
            prompt += "Do not add formatting like # or ##.\n"
            prompt += "Following is the passage:\n\n"
            prompt += row["caption"]
            tasks.append(ask_llm_and_extract_async(prompt, model_main))
        else:
            # If 'figure' is False, append None to the tasks list
            skip_indices.append(index)

    # Use asyncio.gather with 'return_exceptions=True' to handle potential errors
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Correct indices of response. Accounting for skipped indices.
    responses_index_corrected = correct_responses_index(
        responses, len(figtables_df), skip_indices
    )

    # Handle errors and None values
    figtables_df["caption"] = handle_response(responses_index_corrected, "caption")

    return figtables_df


async def get_table_titles(
    figtables_df: pd.DataFrame, cleaned_texts: list, model: str
) -> pd.DataFrame:
    """Extracts table titles from a DataFrame containing table information and cleaned text.

    This function iterates through a DataFrame containing information about tables
    in a document, extracts the relevant text from the cleaned text based on the
    table's page number, and uses a language model to extract the table title.

    Args:
      figtables_df: A Pandas DataFrame containing information about tables in a
        document. It should include columns like 'page', 'table_num', and 'table'.
      cleaned_texts: A list of strings representing the cleaned text of each page in
        the document.
      model: A language model instance.

    Returns:
      A Pandas DataFrame with an additional column 'table_title_cleaned' containing
      the extracted table titles.
    """
    tasks = []
    skip_indices = []
    for index, row in figtables_df.iterrows():
        if row["table"]:
            caption_text = get_caption_text(int(row["page"]), cleaned_texts)
            prompt = f"Find the title of Table {row['table_num']} in the following text. \n\n"
            prompt += caption_text
            tasks.append(ask_llm_and_extract_async(prompt, model))
        else:
            # If 'figure' is False, append None to the tasks list
            skip_indices.append(index)

    # Use asyncio.gather with 'return_exceptions=True' to handle potential errors
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Correct indices of response. Accounting for skipped indices.
    responses_index_corrected = correct_responses_index(
        responses, len(figtables_df), skip_indices
    )

    # Handle errors and None values
    figtables_df["table_title"] = handle_response(
        responses_index_corrected, "table_title"
    )

    # Clean responses, extract only table title
    tasks = []
    skip_indices = []
    for index, row in figtables_df.iterrows():
        if row["table"]:
            prompt = "Following is a passage containing table title and some explanation of a table in a scientific paper.\n"
            prompt += "Give the title of the table.\n"
            prompt += (
                "Do not add explanation or anything. Just the title. Title only.\n"
            )
            prompt += "Do not add formatting like # or ##.\n"
            prompt += "Following is the passage:\n\n"
            prompt += row["table_title"]
            tasks.append(ask_llm_and_extract_async(prompt, model))
        else:
            # If 'figure' is False, append None to the tasks list
            skip_indices.append(index)

    # Use asyncio.gather with 'return_exceptions=True' to handle potential errors
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Correct indices of response. Accounting for skipped indices.
    responses_index_corrected = correct_responses_index(
        responses, len(figtables_df), skip_indices
    )

    # Handle errors and None values
    figtables_df["table_title_cleaned"] = handle_response(
        responses_index_corrected, "table_title_cleaned"
    )

    return figtables_df


async def summarize_section(
    section: str,
    page_clf_df: pd.DataFrame,
    cleaned_texts: list,
    title_abstract: str,
    model: str,
) -> tuple:
    """Summarizes a specific section of a scientific paper using a large language model.

    This function extracts the text from the specified section of the paper,
    builds a prompt for a large language model, and generates a summary of the section
    in Markdown format, suitable for a PowerPoint slide.

    Args:
      section: The name of the section to summarize (e.g., 'background', 'results').
      page_clf_df: A pandas DataFrame containing information about page numbers and section boundaries.
      cleaned_texts: A list of cleaned text from each page of the paper.
      title_abstract: The title and abstract of the paper.
      model: The name of the large language model to use for summarization.

    Returns:
      A tuple containing the section name and the markdown summary of the section,
      or (None, None) if an API call error occurs.
    """

    section_texts = []
    if section == "background":
        # cover page was skipped during page classification, may contain background
        section_texts.append(cleaned_texts[0])

    # get text from corresponding page
    for _, row in page_clf_df.iterrows():
        if row[f"{section}_bd"]:
            section_texts.append(cleaned_texts[row["page"]])
    section_text = "\n\n".join(section_texts)

    # build prompt
    prompt = "You are tasked to make a powerpoint slide of a scientific paper:\n\n"
    prompt += f"Create several slides from the {section} section of the paper.\n"
    prompt += f"Create slides only for the {section} section of the paper."
    prompt += "Even if you are given other sections of the paper, do not add the other sections to the slides."
    prompt += (
        "Give response in markdown format. For example, slide title is # Slide title\n"
    )
    prompt += "Each # represents a new slide.\n"
    prompt += "Make it in bullet points.\n"
    if section == "background":
        prompt += "The last slide is the key question of the study.\n\n"
    prompt += "Do not add anything else in your response unless it will be included in the powerpoint slides.\n\n"
    prompt += "Here is the paper:\n"
    prompt += "Title and abstract:\n"
    prompt += title_abstract
    prompt += "\n\nText:\n"
    prompt += section_text

    try:
        # response_data = await ask_llm_async(prompt, model=model)
        # summary = extract_response(response_data)
        summary = await ask_llm_and_extract_async(prompt, model=model)
        return section, summary
    except APICallError as e:
        print(e)
        return None, None


async def summarize_paper(
    page_clf_df: pd.DataFrame,
    cleaned_texts: list,
    title_abstract: str,
    model: str = "anthropic/claude-3-sonnet",
) -> Dict[str, str]:
    """Summarizes a scientific paper into sections asynchronously.

    This function uses a large language model (LLM) to generate summaries of different sections of a scientific paper.
    It first identifies the sections of the paper based on a predefined list of section names.
    Then, for each section, it calls the `summarize_section` function asynchronously to generate a summary.
    The summaries are then stored in a dictionary, with the section names as keys.

    Args:
        page_clf_df (pd.DataFrame): A pandas DataFrame containing information about page numbers and section boundaries.
        cleaned_texts (list): A list of strings representing the cleaned text of the paper.
        title_abstract (str): The title and abstract of the paper.
        model (str, optional): The name of the LLM to use for summarization. Defaults to 'anthropic/claude-3-sonnet'.

    Returns:
        dict: A dictionary containing the summaries of each section of the paper.
    """
    sections = ["background", "result", "discussion", "conclusion", "method"]

    summaries = {}
    tasks = [
        summarize_section(section, page_clf_df, cleaned_texts, title_abstract, model)
        for section in sections
    ]
    for task in asyncio.as_completed(tasks):
        section, summary = await task
        summaries[section] = summary

    return summaries


def merge_figtables_df(
    figtables_caption_df: pd.DataFrame, figtables_table_title_df: pd.DataFrame
) -> pd.DataFrame:
    """Merges two DataFrames containing figure and table captions and titles.

    This function merges two DataFrames, `figtables_caption_df` and `figtables_table_title_df`,
    which contain captions and table titles respectively. It replaces empty captions in
    `figtables_caption_df` with the corresponding table titles from `figtables_table_title_df`.

    Args:
        figtables_caption_df (pd.DataFrame): DataFrame containing figure and table captions.
        figtables_table_title_df (pd.DataFrame): DataFrame containing table titles.

    Returns:
        pd.DataFrame: The merged DataFrame with captions replaced by table titles where applicable.

    Note:
        This function does not use `pd.merge` due to the presence of similarly named columns.
        It directly modifies the 'caption' column in `figtables_caption_df`.
    """
    # many similarly named column, better to not pd.merge
    figtables_caption_df["caption"] = figtables_caption_df["caption"].where(
        figtables_caption_df["caption"] != " ",
        figtables_table_title_df["table_title_cleaned"],
    )

    return figtables_caption_df


async def get_figtables_caption_table_titles(
    figtables_df: pd.DataFrame,
    cleaned_texts: list,
    model_main: str = "google/gemini-pro",
    model_fallback: str = "anthropic/claude-3-haiku",
) -> pd.DataFrame:
    """Asnchronously extracts captions and table titles from a DataFrame containing figure/table information.

    This function runs two tasks concurrently: extracting captions and extracting table titles.
    It utilizes two language models, `model_main` for the primary task and `model_fallback`
    for a fallback option. The results are then merged into a single DataFrame.

    Args:
      figtables_df: A Pandas DataFrame containing information about figures and tables in a
        document. It should include columns like 'page', 'table_num', 'figure', and 'table'.
      cleaned_texts: A list of strings representing the cleaned text of each page in
        the document.
      model_main: The name of the primary language model to use. Defaults to
        'google/gemini-pro'.
      model_fallback: The name of the fallback language model to use. Defaults to
        'anthropic/claude-3-haiku'.

    Returns:
      A Pandas DataFrame with additional columns 'caption' and 'table_title_cleaned'
      containing the extracted captions and table titles.
    """

    task1 = asyncio.create_task(
        get_caption_async(
            figtables_df,
            cleaned_texts,
            model_main=model_main,
            model_fallback=model_fallback,
        )
    )
    task2 = asyncio.create_task(
        get_table_titles(figtables_df, cleaned_texts, model=model_main)
    )

    figtables_caption_df, figtables_table_title_df = await asyncio.gather(task1, task2)

    # merge two dataframes
    figtables_caption_df = merge_figtables_df(
        figtables_caption_df, figtables_table_title_df
    )
    return figtables_caption_df


async def read_figures(
    figtables_df: pd.DataFrame, title_abstract: str, model: str
) -> pd.DataFrame:
    """Reads figures from a DataFrame and generates explanations using a language model.

    This function iterates through a DataFrame containing figure information and uses a language model
    to generate explanations for each figure. It handles potential errors and missing figures.

    Args:
        figtables_df (pd.DataFrame): DataFrame containing figure information, including paths, captions,
                                    and a boolean column 'figure' indicating whether a figure exists.
        title_abstract (str): Title and abstract of the paper containing the figures.
        model (str): Name of the language model to use for generating explanations.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'fig_exp' containing the
                      figure explanations generated by the language model.

    Raises:
        None
    """
    tasks = []
    skip_indices = []
    for index, row in figtables_df.iterrows():
        if row["figure"]:
            prompt = "Explain the figure.\n"
            prompt += (
                "Following is the title and abstract of the paper of the figure.\n"
                + title_abstract
                + "\n\n"
            )
            prompt += "Following is a possible figure legend taken from the paper.\n"
            prompt += row["caption"]
            figure = encode_image_to_base64(row["path"])
            tasks.append(
                ask_llm_and_extract_async(prompt, model=model, base64_image=figure)
            )
        else:
            # If 'figure' is False, append None to the tasks list
            skip_indices.append(index)

    # Use asyncio.gather with 'return_exceptions=True' to handle potential errors
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Correct indices of response. Accounting for skipped indices.
    responses_index_corrected = correct_responses_index(
        responses, len(figtables_df), skip_indices
    )

    # Handle errors and None values
    figtables_df["fig_exp"] = handle_response(
        responses_index_corrected, "figure explanation"
    )

    return figtables_df


async def read_tables(
    figtables_df: pd.DataFrame, title_abstract: str, model: str
) -> pd.DataFrame:
    """Reads tables from a DataFrame and generates explanations using a language model.

    This function iterates through a DataFrame containing information about tables
    and uses a language model to generate explanations for each table. It handles
    potential errors and returns a DataFrame with the generated explanations.

    Args:
        figtables_df (pd.DataFrame): A DataFrame containing information about tables,
            including paths to table images, captions, and a boolean indicating
            whether the row represents a table.
        title_abstract (str): The title and abstract of the paper containing the tables.
        model (str): The name of the language model to use for generating explanations.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'tbl_exp' containing
            the generated explanations for each table.

    Raises:
        None

    """
    tasks = []
    skip_indices = []
    for index, row in figtables_df.iterrows():
        if row["table"]:
            prompt = "Explain the table.\n"
            prompt += (
                "Following is the title and abstract of the paper of the table.\n"
                + title_abstract
                + "\n\n"
            )
            prompt += "Following is a possible table title taken from the paper.\n"
            prompt += row["caption"]
            table = encode_image_to_base64(row["path"])
            tasks.append(
                ask_llm_and_extract_async(prompt, model=model, base64_image=table)
            )
        else:
            # If 'figure' is False, append None to the tasks list
            skip_indices.append(index)

    # Use asyncio.gather with 'return_exceptions=True' to handle potential errors
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Correct indices of response. Accounting for skipped indices.
    responses_index_corrected = correct_responses_index(
        responses, len(figtables_df), skip_indices
    )

    # Handle errors and None values
    figtables_df["tbl_exp"] = handle_response(
        responses_index_corrected, "table explanation"
    )

    return figtables_df


def merge_figtables_exp_df(
    fig_exp_df: pd.DataFrame, tbl_exp_df: pd.DataFrame
) -> pd.DataFrame:
    """Merges figure and table explanations into a single DataFrame.

    This function takes two DataFrames, one containing figure explanations (`fig_exp_df`)
    and the other containing table explanations (`tbl_exp_df`), and merges them based on
    the presence of explanations in the figure DataFrame. If a figure explanation is empty
    (' '), it is replaced with the corresponding table explanation.

    Args:
      fig_exp_df: DataFrame containing figure explanations.
      tbl_exp_df: DataFrame containing table explanations.

    Returns:
      A DataFrame with merged figure and table explanations.
    """
    fig_exp_df["fig_exp"] = fig_exp_df["fig_exp"].where(
        fig_exp_df["fig_exp"] != " ", tbl_exp_df["tbl_exp"]
    )

    return fig_exp_df


async def get_figure_titles(figtables_exp_df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Generates figure titles from figure explanations using a language model.

    This function iterates through a DataFrame containing figure explanations and uses a language model
    to generate concise titles for each figure. The titles are designed to be suitable for use as slide
    titles in a presentation explaining the paper.

    Args:
        figtables_exp_df (pd.DataFrame): A DataFrame containing figure explanations.
            It should have a column named 'fig_exp' containing the figure explanations and a column
            named 'figure' indicating whether the row corresponds to a figure.
        model (str): The name of the language model to use for title generation.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'figure_title' containing the
            generated figure titles.

    Raises:
        None

    Notes:
        - The function uses asyncio.gather to efficiently handle multiple asynchronous requests to the
          language model.
        - The function handles potential errors and None values in the responses from the language model.
        - The function assumes the existence of an external function `ask_llm_and_extract_async`
          which handles asynchronous communication with the language model and extracts the desired
          information from the response.
        - The function also assumes the existence of an external function `correct_responses_index`
          which corrects the indices of the responses to account for skipped indices.
        - The function assumes the existence of an external function `handle_response` which handles
          errors and None values in the responses.
    """
    tasks = []
    skip_indices = []
    for index, row in figtables_exp_df.iterrows():
        if row["figure"]:
            prompt = "Following is the explanation of a figure in the paper."
            # prompt += "Give the title of the figure to be used as a slide title.\n"
            prompt += "Make up a title to be used as a slide title in a presentation explaining the paper."
            prompt += "If it is possible, make the title like a conclusion. For example, X affect Y. This is not mandatory, adjust with the figure content.  \n\n"
            prompt += "Do not add explanation or anything. Just the title."
            prompt += "Following is the explanation:\n\n"
            prompt += row["fig_exp"]
            tasks.append(ask_llm_and_extract_async(prompt, model=model))
        else:
            # If 'figure' is False, append None to the tasks list
            skip_indices.append(index)

    # Use asyncio.gather with 'return_exceptions=True' to handle potential errors
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Correct indices of response. Accounting for skipped indices.
    responses_index_corrected = correct_responses_index(
        responses, len(figtables_exp_df), skip_indices
    )

    # Handle errors and None values
    figtables_exp_df["figure_title"] = handle_response(
        responses_index_corrected, "figure_title"
    )

    return figtables_exp_df


async def get_read_figures_tables_summarize(
    figtables_df: pd.DataFrame,
    title_abstract: str,
    page_clf_df: pd.DataFrame,
    cleaned_texts: list,
    model_vision: str = "openai/gpt-4o",
    model_text: str = "openai/gpt-4o",
) -> tuple[pd.DataFrame, list]:
    """Asynchronously runs figure and table reading tasks and paper summarization.

    This function uses asyncio to parallelize the execution of three tasks:
    1. Reading figures using `run_read_figures`.
    2. Reading tables using `run_read_tables`.
    3. Summarizing the paper using `summarize_paper`.

    Args:
        figtables_df (pd.DataFrame): DataFrame containing information about figures and tables.
        title_abstract (str): Title and abstract of the paper.
        page_clf_df (pd.DataFrame): DataFrame containing page classification information.
        cleaned_texts (list): List of cleaned text sections from the paper.
        model_vision (str, optional): Name of the vision model to use for figure and table reading. Defaults to 'openai/gpt-4o'.
        model_text (str, optional): Name of the text model to use for paper summarization. Defaults to 'openai/gpt-4o'.

    Returns:
        tuple: A tuple containing two dataframes and a list:
            - figtables_exp_df (pd.DataFrame): Merged dataframe containing figure and table explanations.
            - summaries (list): List of summaries generated from the paper.
    """

    print("Reading figures and tables, summarizing.")
    task1 = asyncio.create_task(
        read_figures(figtables_df, title_abstract, model_vision)
    )
    task2 = asyncio.create_task(read_tables(figtables_df, title_abstract, model_vision))
    task3 = asyncio.create_task(
        summarize_paper(page_clf_df, cleaned_texts, title_abstract, model=model_text)
    )

    fig_exp_df, tbl_exp_df, summaries = await asyncio.gather(task1, task2, task3)

    # merge two dataframes
    figtables_exp_df = merge_figtables_exp_df(fig_exp_df, tbl_exp_df)
    return figtables_exp_df, summaries


def add_prefix(strings: list[str]) -> list[str]:
    """
    Adds the first line of the previous item starting with "# " as the first line of the current item if the current item doesn't start with "# ".

    Args:
      strings: A list of strings.

    Returns:
      A list of strings with the prefix added.
    """

    result = []
    prefix = ""

    for string in strings:
        if string.startswith("# "):
            prefix = string.splitlines()[0]
        else:
            string = prefix + "\n" + string
        result.append(string)

    return result


def split_long_slide_text(text: str, chunk_size: int = 450) -> str:
    """Splits a long text string into smaller chunks.

    This function uses a recursive character text splitter to divide the input
    text into chunks of the specified size. It then adds a prefix to each chunk
    and concatenates them into a single string, separated by newline characters.

    Args:
        text: The long text string to be split.
        chunk_size: The desired size of each text chunk in characters. Defaults to
            450.

    Returns:
        A string containing the split text chunks, separated by newline characters.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.create_documents([text])

    # extract text form weird list of langchain documents
    texts = [t.page_content for t in texts]
    texts = add_prefix(texts)

    # concat list into long str
    texts = "\n\n".join(texts)

    return texts


def prepare_slide_text(
    slide_title: str, content: str, title_suffix: str = "", chunk_size: int = 450
) -> str:
    """Prepares text for a slide, splitting long content into chunks.

    This function takes a slide title, content, and optional title suffix and
    chunk size. It formats the text with a heading and splits the content into
    chunks if it exceeds the specified chunk size.

    Args:
        slide_title: The title of the slide.
        content: The content of the slide.
        title_suffix: An optional suffix to append to the slide title.
            Defaults to an empty string.
        chunk_size: The maximum number of characters per chunk. Defaults to 450.

    Returns:
        A string containing the processed slide text, split into chunks if
        necessary.
    """

    processed_text = f"# {slide_title} {title_suffix}"
    processed_text += "\n\n"
    processed_text += content
    processed_text = split_long_slide_text(processed_text, chunk_size)

    return processed_text


def build_slide(
    figtables_exp_df: pd.DataFrame, title: str, summaries: dict, output_dir: str
) -> str:
    """Builds a slide string for a presentation.

    Args:
        figtables_exp_df: A pandas DataFrame containing information about figures and tables.
        title: The title of the slide.
        summaries: A dictionary containing text summaries for different sections of the slide.
        output_dir: The directory where the slide will be written

    Returns:
        A string representing the slide in Markdown format.
    """
    # Split long text in summaries into multiple slides
    print("Building slides")
    for section, text in summaries.items():
        summaries[section] = split_long_slide_text(text)

    slides = "---\n"
    slides += f'title: "{title}"\n'
    slides += "format: pptx\n"
    slides += "---\n"

    slides += "\n\n#" + title + "\n\n"

    # background
    slides += summaries["background"]
    slides += "\n\n\n"

    # method
    slides += summaries["method"]
    slides += "\n\n\n"

    # figures and tables
    # figtables_exp_df["fullpath"] = figtables_exp_df["path"].apply(
    #     lambda x: os.path.join(os.getcwd(), x)
    # )
    figtables_exp_df["relpath"] = figtables_exp_df["path"].apply(
        lambda x: os.path.relpath(x, output_dir)
    )
    for i, row in figtables_exp_df.iterrows():
        if row["figure"]:
            figure_caption_slide = prepare_slide_text(
                row["figure_title"], row["caption"], title_suffix="(figure legend)"
            )
            figure_exp_slide = prepare_slide_text(
                row["figure_title"], row["fig_exp"], title_suffix="(explanation)"
            )

            slides += f"# {row['figure_title']}"
            slides += "\n\n"
            slides += f"![ ]({row['relpath']})"
            slides += "\n\n\n"
            slides += figure_caption_slide
            slides += "\n\n\n"
            slides += figure_exp_slide
            slides += "\n\n\n"
        elif row["table"]:
            table_exp_slide = prepare_slide_text(row["caption"], row["fig_exp"])

            slides += f"# {row['caption']}"
            slides += "\n\n"
            slides += f"![ ]({row['relpath']})"
            slides += "\n\n\n"
            slides += table_exp_slide
            slides += "\n\n\n"

    # result
    slides += summaries["result"]
    slides += "\n\n\n"

    # discussion
    slides += summaries["discussion"]
    slides += "\n\n\n"

    # conclusion
    slides += summaries["conclusion"]
    slides += "\n\n\n"

    # clean "# ##"
    slides = re.sub(r"#\s#+", "# ", slides)

    return slides


def write_qmd_slides(slides: str, output_dir: str, current_datetime: str):
    """Writes a string of Quarto Markdown (QMD) slides to a file.

    This function takes a string containing QMD slides, an output directory, and a current datetime string.
    It creates a file named with the datetime and "paper_presentation.qmd" in the specified output directory,
    and writes the slides content to it.

    Args:
        slides (str): The QMD slides content as a string.
        output_dir (str): The directory where the QMD file should be saved.
        current_datetime (str): A string representing the current datetime, used for the filename.

    Returns:
        str: The full path to the created QMD file.
    """
    filename = f"{current_datetime}_paper_presentation.qmd"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as file:
        file.write(slides)

    return filepath


def extract_pdf_figtables(
    pdf_file: str,
    pdf_converted_zip: str,
    extracted_zip_dir: str,
    cleaned_texts: str,
    model_main: str = "google/gemini-pro",
    model_fallback: str = "anthropic/claude-3-haiku",
) -> pd.DataFrame:
    """Extracts figure and table information from a PDF file.

    This function utilizes Adobe API to extract figures and tables from a PDF file, classifies them as scientific or not,
    and extracts captions and titles using a language model.

    Args:
        pdf_file (str): Path to the PDF file.
        pdf_converted_zip (str): Path to the zip file containing the converted PDF.
        extracted_zip_dir (str): Path to the directory where extracted figures and tables are stored.
        cleaned_texts (str): Path to the file containing cleaned text from the PDF.
        model_main (str, optional): Name of the main language model to use for caption and title extraction. Defaults to 'google/gemini-pro'.
        model_fallback (str, optional): Name of the fallback language model to use for caption and title extraction. Defaults to 'anthropic/claude-3-haiku'.

    Returns:
        pd.DataFrame: A DataFrame containing information about extracted figures and tables, including their paths, classifications, captions, and titles.

    Raises:
        None

    Example:
        >>> extract_pdf_figtables('path/to/pdf.pdf', 'path/to/converted.zip', 'path/to/extracted', 'path/to/cleaned_texts.txt')
    """
    start = time.time()
    start_adobe = time.time()
    # extract with adobe API
    run_pdf_extraction(pdf_file, pdf_converted_zip, extracted_zip_dir)
    end_adobe = time.time()
    adobe_api_dur = end_adobe - start_adobe
    print(f"Adobe API duration:{adobe_api_dur}")

    # get figtables path
    figtables = get_figtables_info(extracted_zip_dir)

    # classify whether a scientific figtables or not
    figtables_classification = asyncio.run(
        classify_figtables_async(figtables, model=model_fallback)
    )

    # merge figtables path and classification
    figtables_df = merge_figtables_info_classification(
        figtables, figtables_classification
    )

    # Run get figure caption and table title separately
    figtables_caption_df = asyncio.run(
        get_figtables_caption_table_titles(
            figtables_df,
            cleaned_texts,
            model_main=model_main,
            model_fallback=model_fallback,
        )
    )
    end = time.time()
    total_dur = end - start
    print(f"extract_pdf_figtables duration:{total_dur}")
    return figtables_caption_df


def classify_page_extract_title_abstract(
    cleaned_texts: List[str], model: str = "anthropic/claude-3-haiku"
) -> Tuple[pd.DataFrame, str, str]:
    """Classifies pages, extracts title and abstract from a list of cleaned texts.

    Args:
      cleaned_texts: A list of cleaned text strings representing pages.
      model: The name of the language model to use for classification and extraction.
        Defaults to 'anthropic/claude-3-haiku'.

    Returns:
      A tuple containing:
        - page_clf_df: A pandas DataFrame containing the classification results for each page.
        - title: The extracted title of the document.
        - title_abstract: The extracted abstract of the document.
    """
    page_clf_df = asyncio.run(classify_page_async(cleaned_texts, model=model))
    title, title_abstract = extract_title_abstract(cleaned_texts, model=model)
    print("Title and abstract has been extracted.")

    return page_clf_df, title, title_abstract


def get_pptx_path(qmd_path: str) -> str:
    """
    Returns the path to the corresponding PowerPoint (.pptx) file given a Quarto Markdown (.qmd) file path.

    Args:
        qmd_path (str): The path to the Quarto Markdown file.

    Returns:
        str: The path to the corresponding PowerPoint file.
    """
    # Get the file name without the extension
    file_name = os.path.splitext(qmd_path)[0]
    # Construct the new path using os.path.join
    pptx_filename = file_name + ".pptx"

    return pptx_filename


def generate_pptx(qmd_path: str) -> tuple[str, int]:
    """
    Generate a PowerPoint presentation from a Quarto Markdown (QMD) file.

    Args:
        qmd_path (str): The relative path to the Quarto Markdown file.

    Returns:
        A tuple containing:
        - The output of the Quarto rendering command as a string.
        - The return code of the Quarto rendering command (0 for success, non-zero for failure).
    """
    print(qmd_path)
    command = f"quarto render {qmd_path}"
    try:
        # Use subprocess.run to execute the command in the specified directory
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        if result.stderr:
            # Decode the bytes to a string using the appropriate encoding
            error_message = result.stderr.decode("utf-8", errors="replace")
            print(f"Error: {error_message}")
        else:
            print("PPTX has been generated.")
        return result.stdout, result.returncode
    except FileNotFoundError:
        print(f"Error: Command '{command}' not found.")
        return None, 1
    except Exception as e:
        print(f"Error: {e}")
        return None, 1
