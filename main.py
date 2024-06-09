import streamsync as ss
import asyncio
import os
import time
import multiprocess as mp
import slide_generator as sg
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv(".env")

# Shows in the log when the app starts
print("ScholarSlide started.")


def handle_openrouter_api(state, payload):
    state["openrouter_api_key"] = payload
    os.environ["OPENROUTER_API_KEY"] = state["openrouter_api_key"]


def handle_pdf_id(state, payload):
    state["pdf_services_client_id"] = payload
    os.environ["PDF_SERVICES_CLIENT_ID"] = state["pdf_services_client_id"]


def handle_pdf_secret(state, payload):
    state["pdf_services_client_secret"] = payload
    os.environ["PDF_SERVICES_CLIENT_SECRET"] = state["pdf_services_client_secret"]


def handle_model_best(state, payload):
    state["model_best"] = payload


def handle_model_eco(state, payload):
    state["model_eco"] = payload


def handle_model_eco_fallback(state, payload):
    state["model_eco_fallback"] = payload


def handle_model_vision_best(state, payload):
    state["model_vision_best"] = payload


def handle_model_vision_efficient(state, payload):
    state["model_vision_efficient"] = payload


def handle_timer_tick(state):
    time.sleep(9999)


def handle_file_upload(state, payload):

    # An array of dictionaries is provided in the payload
    # The dictionaries have the properties name, type and data
    # The data property is a file-like object

    # Assuming only one file is uploaded
    uploaded_file = payload[0]

    # get file info
    name = uploaded_file.get("name")
    file_type = uploaded_file.get("type")
    file_data = uploaded_file.get("data")

    # set folder to store
    state["current_datetime"] = sg.get_datetime()
    state["_output_dir"] = os.path.join(
        os.path.abspath(os.getcwd()), "output", state["current_datetime"]
    )

    # create output dir
    os.makedirs(state["_output_dir"], exist_ok=True)

    # Check if the file type is PDF
    if file_type == "application/pdf":
        filename = f"{name}.pdf"
        state["_pdf_filepath"] = os.path.join(state["_output_dir"], filename)

        with open(state["_pdf_filepath"], "wb") as file_handle:
            file_handle.write(file_data)
        state["_pdf_ready"] = True
        state["pdf_notification"] = (
            "+PDF file uploaded! Press submit to start generating."
        )
    else:
        state["pdf_notification"] = "-Selected file is not a PDF."
        print(f"Skipping file {name} as it is not a PDF.")


def handle_submit(state):
    if "OPENROUTER_API_KEY" not in os.environ or not os.environ["OPENROUTER_API_KEY"]:
        state["submit_notification"] = "-Please fill OpenRouter API key in the sidebar."

    elif (
        "PDF_SERVICES_CLIENT_ID" not in os.environ
        or not os.environ["PDF_SERVICES_CLIENT_ID"]
    ):
        state["submit_notification"] = (
            "-Please fill Adobe PDF services client ID in the sidebar."
        )

    elif (
        "PDF_SERVICES_CLIENT_SECRET" not in os.environ
        or not os.environ["PDF_SERVICES_CLIENT_SECRET"]
    ):
        state["submit_notification"] = (
            "-Please fill Adobe PDF services client secret in the sidebar."
        )

    elif not state["_pdf_ready"]:
        state["submit_notification"] = "-Please upload a PDF file first."
    else:
        state["submit_notification"] = ""
        state["processing"] = True
        state["pptx_ready"] = False
        state["processing_progress"] = "Processing..."
        generate_pptx_from_pdf(state)
        state["processing"] = False
        state["pptx_ready"] = True


def handle_file_download(state):
    # Pack the file as a FileWrapper object
    data = ss.pack_file(
        state["_pptx_filepath"],
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    file_name = os.path.basename(state["_pptx_filepath"])

    state.file_download(data, file_name)


def handle_subject_expertise(state, payload):
    os.environ["EXPERTISE_FIELD"] = state["expertise_field"]


def generate_pptx_from_pdf(state):
    start_time = time.time()

    state["processing_progress"] = "Extracting texts..."
    pdf_converted_zip = os.path.join(state["_output_dir"], "pdf_converted.zip")
    extracted_zip_dir = os.path.join(state["_output_dir"], "extracted_pdf_converted")

    # Extract PDF with fitz
    cleaned_texts, cleaned_text = sg.extract_pdf_text(state["_pdf_filepath"])

    # Run extract figtables and page classification+extract title abstract in separate process
    # with multiprocessing.Pool(processes=2) as pool:
    start_multiproc = time.time()

    with mp.Pool(2) as pool:
        # run_pdf_extraction(adobe api) ->  get_figtables_info -> run_classify_figtables_async ->
        # merge_figtables_info_classification -> get_figtables_caption_table_titles
        state["processing_progress"] = "Extracting figures and tables..."
        result_one = pool.apply_async(
            sg.extract_pdf_figtables,
            (
                state["_pdf_filepath"],
                pdf_converted_zip,
                extracted_zip_dir,
                cleaned_texts,
                state["model_eco"],
                state["model_eco_fallback"],
            ),
        )
        result_two = pool.apply_async(
            sg.classify_page_extract_title_abstract,
            (cleaned_texts, state["model_eco_fallback"]),
        )

        # Get the results
        figtables_df = result_one.get()
        page_clf_df, title, title_abstract = result_two.get()

    end_multiproc = time.time()
    multiproc_time = end_multiproc - start_multiproc
    print(f"Multiproc time: {multiproc_time}")

    start_read_figtables_summarize = time.time()
    state["processing_progress"] = "Interpreting figures and tables..."
    print("Interpreting figures and tables...")
    figtables_exp_df, summaries = asyncio.run(
        sg.get_read_figures_tables_summarize(
            figtables_df,
            title_abstract,
            page_clf_df,
            cleaned_texts,
            model_vision=state["model_vision_best"],
            model_text=state["model_best"],
        )
    )

    end_read_figtables_summarize = time.time()
    read_figtables_time_summarize = (
        end_read_figtables_summarize - start_read_figtables_summarize
    )
    print(f"Read figtables summarize time: {read_figtables_time_summarize}")

    start_get_fig_title = time.time()
    state["processing_progress"] = "Summarizing..."
    print("Generating figure titles from figure explanations")
    # figtables_exp_df = asyncio.run(sg.run_get_figure_titles(figtables_exp_df, model=state["model_eco"]))
    figtables_exp_df = asyncio.run(
        sg.get_figure_titles(figtables_exp_df, model=state["model_eco"])
    )
    end_get_fig_title = time.time()
    get_figtables_time = end_get_fig_title - start_get_fig_title
    print(f"Get figtables title time: {get_figtables_time}")

    state["processing_progress"] = "Creating slides..."
    print("Compiling slides...")
    slides = sg.build_slide(figtables_exp_df, title, summaries, state["_output_dir"])
    qmd_path = sg.write_qmd_slides(
        slides, state["_output_dir"], state["current_datetime"]
    )

    output, return_code = sg.generate_pptx(qmd_path)
    if return_code == 0:
        print("Command executed successfully")
    else:
        print(f"Command execution failed with return code {return_code}.")

    state["_pptx_filepath"] = sg.get_pptx_path(qmd_path)
    state["processing_progress"] = ""
    end_time = time.time()

    total_time = end_time - start_time

    print(f"Total time: {total_time}")


# Initialise the state
initial_state = ss.init_state(
    {
        "my_app": {"title": "ScholarSlide"},
        "counter": 0,
        "processing_progress": "",
        "processing": False,
        "pptx_ready": False,
        "_current_datetime": None,
        "_pdf_filepath": None,
        "_output_dir": None,
        "_pdf_ready": False,
        "_pdf_converted_zip": None,
        "_extracted_zip_dir": None,
        "_pptx_filepath": None,
        "openrouter_api_key": None,
        "pdf_services_client_id": None,
        "pdf_services_client_secret": None,
        "pdf_notification": "",
        "submit_notification": "",
        "expertise_field": "biomedical",
        "model_best": "openai/gpt-4o",
        "model_eco": "google/gemini-pro",
        "model_eco_fallback": "anthropic/claude-3-haiku",
        "model_vision_best": "openai/gpt-4o",
        "model_vision_efficient": "google/gemini-pro-vision",
    }
)
