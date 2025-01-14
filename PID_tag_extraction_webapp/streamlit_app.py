import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Initialize the API key and configure Gemini
api_key = "AIzaSyAWO-jh0mkP36_TInm5WANzBC27PLDFwqk"
genai.configure(api_key=api_key)

# Constants
canvas_resolution = (8270, 6675)
right = 8400
bottom = 6850
left = 130
top = 175
sat_thresh = 25
val_thresh = 25

# Define functions
def send_patch_to_llm_tag(pic):
    """Send the patch to the LLM API and handle the response."""
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=5000,
        temperature=0,
        top_p=0,
        top_k=1,
    )

    buffered = BytesIO()
    pic.save(buffered, format="PNG")
    png_data = buffered.getvalue()

    model_name = "gemini-2.0-flash-exp"

    system_prompt = """You are a specialized AI assistant trained to extract data from Piping and Instrumentation Diagrams (P&IDs). Your task is to accurately identify and extract various tags from a P&ID provided by the user.

      Objective:
      Extract all instrument and pipeline tags from a given P&ID and bind each tag to its corresponding subsystem and output them as a flat JSON list of strings. Each extracted tag must be prefixed with its corresponding subsystem tag.

      Instructions:

        Step by step Instruction for tag exrtaction:

          1- Identify all subsystem tags that are present in the diagram, more importantly their color
          2- Start to extract pipline and instrument tags of each subsystem based on the color of the tag
          3- Bind the extracted tag to the subsystem tag that have the same color
            Example: If a subsystem is identified to have blue color, all blue tags in the diagram belong to that subsystem.

        Most useful information for tag extraction process:

        *  Subsystem Identification:
            *   Subsystems are visually distinguished by different colors in the P&ID
            *   Note that a color change within the diagram indicates a transition to another subsystem.
            *   Subsystem tags are enclosed in rectangles with solid outlines (same tags in dashed-outline rectangles are not subsystem tags)
            *   They follow a three or four-part structured format (e.g., "GH0101-ABCD-101", "GH0101-AB-012").
            *   There might be a subsystem tag label near each subsystem that represents the tag of the subsystem.

        *  Tag Extraction:
            *   Instrument Tags:
                *   Generally have a two-part structure (e.g., "TI-4072", "PI-3077").
                *   May include suffix variations like "A", "B", "C" (e.g., "FIT-4072A", "FIT-4072B").
                *   May also appear as large underlined text
                *   Possible instrument tag prefixes include: SC, RO, PV, TIT, MOV, PIT, LIT, LG, MZI, CMT, AE, PZV, AIT, HZS, XZV, TI, FO, PY, TSV, IP, PT, CT, MOVS, CP, FV, PALL, PIC, PAHH, ZHS, ZC, LIC, LAHH, LALL, LAH, LAD, PAD, TP, FQI, LY, PDAH, PDI, CMI, FI, ZV, HV, TE, PDIT, PI, CME, FIT, FE, LV, ME
            *   Pipeline Tags:
                *   Have a multi-part structured format with 5-6 sections separated by hyphens.
                *   May include fractional pipe sizes (e.g., "¾").
                *   Example formats: "My-y-M-y-yMMyMy", "My-y-MMM-y-yMMyM".

        * Output Format:
            *   Output a flat JSON array of strings.
            *   Example:
                ```json
                [
                "Subsystem-1-Tag: SC-001",
                "Subsystem-1-Tag: FIT-002A",
                "Subsystem-1-Tag: My-y-M-y-yMMyMy",
                "Subsystem-2-Tag: LIT-003",
                "Subsystem-2-Tag: RO-004",
                "Subsystem-2-Tag: MOV-005",
                "Subsystem-2-Tag: My-y-MMM-y-yMMyMy"
                ]
                ```

      Important considerations:
      *   You must differentiate between colors accurately to be able to bind tags to their correct subsystem correctly.
      *   Focus on color changes to accurately determine subsystem boundaries.
      *   After Identidying each subsystem, make sure all tags of each subsystem will get extracted.
      *   Ensure all tags are captured and prefixed correctly."""

    model = genai.GenerativeModel(model_name, system_instruction=system_prompt, generation_config=generation_config)

    picture = {
        'mime_type': 'image/png',
        'data': png_data,
    }

    user_prompt = "Extract all tags from the P&ID, following the guidelines provided."

    response = model.generate_content([picture, user_prompt])

    if response.parts:
        llm_output = response.text
        return llm_output
    else:
        st.error("No text content generated.")
        return ""

def process_single_page(pdf_file, page_number):
    """Process a single page of the PDF."""
    # Open the PDF file
    doc = fitz.open(stream=pdf_file, filetype="pdf")

    # Load page
    page = doc.load_page(page_number)

    # Get image
    pix = page.get_pixmap(dpi=300)

    # Convert to PIL image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Initial crop
    imge = img.crop((left, top, right, bottom))
    imge = np.array(imge)

    hsv = cv2.cvtColor(imge, cv2.COLOR_BGR2HSV)

    # Threshold for saturation and value (brightness)
    mask = (hsv[:, :, 1] < sat_thresh) | (hsv[:, :, 2] < val_thresh)

    # Set the pixels to white in the original image
    imge[mask] = [255, 255, 255]
    ima = Image.fromarray(imge)

    # Find the first non-white row
    first_non_white_row = next(
        (row for row in range(imge.shape[0]) if not np.all(imge[row] == [255, 255, 255])), 0
    )

    # Find the first non-white column
    first_non_white_col = next(
        (col for col in range(imge.shape[1]) if not np.all(imge[:, col] == [255, 255, 255])), 0
    )

    # Crop the image
    cropped_img = ima.crop((first_non_white_col, first_non_white_row, ima.width, ima.height))

    # Create a new blank canvas with the target resolution
    canvas_img = Image.new("RGB", canvas_resolution, color="white")

    # Paste the cropped image onto the canvas at the top-left corner
    canvas_img.paste(cropped_img, (0, 0))

    # Extract tags from the processed image
    page_output = send_patch_to_llm_tag(canvas_img)

    return page_output

# Streamlit App
st.title("P&ID Tag Extractor")

# File uploader
uploaded_file = st.file_uploader("Upload a P&ID PDF", type="pdf")

if uploaded_file is not None:
    # Display uploaded file name
    st.success(f"Uploaded file: {uploaded_file.name}")

    # Read the uploaded PDF
    pdf_bytes = BytesIO(uploaded_file.read())
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    # Page selector
    page_number = st.number_input(
        "Select Page Number",
        min_value=1,
        max_value=total_pages,
        step=1,
        format="%d",
    )

    if st.button("Extract Tags"):
        st.write("Processing page...")
        # Extract tags
        try:
            output = process_single_page(pdf_bytes, page_number - 1)
            st.success("Processing complete!")
            st.write("Extracted Tags:")
            st.text_area("Extracted Tags", output, height=300)
        except Exception as e:
            st.error(f"Error: {str(e)}")