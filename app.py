import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
import json
import re
import os

# Replace with your actual API key
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

def plot_bounding_boxes(im, noun_phrases_and_positions):
    """
    Plots bounding boxes on an image with markers for each noun phrase, using PIL, normalized coordinates, and different colors.

    Args:
        im: The PIL Image object to draw on.
        noun_phrases_and_positions: A list of tuples containing the noun phrases
         and their positions in normalized [y1, x1, y2, x2] format.
    """
    im = Image.open(im)
    img = im.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)

    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown',
        'gray', 'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 
        'maroon', 'teal', 'olive', 'coral', 'lavender', 'violet', 'gold', 'silver'
    ]

    for i, (noun_phrase, (y1, x1, y2, x2)) in enumerate(noun_phrases_and_positions):
        color = colors[i % len(colors)]

        abs_x1 = int(x1 * width)
        abs_y1 = int(y1 * height)
        abs_x2 = int(x2 * width)
        abs_y2 = int(y2 * height)

        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
        )

        text_position = (abs_x1 + 8, abs_y1 + 6)
        draw.text(text_position, noun_phrase, fill=color)

    return img

def format_input(data):
    """
    Converts the input list of tuples into the format required by the plot_bounding_boxes function.

    Args:
        data: A list of tuples in the format [(noun_phrase, [x1, y1, x2, y2]), ...].

    Returns:
        A list of tuples in the format [(noun_phrase, [y1, x1, y2, x2]), ...],
        where coordinates are normalized (0–1).
    """
    formatted_data = []

    for noun_phrase, coordinates in data:
        x1, y1, x2, y2 = coordinates
        normalized_coordinates = [x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000]
        formatted_data.append((noun_phrase, normalized_coordinates))

    return formatted_data

def parse_list_boxes_with_label(text):
    text = text.split("```\n")[0]
    try:
        result =  json.loads(text.strip("```").strip("python").strip("json").replace("'", '"').replace('\n', '').replace(',}', '}'))
        return result
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return None, {}
    

def generate_prompt(object_list):
    objects_string = ", ".join(object_list)
    return f"Return bounding boxes for each object: {objects_string} in the following format as \
        a list.\n {{'{object_list[0]}_0': [ymin, xmin, ymax, xmax], ...}}. If there are more than \
            one instance of an object, add them as 'object_0', 'object_1', etc. \
            Output only a valid JSON. Do not add any other information."

def add_boxes_to_image(image_file, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    try:
        image_bytes = image_file.getvalue()

        image_part = {
            "mime_type": "image/png" if image_file.type == "image/png" else "image/jpeg",
            "data": image_bytes
        }

        response = model.generate_content([
            image_part, prompt,
        ])

        boxes = parse_list_boxes_with_label(response.text)
        noun_phrases_and_positions = list(boxes.items())
        noun_phrases_and_positions = format_input(noun_phrases_and_positions)

        object_counts = {}
        for noun_phrase, _ in noun_phrases_and_positions:
            object_name = noun_phrase.split("_")[0]
            object_counts[object_name] = object_counts.get(object_name, 0) + 1

        output_image = plot_bounding_boxes(image_file, noun_phrases_and_positions)
        return output_image, object_counts
    
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return None, {}

def main():
    st.title("Detect and Count Objects in an Image using Gemini AI")
    st.write("Upload an image and choose a task for analysis.")

    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    selected_objects = st.text_input("Enter objects to analyze (comma-separated)")

    if st.button("Analyze Image"):
        if selected_objects:
            with st.spinner("Processing..."):
                object_list = [obj.strip() for obj in selected_objects.split(",") if obj.strip()]
                prompt = generate_prompt(object_list)
                response, object_counts = add_boxes_to_image(uploaded_image, prompt)

            if response is not None:
                st.image(response, caption="Analyzed Image", use_container_width=True)
                st.subheader("Detected Objects:")
                for obj, count in object_counts.items():
                    st.write(f"{obj}: {count}")
            else:
                st.write("Error: Could not display image.")

if __name__ == "__main__":
    main()
