from tkinter import HORIZONTAL
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

img = Image.open('C:\Users\dell\OneDrive\Desktop\MLproject\images\logo.jpg')

st.set_page_config(page_title="leafmedi", page_icon=img)

def main():
    st.title("Welcome to Identify Herb Plants")

    # Load your logo image
    logo_image = "C:\Users\dell\OneDrive\Desktop\MLproject\images\logo.jpg"  # Replace with the path to your logo image

    # Display the logo
    st.image(logo_image, width=200)  # Adjust the width as needed

    # Rest of your Streamlit app content
    st.write("Medicine Plants")

if __name__ == "__main__":
    main()

    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Home", "Classify"],
            icons=["house", "bicycle"],
            menu_icon=None,

            styles={
                "container": {"padding": "20px", "background-color": "#3AA746"},  # Background color (leaf green)
                "icon": {"color": "white", "font-size": "30px"},  # Icon color and size
                "nav-link": {
                    "font-size": "20px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#52C98B",  # Icon color after hover (a slightly lighter green)
                },
                "nav-link-selected": {"background-color": "#52C98B", "color": "white"},  # Background color and text color when tab is selected
            }
        )

if selected == "Home":
    st.title("Identify Herb Plants - Medicinal Plant Recognition")
    st.header("Understanding the Importance of Identifying Medicinal Plants")
    
    st.write("Medicinal plants have played a crucial role in traditional medicine for centuries.")
    st.write("The ability to identify herb plants is essential for various reasons:")
    
    st.write("- **Medical Applications:** Many medicinal plants are used to extract compounds that form the basis of pharmaceutical drugs.")
    st.write("- **Traditional Medicine:** Herbal remedies are integral to traditional medicine practices across different cultures.")
    st.write("- **Conservation:** Recognizing and preserving medicinal plants is vital for biodiversity and ecological balance.")
    
    st.header("Why Use Herb Plant Identification Tools?")
    
    st.write("The development of technology has paved the way for herb plant identification tools, offering benefits such as:")
    
    st.write("- **Accessibility:** Easy access to information about various medicinal plants.")
    st.write("- **Safety:** Avoidance of harmful or poisonous plants during herbal remedy preparation.")
    st.write("- **Education:** Learning and spreading awareness about the importance of medicinal plants.")
    
    st.header("Note:")
    
    st.write("Before utilizing any identification tool, it's important to understand the significance of proper plant identification.")
    st.write("This program aims to provide a user-friendly platform for identifying herb plants.")
    
    st.write("Program by: Srilekha S")

# Load the pre-trained ResNet model
model_path = "C:\Users\dell\OneDrive\Desktop\MLproject\model_resnet.h5"
model = tf.keras.models.load_model(model_path)

# Mapping of class indices to class labels (replace with your own mapping)
class_labels = {
    0: 'Alpinia Galanga (Rasna)',
    1: 'Amaranthus Viridis (Arive-Dantu)',
    2: 'Artocarpus Heterophyllus (Jackfruit)',
    3: 'Azadirachta Indica (Neem)',
    4: 'Basella Alba (Basale)',
    5: 'Brassica Juncea (Indian Mustard)',
    6: 'Carissa Carandas (Karanda)',
    7: 'Citrus Limon (Lemon)',
    8: 'Ficus Auriculata (Roxburgh fig)',
    9: 'Ficus Religiosa (Peepal Tree)',
    10: 'Hibiscus Rosa-sinensis',
    11: 'Jasminum (Jasmine)',
    12: 'Mangifera Indica (Mango)',
    13: 'Mentha (Mint)',
    14: 'Moringa Oleifera (Drumstick)',
    15: 'Muntingia Calabura (Jamaica Cherry-Gasagase)',
    16: 'Murraya Koenigii (Curry)',
    17: 'Nerium Oleander (Oleander)',
    18: 'Nyctanthes Arbor-tristis (Parijata)',
    19: 'Ocimum Tenuiflorum (Tulsi)',
    20: 'Piper Betle (Betel)',
    21: 'Plectranthus Amboinicus (Mexican Mint)',
    22: 'Pongamia Pinnata (Indian Beech)',
    23: 'Psidium Guajava (Guava)',
    24: 'Punica Granatum (Pomegranate)',
    25: 'Santalum Album (Sandalwood)',
    26: 'Syzygium Cumini (Jamun)',
    27: 'Syzygium Jambos (Rose Apple)',
    28: 'Tabernaemontana Divaricata (Crape Jasmine)',
    29: 'Trigonella Foenum-graecum (Fenugreek)'
}

def preprocess_image(image):
    # Resize the image to the required input size of the model
    image = image.resize((100, 100))

    # Preprocess the image for the model
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = preprocess_input(img_array)

    return img_array

def classify_image(image):
    # Preprocess the input image
    img_array = preprocess_image(image)

    # Get model predictions
    predictions = model.predict(img_array)

    # Process predictions based on your model's output shape
    decoded_predictions = process_predictions(predictions)

    return decoded_predictions

def process_predictions(predictions):
    # This is a placeholder function; adjust it based on your model's output shape and logic
    # For example, if your model outputs class probabilities directly, you might return the top-k classes.
    top_classes = tf.math.top_k(predictions, k=3).indices.numpy()[0]
    class_names = [class_labels[class_index] for class_index in top_classes]
    return class_names

def main():
    st.title("Herb Plant Classification")

    # Allow users to upload an image file (supports common image formats)
    uploaded_image = st.file_uploader("Note: Only image files (jpg, jpeg, png) are allowed", type=["jpg", "jpeg", "png"])

    # Image Classification
    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

        # Perform classification on the uploaded image using the ResNet model
        predictions = classify_image(Image.open(uploaded_image))

        st.header('**Classification Results:**')
        st.write(predictions)
    else:
        st.info('Awaiting for an image to be uploaded.')

    if st.button('Use Example Image'):
        # Load an example image for testing
        example_image_path = "C:\Users\dell\OneDrive\Desktop\MLproject\images\tulsi.jpg"
        example_image = Image.open(example_image_path)

        # Display the example image
        st.image(example_image, caption="Example Image.", use_column_width=True)

        # Perform classification on the example image using the ResNet model
        example_predictions = classify_image(example_image)

        st.header('**Example Classification Results:**')
        st.write(example_predictions)

if __name__ == "__main__":
    main()