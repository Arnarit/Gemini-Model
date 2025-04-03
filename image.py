import streamlit as st
import os
import requests
from PIL import Image
import io
from groq import Groq

# --- Configuration and API Keys ---

# 1. Groq API Key:
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets["groq"]["GROQ_API_KEY"]
    except KeyError:
        st.error("Groq API key is required. Set GROQ_API_KEY or use Streamlit secrets.")
        st.stop()

# 2. Replicate API Token (for image generation):
replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
if not replicate_api_token:
    try:
        replicate_api_token = st.secrets["replicate"]["REPLICATE_API_TOKEN"]
    except KeyError:
        st.error("Replicate API token is required. Set REPLICATE_API_TOKEN or use Streamlit secrets.")
        st.stop()

# --- Groq Client Setup ---
client = Groq(api_key=groq_api_key)

# --- Prompt Generation (Groq) ---

def generate_image_prompt(user_input, model="llama-3.3-70b-versatile"):
    """Generates a detailed image prompt using Groq."""

    system_prompt = """You are a helpful assistant that generates detailed, high-quality prompts for image generation models.
    The user will provide a basic description of a yoga pose or fitness activity.  Your task is to expand this into a
    rich, descriptive prompt that includes details about the setting, lighting, style, and any other relevant visual elements.
    Focus on creating a prompt that will result in a photorealistic, visually appealing, and accurate depiction of the pose.
    
    Example:
    User: "person doing tree pose"
    Assistant: "A photorealistic image of a person in a serene yoga studio, performing the tree pose (Vrksasana). Soft, natural light streams in through a large window, illuminating the scene. The person is wearing comfortable yoga attire, and their expression is calm and focused. The background is slightly blurred, emphasizing the subject.  High resolution, 8k, cinematic lighting."
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model=model,
        temperature=0.7,
        max_tokens=200,
        top_p=0.95,
        stop=None,
        stream=False #Stream is False for simplicity
    )
    return chat_completion.choices[0].message.content


# --- Image Generation (Replicate) ---

def generate_image_replicate(prompt, replicate_model="bytedance/sdxl-lightning-4step:5599ed30703defd1d160a25a63321b4dec97101d98b4674bcc56e41f62f35637"):  # SDXL by default
    """Generates an image using the Replicate API."""
    try:
        output = replicate.run(
            replicate_model,
            input={"prompt": prompt,
                   "width": 512,
                   "height": 512,
                   }
        )
        if output and len(output) > 0:
            image_url = output[0]
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return image
        else:
            st.error("Replicate API returned no image.")
            return None

    except Exception as e:
        st.error(f"Error generating image with Replicate: {e}")
        return None

# --- Yoga Pose Recommendation (Groq) ---
def generate_yoga_recommendation(user_input, model="llama-3.3-70b-versatile"):
    """Generates a text-based yoga pose recommendation using Groq."""

    system_prompt = """You are a knowledgeable yoga instructor.  The user will describe their fitness level, goals, or any limitations.
    Provide a yoga pose recommendation, including:
    1.  The name of the pose (Sanskrit and English).
    2.  A brief, step-by-step description of how to perform the pose.
    3.  Any modifications or precautions, especially if the user mentioned limitations.
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model=model,
        temperature=0.7,
        max_tokens=300,
        top_p=0.95,
        stop=None,
        stream=False #Stream is False for simplicity
    )

    return chat_completion.choices[0].message.content

# --- Streamlit App ---

def main():
    st.title("Yoga Pose Generator and Recommender (Groq)")
    st.write("Describe your desired yoga pose or fitness needs, and I'll generate an image and provide a recommendation!")

    user_input = st.text_area("Describe your yoga pose or fitness needs:",
                               placeholder="e.g., 'I want a relaxing pose for beginners.' or 'A challenging pose for core strength.'",
                               height=100)
    
     # --- Replicate Model Selection (optional) ---
    with st.expander("Advanced: Image Model Selection"):
      replicate_model = st.selectbox("Choose an image generation model on Replicate:", [
          "bytedance/sdxl-lightning-4step:5599ed30703defd1d160a25a63321b4dec97101d98b4674bcc56e41f62f35637"
      ], index=0)


    if st.button("Generate"):
        if user_input:
            with st.spinner("Generating prompt with Groq..."):
                image_prompt = generate_image_prompt(user_input)
                st.markdown("**Generated Image Prompt:**")
                st.text_area("", value=image_prompt, height=150, disabled=True)

            with st.spinner("Generating image with Replicate..."):
                image = generate_image_replicate(image_prompt, replicate_model)
                if image:
                    st.image(image, caption="Generated Yoga Image", use_column_width=True)

            with st.spinner("Generating yoga pose recommendation with Groq..."):
                recommendation = generate_yoga_recommendation(user_input)
                st.markdown("**Yoga Pose Recommendation:**")
                st.write(recommendation)
        else:
            st.warning("Please enter a description.")

if __name__ == "__main__":
    import replicate  # Import replicate here for modularity
    main()
    
