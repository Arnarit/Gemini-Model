import streamlit as st
import google.generativeai as genai
import os
import tempfile
import time
from dotenv import load_dotenv
from PIL import Image

# --- Configuration ---
load_dotenv()  # Load environment variables from .env
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("🚨 Google API Key not found! Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    # Using gemini-1.5-flash as it's generally available and good with multimodal tasks
    # If you have access to gemini-1.5-pro, you might prefer it for potentially higher quality.
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
except Exception as e:
    st.error(f"🚨 Error configuring Google AI: {e}")
    st.stop()

# --- Streamlit App ---

st.set_page_config(page_title="Fitness/Medical Video Analysis Chatbot", layout="wide")

# --- Crucial Disclaimer ---


st.title("🎬 Video Analysis Chatbot 💬")

# --- Helper Functions ---

@st.cache_resource # Cache the video processing part
def upload_video_to_gemini(uploaded_file):
    """Uploads the video file to Google AI Studio files."""
    if uploaded_file is None:
        return None

    try:
        # Create a temporary file to ensure Gemini can access it
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            st.info(f"⏳ Uploading video: {uploaded_file.name}...")

        # Upload the file using the Gemini API
        video_file = genai.upload_file(path=tmp_file_path)
        st.info(f"✨ Video '{uploaded_file.name}' uploaded successfully. URI: {video_file.uri}")

        # Polling to check if the file is ready (ACTIVE state)
        # This is important because processing takes time.
        polling_interval = 5 # seconds
        max_polling_time = 300 # 5 minutes
        start_time = time.time()

        while video_file.state.name == "PROCESSING":
            if time.time() - start_time > max_polling_time:
                st.error(f"🚨 Video processing timed out after {max_polling_time} seconds.")
                os.unlink(tmp_file_path) # Clean up temp file
                return None
            time.sleep(polling_interval)
            video_file = genai.get_file(video_file.name) # Refresh file state
            st.info(f"⌛ Video state: {video_file.state.name} (checking again in {polling_interval}s)")

        if video_file.state.name == "FAILED":
            st.error("🚨 Video processing failed on the server.")
            os.unlink(tmp_file_path) # Clean up temp file
            return None

        os.unlink(tmp_file_path) # Clean up the temporary file after successful upload and processing
        st.success(f"✅ Video '{uploaded_file.name}' is ready for analysis!")
        return video_file

    except Exception as e:
        st.error(f"🚨 An error occurred during video upload or processing: {e}")
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path) # Ensure cleanup on error
        return None


def get_gemini_response(video_file_obj, user_prompt):
    """Generates content using Gemini based on video and text prompt."""
    if not video_file_obj:
        return "Please upload a video first."

    system_prompt = """
    Analyze the uploaded video for content and context. As an fitness trainers you are expert in understanding healthcare domain including medical, medicines, fitness, sports and Brahmacharya.
    You have extreme deep knowledge of medical sciences, Brahmacharya, physiological and psychological fitness, spiritual and ayurveda.
    We will ask you many questions on health domain and you will have to answer any questions based on the any types of
    resolutions of uploaded video in medical, scientific and evidence-based in between 1000 to 10000 numbers of words.

    Disclaimer: This AI Fitness Trainer application provides information and recommendations for general fitness and wellness purposes only.
    It is not intended legally,and should not be used as, a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider physically with any questions you may have regarding
    a medical condition for getting better benefits.
    Never disregard professional medical advice or delay in seeking it because of something you have read or received through this application.


    If question is not related to health domain then say just this line : "Sorry, I am an AI fitness trainer, I can only answer questions
    related to health domain. Please ask a question related to health domain."

    if the answer is not in provided uploaded images just say, "answer is not available in the uploaded videos", don't provide the wrong answer
    Respond to the following query using video insights and supplementary web research:
                        

    Provide a detailed, user-friendly, and actionable response.
    """

    try:
        st.info("🧠 Thinking...")
        # Combine the system prompt, the uploaded video file object, and the user's question
        response = model.generate_content(
            [system_prompt, video_file_obj, user_prompt],
            request_options={"timeout": 600} # Increased timeout for potentially long video analysis
        )
        
        # Check for safety ratings or blocked responses
        if not response.candidates:
             return "⚠️ The model could not generate a response. This might be due to safety filters or limitations. Please try a different question or video."
        
        # Check safety ratings if needed (optional refinement)
        # safety_ratings = response.prompt_feedback.safety_ratings
        # print(f"Safety Ratings: {safety_ratings}")

        return response.text

    except Exception as e:
        st.error(f"🚨 An error occurred while generating the response: {e}")
        # Attempt to delete the file from Gemini server if analysis fails midway
        # This might fail if the object wasn't fully created or due to permissions
        # try:
        #     if video_file_obj:
        #         genai.delete_file(video_file_obj.name)
        #         st.info(f"Cleaned up file {video_file_obj.name} from server due to error.")
        # except Exception as del_e:
        #     st.warning(f"Could not delete file {video_file_obj.name} after error: {del_e}")
        return "Sorry, I encountered an error trying to analyze the video with your question."


# --- Streamlit UI Elements ---

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_video_file_obj" not in st.session_state:
    st.session_state.uploaded_video_file_obj = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0 # To reset file uploader

# Sidebar for Upload
with st.sidebar:
    st.header("Upload Video")
    # Use the key to allow re-uploading the same file name
    uploaded_file = st.file_uploader(
        "Choose a video file...",
        type=["mp4", "mov", "avi", "mpg", "mpeg", "wmv"],
        key=f"file_uploader_{st.session_state.upload_key}"
    )

    if uploaded_file is not None:
        # Check if it's a new file or the same file being reprocessed
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.messages = [] # Reset chat on new file upload
            st.session_state.uploaded_file_name = uploaded_file.name
            with st.spinner("Processing video... This may take a few minutes."):
                # Upload and process the video, store the resulting file object
                st.session_state.uploaded_video_file_obj = upload_video_to_gemini(uploaded_file)

            # Increment key to allow re-uploading the same file again if needed
            st.session_state.upload_key += 1
            st.rerun() # Rerun to update UI state after upload completes/fails

        # Display video preview in sidebar if successfully processed
        if st.session_state.uploaded_video_file_obj:
            st.video(uploaded_file)
            st.success(f"Ready to analyze: **{st.session_state.uploaded_file_name}**")
        else:
            st.error("Video processing failed or was cancelled. Please try uploading again.")
            # Clean up state if processing failed
            st.session_state.uploaded_file_name = None


    elif st.session_state.uploaded_video_file_obj:
        # If no file is currently uploaded, but one was processed before, show status
        st.success(f"Ready to analyze: **{st.session_state.uploaded_file_name}**")
        st.info("Ask questions about the video in the main chat area.")


# Main Chat Area
st.header("Chat about the Video")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the video..."):
    # Check if video is ready
    if not st.session_state.uploaded_video_file_obj:
        st.warning("Please upload and process a video first using the sidebar.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get Gemini response
        response_text = get_gemini_response(st.session_state.uploaded_video_file_obj, prompt)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response_text)

# --- Optional: Button to clear chat and video ---
if st.sidebar.button("Clear Chat & Remove Video"):
    st.session_state.messages = []
    st.session_state.uploaded_file_name = None
    # Attempt to delete the file from Gemini server (optional cleanup)
    if st.session_state.uploaded_video_file_obj:
         try:
            # Check the state before attempting deletion
            file_to_delete = genai.get_file(st.session_state.uploaded_video_file_obj.name)
            if file_to_delete.state.name == "ACTIVE":
                 genai.delete_file(st.session_state.uploaded_video_file_obj.name)
                 st.sidebar.success(f"Removed file '{st.session_state.uploaded_video_file_obj.name}' from server.")
            else:
                 st.sidebar.warning(f"File '{st.session_state.uploaded_video_file_obj.name}' state is {file_to_delete.state.name}, skipping deletion.")
         except Exception as del_e:
             st.sidebar.warning(f"Could not remove file from server: {del_e}")
    st.session_state.uploaded_video_file_obj = None
    st.session_state.upload_key += 1 # Reset file uploader state
    st.rerun()