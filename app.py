import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace

# Function to analyze facial attributes using DeepFace
def analyze_frame(frame):
    result = DeepFace.analyze(img_path=frame, actions=['age', 'gender', 'race', 'emotion'],
                              enforce_detection=False,
                              detector_backend="opencv",
                              align=True,
                              silent=False)
    return result

def overlay_text_on_frame(frame, texts):
    overlay = frame.copy()
    alpha = 0.9  # Adjust the transparency of the overlay
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (255, 255, 255), -1)  # White rectangle
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    text_position = 15  # Where the first text is put into the overlay
    for text in texts:
        cv2.putText(frame, text, (10, text_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        text_position += 20

    return frame

def facesentiment():
    cap = cv2.VideoCapture(0)
    stframe = st.image([])  # Placeholder for the webcam feed

    while True:
        ret, frame = cap.read()

        # Analyze the frame using DeepFace
        result = analyze_frame(frame)

        # Extract the face coordinates
        face_coordinates = result[0]["region"]
        x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['w'], face_coordinates['h']

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{result[0]['dominant_emotion']}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Convert the BGR frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Overlay white rectangle with text on the frame
        texts = [
            f"Age: {result[0]['age']}",
            f"Face Confidence: {round(result[0]['face_confidence'], 3)}",
            f"Gender: {result[0]['dominant_gender']} ({round(result[0]['gender'][result[0]['dominant_gender']], 3)})",
            f"Race: {result[0]['dominant_race']}",
            f"Dominant Emotion: {result[0]['dominant_emotion']} ({round(result[0]['emotion'][result[0]['dominant_emotion']], 1)})",
        ]

        frame_with_overlay = overlay_text_on_frame(frame_rgb, texts)

        # Display the frame in Streamlit
        stframe.image(frame_with_overlay, channels="RGB")

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def project_description():
    st.title("Facial Emotion Recognition System")
    st.markdown(
        """<style> 
        .title {
            font-size: 24px; 
            font-weight: bold;
            color: #2E8B57;
        }
        .content {
            font-size: 18px; 
            line-height: 1.6;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='title'>The Problem</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='content'>"
        "This application addresses the need for real-time facial emotion recognition, which can be used in areas like human-computer interaction, security, and customer behavior analysis."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='title'>Approach</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='content'>"
        "We use DeepFace, a state-of-the-art facial attribute analysis library, combined with OpenCV for webcam access and Streamlit for the user interface."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='title'>System Workflow</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='content'>"
        "The system captures real-time video feed from the webcam, analyzes each frame using DeepFace to detect age, gender, race, and emotion, and overlays the results on the video feed."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='title'>Challenges and Solutions</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='content'>"
        "Challenges included ensuring real-time performance and handling variations in lighting and face angles. These were addressed by optimizing DeepFace configurations and using the OpenCV backend for face detection."
        "</div>",
        unsafe_allow_html=True,
    )

def main():
    st.sidebar.title("Navigation")
    activities = ["Project Description", "Webcam Face Detection", "About"]
    choice = st.sidebar.radio("Select Activity", activities)

    st.sidebar.markdown(
        """<style> 
        .footer {
            font-size: 14px; 
            font-weight: normal;
            color: #808080;
            text-align: center;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """<div class='footer'>
        Developed by Koushik Balla, Chris Alister, Manoj Kumar D.<br>
        Reg No: 2347217, 2347218, 2347237
        </div>""",
        unsafe_allow_html=True,
    )

    if choice == "Webcam Face Detection":
        st.markdown(
            """<div style="background-color:#4682B4;padding:10px">
            <h2 style="color:white;text-align:center;">
            Real-time Face Emotion Recognition
            </h2></div><br>""",
            unsafe_allow_html=True,
        )
        facesentiment()

    elif choice == "About":
        st.subheader("About this app")
        st.markdown(
            """<div style="background-color:#B0C4DE;padding:10px">
            <h4 style="color:black;text-align:center;">This application is developed by a dedicated team. Thanks for visiting!</h4>
            </div>""",
            unsafe_allow_html=True,
        )

    elif choice == "Project Description":
        project_description()

if __name__ == "__main__":
    main()
