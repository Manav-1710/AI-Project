import streamlit as st
from PIL import Image
import pytesseract
import io
import time
import matplotlib.pyplot as plt
import pandas as pd

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Dummy AI model for demonstration
def predict_mental_health(text):
    # Simple keyword-based prediction for demo purposes
    keywords = {
        'depression': ['sad', 'hopeless', 'tired', 'unhappy', 'depressed'],
        'anxiety': ['worried', 'nervous', 'anxious', 'panic', 'fear'],
        'stress': ['stress', 'overwhelmed', 'pressure', 'tense'],
        'bipolar disorder': ['mood swings', 'manic', 'depressed', 'highs and lows'],
        'eating disorder': ['diet', 'weight', 'food', 'binge', 'purge'],
        'obsessive-compulsive disorder': ['obsessive', 'compulsive', 'ritual', 'check', 'repeat'],
        'post-traumatic stress disorder': ['trauma', 'flashback', 'nightmare', 'ptsd', 'panic']
    }
    text_lower = text.lower()
    scores = {k:0 for k in keywords.keys()}
    for condition, kw_list in keywords.items():
        for kw in kw_list:
            if kw in text_lower:
                scores[condition] += 1
    # Find condition with max score
    max_condition = max(scores, key=scores.get)
    if scores[max_condition] == 0:
        return "No significant signs detected", scores
    else:
        return f"Signs of {max_condition.capitalize()} detected", scores

def main():
    st.set_page_config(page_title="AI Mental Health Analysis Tool", page_icon="🧠", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        @keyframes slideIn {
            from {transform: translateY(20px); opacity: 0;}
            to {transform: translateY(0); opacity: 1;}
        }
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {
                transform: translateY(0);
            }
            40% {
                transform: translateY(-15px);
            }
            60% {
                transform: translateY(-7px);
            }
        }
        @keyframes zoomIn {
            from {
                opacity: 0;
                transform: scale(0.8);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        @keyframes rotateIn {
            from {
                transform: rotate(-200deg);
                opacity: 0;
            }
            to {
                transform: rotate(0deg);
                opacity: 1;
            }
        }
        @keyframes typing {
            from { width: 0 }
            to { width: 100% }
        }
        @keyframes blinkCaret {
            50% { border-color: transparent; }
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes growBar {
            from { width: 0; }
            to { width: 100%; }
        }
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 5px #007bff; }
            50% { box-shadow: 0 0 20px #00d4ff; }
        }
        .main {
            background: linear-gradient(-45deg, #f0f4f8, #d9e2ec, #bcccdc, #f0f4f8);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            font-family: 'Roboto', sans-serif;
            animation-fill-mode: forwards;
            color: #333333;
            padding: 20px;
            border-radius: 15px;
        }
        .card {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-bottom: 25px;
            animation: slideIn 1s ease forwards;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px) scale(1.05) rotate(1deg);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }
        .title {
            font-size: 2.8rem;
            font-weight: 700;
            color: #222222;
            text-align: center;
            margin-bottom: 25px;
            white-space: nowrap;
            overflow: hidden;
            border-right: none;
            width: 100%;
            animation: typing 3s steps(30, end) forwards;
            text-shadow: none;
        }
        .sidebar-image {
            display: block;
            margin: 10px auto 20px auto;
            width: 80%;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            animation: zoomIn 1.5s ease forwards;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .sidebar-image:hover {
            transform: scale(1.05) rotate(3deg);
            box-shadow: 0 0 15px #00d4ff;
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }
        .footer {
            text-align: center;
            font-style: italic;
            color: #555555;
            margin-top: 40px;
            padding: 10px;
            border-top: 1px solid #ddd;
            font-size: 1rem;
            font-weight: 400;
            text-shadow: none;
        }
        .btn-primary:hover {
            background-color: #007bff !important;
            color: white !important;
            box-shadow: 0 0 10px #00d4ff;
            transition: background-color 0.3s ease, box-shadow 0.4s ease;
        }
        .btn-primary {
            animation: rotateIn 1s ease forwards;
            transition: transform 0.3s ease;
        }
        .btn-primary:hover {
            transform: scale(1.1) rotate(5deg);
            transition: transform 0.4s ease;
        }
        .loading-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 10px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .bar-animate {
            animation: growBar 1s ease forwards;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    st.sidebar.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=400&q=80", caption="Mental Health Awareness", clamp=True, output_format="auto", channels="RGB", width=300)
    st.sidebar.title("About")
    st.sidebar.info(
        "Hey! This cool AI tool checks your chat screenshots for mental health vibes. Upload an image and see what's up! 🚀✨"
    )
    st.sidebar.title("User Details")
    name = st.sidebar.text_input("Your Name")
    age = st.sidebar.number_input("Your Age", min_value=18, max_value=100, step=1)
    gender = st.sidebar.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
    pre_existing_issues_option = st.sidebar.selectbox("Do you have any prior or pre-existing issues about your mental health?", ["No", "Yes"])
    mental_health_status = ""
    if pre_existing_issues_option == "Yes":
        mental_health_status = st.sidebar.text_area("Please share your previous or pre-existing thoughts or issues about your mental health.")

    # Remove dark mode toggle and mental health tips carousel

    # Remove government ID input and related logic
    # Add save user details button without government ID
    if age or not age:
        if st.sidebar.button("Save User Details"):
            user_details = f"User Details\n\nName: {name}\nAge: {age}\nGender: {gender}\nPrevious Mental Health Thoughts/Issues: {mental_health_status}\n"
            user_details_bytes = user_details.encode('utf-8')
            st.sidebar.download_button(label="Download User Details as TXT", data=user_details_bytes, file_name="user_details.txt", mime="text/plain")

    st.markdown('<h1 class="title">🧠 AI-based Mental Health Analysis Tool</h1>', unsafe_allow_html=True)
    st.write("Upload screenshots of your chat conversations to analyze mental health signs.")

    # Add animated GIF below title
    st.markdown(
        """
        <div style="text-align:center;">
            <img src="https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif" alt="Brain Animation" width="200" />
        </div>
        """,
        unsafe_allow_html=True
    )

    # Remove dark mode styles application

    # User details display
    if name:
        st.markdown(f"### Hello, {name}! 👋")
        st.markdown(f"**Age:** {age}  |  **Gender:** {gender}")
        if mental_health_status:
            st.markdown(f"**Previous Mental Health Thoughts/Issues:** {mental_health_status}")

    # Add footer with motivational quote
    st.markdown(
        """
        <div class="footer">
            "Mental health is not a destination, but a process. It's about how you drive, not where you're going." – Noam Shpancer
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1,1])

        with col1:
            st.markdown('<div class="card"><h4>Uploaded Image</h4></div>', unsafe_allow_html=True)
            st.image(image)

        with col2:
            with st.spinner('Extracting text from image...'):
                st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
                time.sleep(1)  # simulate delay
                extracted_text = pytesseract.image_to_string(image)

            st.markdown('<div class="card"><h4>Extracted Text</h4></div>', unsafe_allow_html=True)
            st.text_area("Text extracted from the image:", extracted_text, height=200)

            if extracted_text.strip():
                with st.spinner('Analyzing mental health status...'):
                    st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
                    time.sleep(1)  # simulate delay
                    prediction, scores = predict_mental_health(extracted_text)

                st.markdown('<div class="card"><h4>Mental Health Analysis Result</h4></div>', unsafe_allow_html=True)
                st.success(prediction)
                st.balloons()

                # Show bar chart of scores with animation
                df_scores = pd.DataFrame(list(scores.items()), columns=['Condition', 'Score'])
                fig, ax = plt.subplots()
                bars = ax.barh(df_scores['Condition'], df_scores['Score'], color='skyblue')
                ax.set_xlabel('Score')
                ax.set_title('Condition Scores')
                for bar in bars:
                    bar.set_width(0)
                def animate_bars():
                    for i in range(1, 101):
                        for bar, score in zip(bars, df_scores['Score']):
                            bar.set_width(score * i / 100)
                        fig.canvas.draw_idle()
                        time.sleep(0.01)
                animate_bars()
                st.pyplot(fig)

                # Suggestions based on prediction
                suggestions = {
                    'depression': "Consider seeking support from a mental health professional and talking to trusted friends or family.",
                    'anxiety': "Practice relaxation techniques and consider consulting a therapist for anxiety management.",
                    'stress': "Try stress management techniques like exercise, meditation, and time management.",
                    'bipolar disorder': "Consult a psychiatrist for diagnosis and treatment options.",
                    'eating disorder': "Seek help from healthcare providers specializing in eating disorders.",
                    'obsessive-compulsive disorder': "Cognitive-behavioral therapy (CBT) can be effective; consider professional help.",
                    'post-traumatic stress disorder': "Therapy and support groups can help; consult a mental health professional."
                }
                for condition in suggestions.keys():
                    if condition in prediction.lower():
                        st.info("Helpful Suggestion:")
                        st.write(suggestions[condition])
                        break

                # Option to save report
                if st.button("Save Report"):
                    report = f"Mental Health Analysis Report\n\nUser: {name}\nAge: {age}\nGender: {gender}\n\nExtracted Text:\n{extracted_text}\n\nPrediction:\n{prediction}\n"
                    report_bytes = report.encode('utf-8')
                    st.download_button(label="Download Report as TXT", data=report_bytes, file_name="mental_health_report.txt", mime="text/plain")

if __name__ == "__main__":
    main()
