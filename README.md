# AI-Project
# 🧠 AI-based Mental Health Analysis Tool

A Streamlit-based web application that analyzes chat conversations and screenshots for potential mental health indicators using AI and OCR technology.

## 🌟 Features

- 📸 Upload chat screenshots for analysis
- 🔍 Automatic text extraction using OCR
- 🤖 AI-powered mental health analysis
- 📊 Visual representation of analysis results
- 💡 Personalized suggestions based on analysis
- 📝 Save and download analysis reports
- 👤 User profile management
- 🎨 Modern and intuitive UI with animations

## 🛠️ Prerequisites

- Python 3.7 or higher
- Tesseract OCR
- Required Python packages (listed in requirements.txt)

## ⚙️ Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd mental_health_analysis_tool
```

2. Install Tesseract OCR:
   - Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
   - Choose the appropriate version (64-bit recommended)
   - Run the installer
   - Add Tesseract to your system PATH:
     - Open System Properties (Win + Pause/Break)
     - Click "Advanced system settings"
     - Click "Environment Variables"
     - Under "System variables", find and select "Path"
     - Click "Edit"
     - Click "New"
     - Add the Tesseract installation path (e.g., `C:\Program Files\Tesseract-OCR`)
     - Click "OK" on all windows

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (usually http://localhost:8501)

3. Use the application:
   - Enter your details in the sidebar
   - Upload chat screenshots
   - View analysis results
   - Download reports if needed

## 📋 Features in Detail

### User Profile
- Enter your name, age, and gender
- Option to share previous mental health history
- Save and download user details

### Image Analysis
- Supports PNG, JPG, and JPEG formats
- Automatic text extraction using OCR
- Real-time processing with visual feedback

### Analysis Results
- Detailed mental health condition assessment
- Visual representation of scores
- Personalized suggestions based on findings
- Option to download comprehensive reports

## 🛡️ Privacy and Security

- All processing is done locally
- No data is stored permanently
- Users can download and delete their reports
- No personal information is shared with third parties

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit for the web framework
- Tesseract OCR for text extraction
- All contributors and users of this tool

## 📞 Support

For support, please open an issue in the repository or contact the maintainers.

---

Made with ❤️ for mental health awareness 
