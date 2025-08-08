# AI-PPT-generation
I have done the PPT generation application, in this firstly I have to upload the topic name or the pdf of the topic on which I want to make a ppt then click on generate the ppt button and get ppt easlily, I can download this ppt in pdf

# 📊 AI PDF Presentation Generator

Create beautiful, professional-grade PDF presentations enhanced with AI-generated content, relevant images, additional insights, and references — all from a topic or uploaded document. Powered by **Google Gemini** and built using **Streamlit** and **ReportLab**.



---

## 🚀 Features

- 🔥 Generate 6–10 slide presentations from a **topic** or **uploaded document**
- 📄 Supports PDF, DOCX, TXT, and CSV files
- 🧠 Uses **Google Gemini AI** to:
  - Extract key ideas
  - Generate bullet points and explanations
  - Suggest images and references
- 🖼️ Automatically downloads and embeds suggested images
- 📎 Includes source links and further reading
- 📥 One-click **PDF download** of your enhanced presentation

---

## 🧰 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **AI:** [Google Gemini (via GenerativeAI SDK)](https://makersuite.google.com/app/apikey)
- **PDF Generation:** [ReportLab](https://www.reportlab.com/)
- **Data Handling:** pandas, PyPDF2, python-docx

---

## 🔧 Installation

Clone this repo and install the dependencies:

```bash
git clone https://github.com/yourusername/ai-pdf-presentation-generator.git
cd ai-pdf-presentation-generator
pip install -r requirements.txt
Create a .env file or directly paste your Google Gemini API key in the app when prompted.

▶️ Running the App
bash
Copy
Edit
streamlit run app.py
Then open your browser at http://localhost:8501

🔑 Get Your Google Gemini API Key
Go to Google AI Studio

Sign in with your Google account

Generate an API Key

Paste it into the app when prompted

📸 Screenshots
Topic Input	Upload Document	Slide Preview	Download PDF

📁 Supported File Types
.pdf

.docx

.txt

.csv (data summary will be generated)

✨ Example Use Cases
Academic presentations

Business proposals

Data analysis reports

Research summaries

Quick topic overviews

✅ To-Do / Enhancements
 Export to PowerPoint

 Support for Markdown input

 Theme customization (colors, fonts)

 Editable slide previews

🤝 Contributing
Pull requests and suggestions are welcome!

Fork the repo

Create a branch (git checkout -b feature-name)

Commit changes (git commit -m 'Add feature')

Push (git push origin feature-name)

Open a Pull Request

📄 License
This project is licensed under the MIT License. See LICENSE file for details.

💬 Questions?
Open an issue or contact me directly.

🌐 Connect
LinkedIn

Twitter

Portfolio

Built with ❤️ using AI and Streamlit.

vbnet
Copy
Edit

---

### ✅ Next Steps:
- Replace placeholders (like GitHub repo URL, image links, and social media) with your actual info.
- Rename your main script file to `app.py` if it's not already.
- Create a `requirements.txt` file (if you need help, let me know).
- Add a license (MIT is a good default) in `LICENSE`.

Let me know if you want a logo, banner, or deployment help (e.g., Streamlit Cloud, Render, Hugging Face, etc.).
