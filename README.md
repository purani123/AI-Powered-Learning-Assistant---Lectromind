# AI-Powered-Learning-Assistant--Lectromind

![Lectromind AI Assistant Banner](https://placehold.co/800x200/1F2F4F/E0E0E0?text=Lectromind+AI+Assistant)

## Project Title: AI Lecture Assistant for Classrooms

**Developed by Team Lectromind**

### Project Overview

The AI Lecture Assistant is an innovative web application designed to enhance the classroom learning experience. It automates lecture note-taking, provides intelligent comprehension assessment, and offers personalized learning support.

Developed as part of the **Intel® Unnati Industrial Training Program 2024**, this project leverages cutting-edge AI to make learning more interactive and efficient.

### Key Features

* **Automated Lecture Transcription:** Converts live or uploaded audio into text using **Whisper ASR**.
* **Intelligent Summarization:** Condenses transcripts into concise summaries.
* **Interactive Brain Breaks:** Provides short, engaging video breaks between lecture segments.
* **Dynamic AI-Powered Content:** Utilizes **Google Gemini API** for:
    * Key concept extraction and definitions.
    * Context-aware Q&A Chatbot.
    * Generative, relevant Multiple-Choice Quizzes.
* **Gamification:** Awards points and badges for engagement.
* **Teacher Insights Dashboard:** Logs student performance and interactions for educators.
* **User-Friendly Interface:** Built with **Streamlit** featuring a modern lavender, silver, and navy blue theme.

### Technologies Used

* **Python**
* **Streamlit**
* **Google Gemini API (`google-generativeai`)**
* **Whisper (`openai-whisper`)**
* **Hugging Face Transformers** (T5-small, DistilBERT)
* **Sounddevice & SciPy**
* **FFmpeg** (external dependency)
* **SQLite3**
* **Wikipedia API**

### Setup and Installation

Follow these steps to get the AI Lecture Assistant running on your local machine.

#### 1. Prerequisites

* **Python 3.10+:** (Ensure "Add python.exe to PATH" during installation).
* **FFmpeg:**
    * Download `ffmpeg-release-full.zip` from [Gyan's Builds](https://www.gyan.dev/ffmpeg/builds/).
    * Extract to `C:\ffmpeg_tools` (or similar non-synced path).
    * Add `C:\ffmpeg_tools\ffmpeg-X.X-full_build\bin` (replace X.X with version) to your system's PATH environment variables.
    * Verify: `ffmpeg -version` in a new terminal.
* **Git:** Install from [git-scm.com](https://git-scm.com/download/win).

#### 2. Clone the Repository

Navigate to a non-OneDrive synced directory (e.g., `C:\Projects\`) in your terminal and clone the repository:

```bash
git clone [https://github.com/purani123/AI-Powered-Learning-Assistant---Lectromind.git](https://github.com/purani123/AI-Powered-Learning-Assistant---Lectromind.git)
cd AI-Powered-Learning-Assistant---Lectromind
