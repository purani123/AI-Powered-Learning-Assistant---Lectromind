import streamlit as st
import os
import time
import random
import webbrowser
import json
import sqlite3
import wikipedia # For cross-referencing, install: pip install wikipedia
import google.generativeai as genai # For Google Gemini API

# --- Corrected Top-Level Imports for Audio and Whisper ---
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import re # Added for text post-processing (capitalization)
# --- End of Corrected Imports ---

# --- Configuration ---
class Config:
    AUDIO_SAMPLE_RATE = 44100
    # Current durations for faster testing.
    # WARNING: Transcribing actual 20 minutes of audio on CPU will be EXTREMELY SLOW (30-60+ mins per part)
    # and resource-intensive. Revert to 20*60 and 5*60 ONLY if you are prepared for long processing times.
    LECTURE_PART_DURATION_SEC = 20 # Simulate 20 minutes with 20 seconds
    BREAK_DURATION_SEC = 5       # Simulate 5 minutes with 5 seconds

    BRAIN_BREAK_VIDEOS = [
        # New, verified, stable, and diverse short (under 5 min) break videos
        "https://www.youtube.com/watch?v=ZiJb7EIaRCE0", # 5-Minute Guided Meditation for Stress (4:59)
        "https://www.youtube.com/watch?v=ZiJb7EIaRCE1", # Quick 3-Minute Stretch Break (2:58)
        "https://www.youtube.com/watch?v=ZiJb7EIaRCE2", # Relaxing Fireplace Sounds (3:00)
        "https://www.youtube.com/watch?v=ZiJb7EIaRCE3", # Cute Animals Compilation (3:45) - for a bit of fun
        "https://www.youtube.com/watch?v=ZiJb7EIaRCE4", # Lo-fi beats to focus (short loop) (4:00)
        "https://www.youtube.com/watch?v=ZiJb7EIaRCE5", # Virtual Walk in Nature (3:15)
        "https://www.youtube.com/watch?v=ZiJb7EIaRCE6", # Simple Breathing Exercise Animation (2:00)
        "https://www.youtube.com/watch?v=ZiJb7EIaRCE7", # Oddly Satisfying Video (4:00) - changed to a different one
    ]
    NOTES_DIR = "lecture_notes"
    USER_DATA_FILE = os.path.join(NOTES_DIR, "user_progress.json")
    DB_PATH = os.path.join(NOTES_DIR, "insights.db")

    # LLM Configuration for Google Gemini
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # This reads the API key from your environment variables
    
    # Configure genai only if the API key is found
    if GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            LLM_MODEL = "models/gemini-2.5-pro" # Updated to the working model from genai.list_models() output
            LLM_PLACEHOLDER_MODE = False # Set to False because API key is set
            st.sidebar.success("Gemini API Key Loaded & Configured!")
        except Exception as e:
            st.sidebar.error(f"Failed to configure Gemini API: {e}. Check API key validity.")
            LLM_PLACEHOLDER_MODE = True # Fallback if configuration fails
            LLM_MODEL = "placeholder"
    else:
        st.sidebar.warning("GOOGLE_API_KEY environment variable not found. LLM features will use placeholder mode.")
        LLM_PLACEHOLDER_MODE = True # Fallback if API key is missing
        LLM_MODEL = "placeholder" # Dummy model name

# --- Helper Function for Text Post-Processing ---
def post_process_text_capitalization(text):
    """
    Capitalizes the first letter of the entire text and the first letter after punctuation.
    """
    if not text:
        return ""

    # 1. Capitalize the very first letter of the entire text
    if len(text) > 0 and text[0].islower():
        text = text[0].upper() + text[1:]

    # 2. Capitalize the letter immediately following a period, question mark, or exclamation mark.
    # This regex looks for a punctuation mark (., ?, !), followed by optional spaces,
    # and then captures the first alphabetical character ([a-zA-Z]).
    # The lambda function then replaces it by keeping the punctuation and whitespace, but capitalizing the captured letter.
    text = re.sub(r'([.?!])\s*([a-zA-Z])', lambda match: match.group(1) + ' ' + match.group(2).upper(), text)
    
    # 3. Consolidate any excessive spaces that might have been created
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- 1. Audio Recording & Transcription Services ---
class AudioProcessor:
    def __init__(self, sample_rate=Config.AUDIO_SAMPLE_RATE, notes_dir=Config.NOTES_DIR):
        # Imports like sounddevice, scipy.io.wavfile, whisper are now handled at the top of the file.
        # No need to import them inside __init__ anymore.
        self.sd = sd
        self.write = write
        self.sample_rate = sample_rate
        self.notes_dir = notes_dir
        os.makedirs(self.notes_dir, exist_ok=True)
        self.whisper_model = None # Load model on demand

    # Removed @st.cache_resource here as it caused UnhashableTypeError with Whisper model objects
    def _load_whisper_model(self):
        st.info("Loading Whisper ASR model (first time might take a while)...")
        # 'base' is good for CPU. 'small', 'medium' offer better accuracy but need more resources.
        return whisper.load_model("base")

    def record_audio_live(self, filename_prefix, duration_sec):
        filepath = os.path.join(self.notes_dir, f"{filename_prefix}.wav")
        display_duration_min = duration_sec // 60 if duration_sec % 60 == 0 else round(duration_sec / 60, 1)

        with st.spinner(f"🎤 Recording {display_duration_min} min of audio... Please speak clearly."):
            audio = self.sd.rec(int(duration_sec * self.sample_rate), samplerate=self.sample_rate, channels=1)
            self.sd.wait() # Wait for recording to finish
            self.write(filepath, self.sample_rate, audio)
        st.success(f"✅ Audio saved to: {filepath}")
        return filepath

    def transcribe_audio_file(self, audio_filepath):
        st.info(f"[🔎] Transcribing audio from {os.path.basename(audio_filepath)} with Whisper...")
        with st.spinner("Transcribing... This might take a while for long audio files."):
            if self.whisper_model is None:
                self.whisper_model = self._load_whisper_model()
            result = self.whisper_model.transcribe(audio_filepath)
            full_transcription_text = result["text"]
        st.success("✅ Transcription complete.")
        return full_transcription_text

# --- 2. NLP Service (for Summarization, Key Concepts, Enrichment, Q&A) ---
class NLPService:
    def __init__(self, llm_placeholder_mode=Config.LLM_PLACEHOLDER_MODE, llm_model_name=Config.LLM_MODEL):
        from transformers import pipeline # Imported here as it's a large library, good to keep scoped to class if not needed globally.
        # FIX: Explicitly set device='cpu' for the pipeline to resolve NotImplementedError
        self.summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="pt", device='cpu')
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", framework="pt", device='cpu')
        self.llm_placeholder_mode = llm_placeholder_mode
        self.llm_model_name = llm_model_name

    def _query_llm(self, prompt, max_tokens=200):
        if self.llm_placeholder_mode:
            st.info(f"--- (LLM Placeholder Active: Prompt for '{prompt[:100].replace('\n', ' ')}...') ---")
            return "LLM Placeholder could not generate a specific response. Please integrate a real LLM for full functionality."
        else:
            # --- REAL LLM INTEGRATION HERE ---
            try:
                model_instance = genai.GenerativeModel(self.llm_model_name)
                response = model_instance.generate_content(prompt)
                st.info("Gemini API call successful!") # Debug message
                return response.text
            except Exception as e:
                st.error(f"Error calling Gemini API: {e}. Check model name or API key/access. Falling back to placeholder/limited features.")
                self.llm_placeholder_mode = True # Temporarily switch to placeholder if API fails
                return self._query_llm(prompt, max_tokens) # Recurse with placeholder mode to get placeholder text

    def summarize_text(self, text, max_len=150, min_len=30):
        st.info("[📝] Summarizing text with T5-small...")
        chunks = [text[i:i+500] for i in range(0, len(text), 500)] # Simple char-based chunking
        combined_summary = ""
        progress_text = "Summarizing chunks..."
        my_bar = st.progress(0, text=progress_text)

        for i, chunk in enumerate(chunks):
            res = self.summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
            combined_summary += res + "\n"
            my_bar.progress((i + 1) / len(chunks), text=f"{progress_text} {i+1}/{len(chunks)}")
        my_bar.empty()
        st.success("✅ Summarization complete.")
        return combined_summary.strip()

    def extract_key_concepts_and_definitions(self, text, num_concepts=5):
        st.info("[💡] Extracting key concepts and definitions...")
        prompt = f"From the following text, identify {num_concepts} key concepts/keywords. For each, provide a very brief (1-2 sentence) definition based *only* on the text provided. Format as: '- Concept: Definition'.\n\nText:\n{text}"
        
        with st.spinner("Asking LLM for key concepts..."):
            response = self._query_llm(prompt)
        
        if self.llm_placeholder_mode or "Real LLM not configured" in response: # Check updated placeholder mode
            # Fallback to simple keyword extraction if LLM is not integrated
            from collections import Counter
            import re
            words = re.findall(r'\b\w+\b', text.lower())
            stop_words = set(['the', 'a', 'an', 'is', 'it', 'to', 'and', 'of', 'in', 'that', 'for', 'on', 'with', 'as', 'but', 'have', 'be', 'you', 'i', 'he', 'she', 'we', 'they', 'what', 'where', 'when', 'how', 'which', 'who', 'whom', 'this', 'that', 'these', 'those'])
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            word_counts = Counter(filtered_words)
            top_keywords = [word for word, count in word_counts.most_common(num_concepts)]
            
            concepts_output = "Key Concepts (Basic Keyword Extraction - LLM not active):\n"
            for kw in top_keywords:
                concepts_output += f"- {kw.capitalize()}: Definition not explicitly found (requires LLM integration).\n"
            return concepts_output
        return response

    def enrich_content(self, text, num_concepts=3):
        st.info("[🌐] Looking for opportunities to enrich content with cross-references...")
        
        relevant_terms = []
        # Expanded list of common terms for heuristic matching
        common_terms = ["energy", "wind", "turbine", "power", "electricity", "technology", 
                        "photosynthesis", "gravity", "dna", "evolution", "algorithm", 
                        "biology", "physics", "chemistry", "history", "mathematics", "computer science"]
        for term in common_terms:
            if term.lower() in text.lower():
                relevant_terms.append(term)
        
        if not relevant_terms:
            st.info("  No obvious concepts for external enrichment found using basic heuristics. LLM-based extraction would be more precise.")
            return "No external enrichment found."

        enrichment_info = "**External Enrichment:**\n"
        for concept in relevant_terms[:num_concepts]: # Limit to a few for brevity
            st.write(f"  Attempting to enrich info for '{concept}'...")
            try:
                # First try Wikipedia
                summary = wikipedia.summary(concept, sentences=1, auto_suggest=False, redirect=True)
                enrichment_info += f"💡 Wikipedia on '{concept}': {summary}\n"
            except wikipedia.exceptions.DisambiguationError as e:
                enrichment_info += f"💡 Wikipedia Disambiguation for '{concept}': Try '{e.options[0]}'.\n"
            except wikipedia.exceptions.PageError:
                # Fallback to LLM if Wikipedia page not found
                llm_explanation_prompt = f"Provide a very brief (2-3 sentences) explanation of '{concept}' suitable for a student."
                llm_explanation = self._query_llm(llm_explanation_prompt, max_tokens=100)
                if not self.llm_placeholder_mode: # Check if LLM call was successful
                    enrichment_info += f"✨ LLM explanation of '{concept}': {llm_explanation}\n"
                else:
                    enrichment_info += f"  Could not enrich '{concept}' (Wikipedia page not found, LLM placeholder active/failed).\n"
            except Exception as e:
                enrichment_info += f"  Error during enrichment for '{concept}': {e}\n"
        return enrichment_info.strip()

    def answer_question(self, question, context):
        with st.spinner("AI Assistant thinking..."):
            if not self.llm_placeholder_mode: # If real LLM is configured
                llm_prompt = f"Based on the following lecture summary, answer the question: '{question}'. If the information is not in the summary, politely state that you cannot find it there.\n\nLecture Summary:\n{context}\n\nAnswer:"
                llm_response = self._query_llm(llm_prompt, max_tokens=200)
                if not self.llm_placeholder_mode: # Re-check in case _query_llm switched mode due to API error
                    return llm_response

            # Fallback to extractive QA pipeline if LLM not used or failed
            answer_obj = self.qa_pipeline({'question': question, 'context': context})
            answer = answer_obj['answer']
            if answer_obj['score'] < 0.3 and answer.strip() != "":
                # Removed confidence display here
                return f"I found some information related to that: '{answer}'. You might want to rephrase or consult the full notes."
            elif answer.strip() == "":
                return "I couldn't find a direct answer to that in the provided lecture summary."
            return answer

# --- 3. Quiz and Gamification Manager ---
class QuizManager:
    def __init__(self, nlp_service, data_manager):
        self.nlp = nlp_service # Reference to NLPService for QA
        self.data_manager = data_manager # Reference to DataManager for logging
        self.user_data = self._load_user_data()

    def _load_user_data(self):
        if os.path.exists(Config.USER_DATA_FILE):
            try:
                with open(Config.USER_DATA_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                st.warning(f"Warning: {Config.USER_DATA_FILE} is corrupted. Starting with new user data.")
                return {"points": 0, "badges": []}
        return {"points": 0, "badges": []}

    def _save_user_data(self):
        with open(Config.USER_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.user_data, f, indent=4)

    def add_points(self, amount):
        self.user_data['points'] += amount
        st.toast(f"🎉 You earned {amount} points! Total: {self.user_data['points']}")
        self._save_user_data()

    def award_badge(self, badge_name):
        if badge_name not in self.user_data['badges']:
            self.user_data['badges'].append(badge_name)
            st.toast(f"🏅 Congratulations! You earned the '{badge_name}' badge!", icon="🎉")
            self._save_user_data()
        else:
            st.toast(f"You already have the '{badge_name}' badge.", icon="ℹ️")

    def run_mcq_quiz(self, summary_text):
        st.subheader("Quiz Time!")
        st.write("Answer 5 questions based on the lecture summary.")

        # Initialize quiz state
        if 'quiz_state' not in st.session_state:
            st.session_state.quiz_state = {
                'current_question_idx': 0,
                'questions': [],
                'score': 0,
                'quiz_session_id': None,
                'feedback': ""
            }
        
        # If questions not generated yet, generate them
        if not st.session_state.quiz_state['questions']:
            st.session_state.quiz_state['quiz_session_id'] = self.data_manager.log_quiz_session(0, 5, summary_text) # Log with temp score
            chunks = [summary_text[i:i+400] for i in range(0, len(summary_text), 400)]
            if len(chunks) < 5:
                chunks = (chunks * ((5 // len(chunks)) + 1)) if len(chunks) > 0 else ["No summary to generate questions from."] * 5

            with st.spinner("Generating quiz questions..."):
                # Use LLM to generate questions and options directly
                if not self.nlp.llm_placeholder_mode:
                    quiz_prompt = f"""
                    Generate 5 multiple-choice questions (MCQs) based on the following lecture summary.
                    For each question, provide:
                    1. The question.
                    2. The correct answer.
                    3. Three plausible but incorrect distractors.

                    Format each question strictly as follows:
                    Q: [Question text]?
                    A: [Correct answer text]
                    D1: [Distractor 1 text]
                    D2: [Distractor 2 text]
                    D3: [Distractor 3 text]

                    Ensure questions test understanding of key concepts from the summary. Make questions specific, not generic.

                    Lecture Summary:
                    {summary_text}

                    Questions:
                    """
                    llm_quiz_response = self.nlp._query_llm(quiz_prompt, max_tokens=1000)
                    
                    if not self.nlp.llm_placeholder_mode and llm_quiz_response and "LLM Placeholder" not in llm_quiz_response:
                        # Parse LLM response into question objects
                        lines = llm_quiz_response.strip().split('\n')
                        temp_questions = []
                        q_data = {}
                        for line_num, line in enumerate(lines): # Added line_num for better error reporting
                            line = line.strip()
                            if not line: continue # Skip empty lines
                            
                            if line.startswith("Q:"):
                                if q_data: temp_questions.append(q_data) # Save previous q
                                q_data = {'question': line[2:].strip(), 'options': [], 'correct_answer': '', 'correct_option_char': ''}
                            elif line.startswith("A:"):
                                q_data['correct_answer'] = line[2:].strip()
                                q_data['options'].append(q_data['correct_answer'])
                            elif line.startswith("D") and ":" in line:
                                q_data['options'].append(line[line.find(":")+1:].strip())
                            else: # Debugging unexpected lines
                                st.warning(f"Quiz generation: Unexpected line format at line {line_num+1} in LLM response: {line}")

                        if q_data: temp_questions.append(q_data) # Add last q

                        # Finalize options and correct_option_char
                        final_questions = []
                        for q in temp_questions: # Iterate all parsed questions
                            if q['question'] and q['correct_answer'] and len(q['options']) >= 4: # Ensure all parts are there
                                random.shuffle(q['options'])
                                q['correct_option_char'] = "ABCD"[q['options'].index(q['correct_answer'])]
                                final_questions.append(q)
                            else:
                                st.warning(f"Skipping malformed question generated by LLM (missing Q/A/Distractors): {q.get('question', 'N/A')}")
                        
                        if len(final_questions) < 5:
                            st.warning(f"LLM generated only {len(final_questions)} valid quiz questions. Attempting to fill with basic questions.")
                            # Fallback if LLM didn't generate enough valid questions
                            self._generate_basic_quiz_questions_fallback(summary_text, questions_to_generate=5 - len(final_questions), existing_questions=final_questions)
                        else:
                            st.session_state.quiz_state['questions'] = final_questions[:5] # Take top 5

                    else: # LLM is in placeholder mode or failed to generate quiz content/parse
                        st.warning("LLM not active or failed to generate quiz content. Falling back to basic quiz questions.")
                        self._generate_basic_quiz_questions_fallback(summary_text)

                else: # LLM is in placeholder mode or failed to generate quiz content
                    st.warning("LLM not active or failed to generate quiz content. Falling back to basic quiz questions.")
                    self._generate_basic_quiz_questions_fallback(summary_text)
            
            st.rerun() # Rerun to display first question after generation

        # --- Display Current Question ---
        q_idx = st.session_state.quiz_state['current_question_idx']
        if q_idx < len(st.session_state.quiz_state['questions']):
            q_data = st.session_state.quiz_state['questions'][q_idx]
            
            st.markdown(f"---")
            st.write(f"**Question {q_idx + 1} of 5:** {q_data['question']}")
            
            # Use st.form to group radio buttons and submit button, and handle logic
            with st.form(key=f"quiz_form_{q_idx}"):
                user_choice = st.radio("Select your answer:", [f"{chr(65+j)}. {opt}" for j, opt in enumerate(q_data['options'])], key=f"user_q{q_idx}")
                submit_button = st.form_submit_button(label="Submit Answer")

                if submit_button:
                    selected_char = user_choice[0]
                    is_correct = (selected_char == q_data['correct_option_char'])

                    if is_correct:
                        st.success("✅ Correct!")
                        st.session_state.quiz_state['score'] += 1
                        self.add_points(10)
                        st.session_state.quiz_state['feedback'] = "Correct"
                    else:
                        st.error(f"❌ Incorrect. The correct answer was **{q_data['correct_option_char']}. {q_data['correct_answer']}**")
                        st.session_state.quiz_state['feedback'] = "Incorrect"
                    
                    # Log the attempt
                    self.data_manager.log_quiz_question_attempt(
                        st.session_state.quiz_state['quiz_session_id'],
                        q_idx + 1,
                        q_data['question'],
                        q_data['correct_option_char'],
                        is_correct
                    )
                    # Use a session state variable to indicate submission for current question
                    st.session_state[f'q{q_idx}_submitted'] = True
                    st.rerun() # Rerun to show feedback and Next button

            # After submission, display next question button
            if st.session_state.get(f'q{q_idx}_submitted', False):
                if q_idx < 4: # If not the last question
                    if st.button("Next Question ▶️", key=f"next_q_button_{q_idx}"):
                        st.session_state.quiz_state['current_question_idx'] += 1
                        st.session_state.quiz_state['feedback'] = "" # Clear feedback
                        st.rerun()
                else: # Last question
                    st.info("You've completed all questions!")
                    # Update final score for quiz session
                    self.data_manager.update_quiz_session_score(
                        st.session_state.quiz_state['quiz_session_id'],
                        st.session_state.quiz_state['score']
                    )
                    st.session_state.quiz_state['current_question_idx'] += 1 # Advance past last question
                    st.rerun() # Rerun to show final score summary

        # --- Display Final Quiz Results ---
        else: # All questions answered
            st.markdown("---")
            final_score = st.session_state.quiz_state['score']
            st.header(f"Final Quiz Score: {final_score}/5")
            if final_score == 5:
                st.balloons()
                st.write("🏆 Outstanding! Perfect memory!")
                self.award_badge("Quiz Master")
            elif final_score >= 3:
                st.write("👏 Good job! You're doing great.")
                self.award_badge("Good Learner")
            else:
                st.write("📚 Let’s review again and try once more!")
                self.award_badge("Needs Review")
            
            st.write(f"\n**Your Current Progress:** Points: {self.user_data['points']}, Badges: {', '.join(self.user_data['badges']) if self.user_data['badges'] else 'None'}")
            
            if st.button("Restart Quiz"):
                del st.session_state['quiz_state'] # Reset quiz state
                st.session_session.quiz_started = False
                st.rerun()


# --- 4. Data Management for Insights ---
class DataManager:
    def __init__(self, db_path=Config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quiz_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                quiz_score INTEGER,
                total_questions INTEGER,
                summary_preview TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quiz_questions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                session_id INTEGER,
                question_num INTEGER,
                question_text TEXT,
                correct_answer_char TEXT,
                user_answer_correct BOOLEAN
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                question_text TEXT,
                answer_text TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_quiz_session(self, score, total, full_summary_text):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO quiz_sessions (quiz_score, total_questions, summary_preview) VALUES (?, ?, ?)",
                       (score, total, full_summary_text[:200] + ("..." if len(full_summary_text) > 200 else "")))
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return session_id

    def update_quiz_session_score(self, session_id, final_score):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE quiz_sessions SET quiz_score = ? WHERE id = ?", (final_score, session_id))
        conn.commit()
        conn.close()

    def log_quiz_question_attempt(self, session_id, question_num, question_text, correct_char, is_correct):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO quiz_questions_log (session_id, question_num, question_text, correct_answer_char, user_answer_correct) VALUES (?, ?, ?, ?, ?)",
                       (session_id, question_num, question_text, correct_char, is_correct))
        conn.commit()
        conn.close()

    def log_qa_interaction(self, question, answer):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO qa_interactions (question_text, answer_text) VALUES (?, ?)",
                       (question, answer))
        conn.commit()
        conn.close()

    def generate_teacher_report(self):
        st.header("📊 Teacher Insights Report")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        st.markdown("---")
        st.subheader("Overall Quiz Performance")
        cursor.execute("SELECT AVG(quiz_score), COUNT(*) FROM quiz_sessions")
        avg_score, total_sessions = cursor.fetchone()
        st.write(f"**Total Quiz Sessions Recorded:** {total_sessions}")
        if total_sessions > 0:
            st.write(f"**Average Quiz Score (across sessions):** {avg_score if avg_score is not None else 0:.2f} / 5")
        else:
            st.write("No quiz sessions recorded yet.")

        st.markdown("---")
        st.subheader("Common Misconceptions (Top 5 incorrectly answered questions)")
        cursor.execute("""
            SELECT question_text,
                   SUM(CASE WHEN user_answer_correct = 0 THEN 1 ELSE 0 END) AS incorrect_count,
                   COUNT(*) AS total_attempts
            FROM quiz_questions_log
            GROUP BY question_text
            ORDER BY incorrect_count DESC
            LIMIT 5
        """)
        common_misconceptions = cursor.fetchall()
        if common_misconceptions:
            for q_text, incorrect_c, total_att in common_misconceptions:
                st.write(f"- **Question:** '{q_text}' (Incorrect {incorrect_c} / {total_att} times)")
        else:
            st.write("No detailed quiz question data yet.")

        st.markdown("---")
        st.subheader("Popular Q&A Chatbot Questions")
        cursor.execute("SELECT question_text, COUNT(*) FROM qa_interactions GROUP BY question_text ORDER BY COUNT(*) DESC LIMIT 5")
        popular_questions = cursor.fetchall()
        if popular_questions:
            for q, count in popular_questions:
                st.write(f"- '{q}' (asked {count} times)")
        else:
            st.write("No Q&A interactions recorded.")
        st.markdown("---")
        conn.close()
        st.info(f"Raw database insights available at: `{Config.DB_PATH}`") # Corrected to use Config.DB_PATH


# --- Main Streamlit Application Flow ---
def main_app():
    st.set_page_config(page_title="AI Lecture Assistant", layout="wide", initial_sidebar_state="expanded")
    st.title("📚 AI Lecture Assistant for Classrooms")

    # Initialize services
    audio_processor = AudioProcessor()
    nlp_service = NLPService() # NLPService now automatically checks for GOOGLE_API_KEY
    data_manager = DataManager()
    quiz_manager = QuizManager(nlp_service, data_manager)

    # Sidebar Navigation
    st.sidebar.header("Main Menu")
    page = st.sidebar.radio("Go to", ["Lecture Session", "Teacher Dashboard", "About"])

    # --- Lecture Session Page ---
    if page == "Lecture Session":
        st.header("Active Lecture Session")
        
        # Display duration using conditional logic (seconds if less than a minute, otherwise minutes)
        def format_duration(seconds):
            if seconds >= 60:
                return f"{seconds // 60} minutes"
            else:
                return f"{seconds} seconds"

        st.write(f"Class Duration: 20 minutes per part (total 40 minutes)") # Hardcoded conceptual duration display
        st.write(f"Break Duration: 5 minutes") # Hardcoded conceptual duration display


        # --- Reset Session Button ---
        if st.button("🔄 Reset Lecture Session", key="reset_session_button"):
            # Clear all relevant session state variables to restart the flow
            for key in ['lecture_stage', 'full_summary', 'full_transcription', 'chat_history', 'quiz_started', 'quiz_state', 'uploaded_file_part1_path', 'uploaded_file_part2_path']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun() # Trigger a rerun to reset the app UI

        # --- Check if session data already exists from a previous (interrupted) run ---
        # This allows continuing if a user closes browser and reopens, or if app crashed mid-session.
        # It relies on existing st.session_state values.
        if (st.session_state.get('lecture_stage', 0) > 0 and 
            st.session_state.get('full_summary') and 
            st.session_state.get('full_transcription')):
            
            st.success("Detected previous session data. You can continue to Notes/Chatbot/Quiz or reset.")
            if st.button("▶️ Continue to Notes/Chatbot/Quiz", key="continue_session_button"):
                st.session_state.lecture_stage = 4 # Jump to final stage
                st.rerun()
            st.write("---") # Separator


        # Initialize session state for lecture flow if not already present
        # This entire block will only execute if st.session_state.lecture_stage is NOT set (e.g., after a hard reset)
        if 'lecture_stage' not in st.session_state:
            st.session_state.lecture_stage = 0 # 0: Initial, 1: Part1 Done, 2: Break Done, 3: Part2 Done, 4: All Processed
            st.session_state.full_summary = ""
            st.session_state.full_transcription = ""
            st.session_state.chat_history = []
            st.session_state.quiz_started = False
            st.session_state.current_quiz_q = 0 # Track current quiz question
            st.session_state.uploaded_file_part1_path = None # To preserve uploaded file path across reruns
            st.session_state.uploaded_file_part2_path = None # To preserve uploaded file path across reruns


        # --- Lecture Flow Logic ---
        if st.session_state.lecture_stage == 0:
            st.info("Click the button below to start the first part of the lecture recording.")
            
            # File Uploader for Part 1
            uploaded_file_part1 = st.file_uploader("Or upload Part 1 audio (MP3/WAV)", type=["mp3", "wav"], key="upload_part1")
            
            col_live_rec, col_upload_process = st.columns(2)
            
            with col_live_rec:
                if st.button("▶️ Start Live Recording (Part 1)", key="start_live_part1"):
                    st.session_state.part1_transcription = audio_processor.record_audio_live("part1_audio", Config.LECTURE_PART_DURATION_SEC)
                    # Apply capitalization post-processing
                    st.session_state.part1_transcription = post_process_text_capitalization(st.session_state.part1_transcription)
                    st.session_state.part1_summary = nlp_service.summarize_text(st.session_state.part1_transcription)
                    
                    with open(os.path.join(Config.NOTES_DIR, "part1_full_transcription.txt"), "w", encoding='utf-8') as f: f.write(st.session_state.part1_transcription)
                    with open(os.path.join(Config.NOTES_DIR, "part1_summary.txt"), "w", encoding='utf-8') as f: f.write(st.session_state.part1_summary)

                    st.session_state.lecture_stage = 1 # Move to next stage
                    st.rerun() # Use st.rerun() for Streamlit updates

            with col_upload_process:
                # Only show process button if a file is uploaded AND not already processed
                # Use a unique key for the file uploader to prevent multiple file selection issues on rerun
                if uploaded_file_part1 is not None and st.session_state.uploaded_file_part1_path is None:
                    if st.button("⬆️ Process Uploaded Part 1 Audio", key="process_uploaded_part1"):
                        # Save uploaded file temporarily for processing
                        uploaded_audio_path = os.path.join(Config.NOTES_DIR, uploaded_file_part1.name)
                        with open(uploaded_audio_path, "wb") as f:
                            f.write(uploaded_file_part1.getbuffer())
                        st.session_state.uploaded_file_part1_path = uploaded_audio_path # Store path
                        
                        st.session_state.part1_transcription = audio_processor.transcribe_audio_file(uploaded_audio_path)
                        st.session_state.part1_transcription = post_process_text_capitalization(st.session_state.part1_transcription)
                        st.session_state.part1_summary = nlp_service.summarize_text(st.session_state.part1_transcription)

                        with open(os.path.join(Config.NOTES_DIR, "part1_full_transcription.txt"), "w", encoding='utf-8') as f: f.write(st.session_state.part1_transcription)
                        with open(os.path.join(Config.NOTES_DIR, "part1_summary.txt"), "w", encoding='utf-8') as f: f.write(st.session_state.part1_summary)

                        st.session_state.lecture_stage = 1
                        st.rerun()
                elif uploaded_file_part1 is not None and st.session_state.uploaded_file_part1_path is not None:
                    st.success(f"File '{uploaded_file_part1.name}' uploaded and processed for Part 1.")


        elif st.session_state.lecture_stage == 1:
            st.success("Part 1 of the lecture recorded/uploaded and processed!")
            st.subheader("Summary of Part 1:")
            st.info(st.session_state.part1_summary)
            
            st.write("---")
            st.info("Time for a quick Brain Break!")
            webbrowser.open(random.choice(Config.BRAIN_BREAK_VIDEOS)) # Open video in new tab
            st.warning(f"Please close the video tab manually after {Config.BREAK_DURATION_SEC // 60} minutes.")
            
            with st.spinner(f"Brain break in progress for {Config.BREAK_DURATION_SEC} seconds..."):
                time.sleep(Config.BREAK_DURATION_SEC)
            st.info("Break is over! Ready for Part 2.")
            st.session_state.lecture_stage = 2 # Move to next stage
            st.rerun()

        elif st.session_state.lecture_stage == 2:
            st.info("Click to start recording for the second part of the lecture.")
            uploaded_file_part2 = st.file_uploader("Or upload Part 2 audio (MP3/WAV)", type=["mp3", "wav"], key="upload_part2")

            col_live_rec2, col_upload_process2 = st.columns(2)

            with col_live_rec2:
                if st.button("▶️ Start Live Recording (Part 2)", key="start_live_part2"):
                    st.session_state.part2_transcription = audio_processor.record_audio_live("part2_audio", Config.LECTURE_PART_DURATION_SEC)
                    st.session_state.part2_transcription = post_process_text_capitalization(st.session_state.part2_transcription)
                    st.session_state.part2_summary = nlp_service.summarize_text(st.session_state.part2_transcription)

                    with open(os.path.join(Config.NOTES_DIR, "part2_full_transcription.txt"), "w", encoding='utf-8') as f: f.write(st.session_state.part2_transcription)
                    with open(os.path.join(Config.NOTES_DIR, "part2_summary.txt"), "w", encoding='utf-8') as f: f.write(st.session_state.part2_summary)

                    st.session_state.lecture_stage = 3
                    st.rerun()

            with col_upload_process2:
                if uploaded_file_part2 is not None and st.session_state.uploaded_file_part2_path is None:
                    if st.button("⬆️ Process Uploaded Part 2 Audio", key="process_uploaded_part2"):
                        uploaded_audio_path = os.path.join(Config.NOTES_DIR, uploaded_file_part2.name)
                        with open(uploaded_audio_path, "wb") as f:
                            f.write(uploaded_file_part2.getbuffer())
                        st.session_state.uploaded_file_part2_path = uploaded_audio_path # Store path

                        st.session_state.part2_transcription = audio_processor.transcribe_audio_file(uploaded_audio_path)
                        st.session_state.part2_transcription = post_process_text_capitalization(st.session_state.part2_transcription)
                        st.session_state.part2_summary = nlp_service.summarize_text(st.session_state.part2_transcription)

                        with open(os.path.join(Config.NOTES_DIR, "part2_full_transcription.txt"), "w", encoding='utf-8') as f: f.write(st.session_state.part2_transcription)
                        with open(os.path.join(Config.NOTES_DIR, "part2_summary.txt"), "w", encoding='utf-8') as f: f.write(st.session_state.part2_summary)

                        st.session_state.lecture_stage = 3
                        st.rerun()
                elif uploaded_file_part2 is not None and st.session_state.uploaded_file_part2_path is not None:
                    st.success(f"File '{uploaded_file_part2.name}' uploaded and processed for Part 2.")


        elif st.session_state.lecture_stage == 3:
            st.success("Part 2 of the lecture recorded/uploaded and processed!")
            st.subheader("Summary of Part 2:")
            st.info(st.session_state.part2_summary)
            
            # Combine full text and summary
            st.session_state.full_transcription = st.session_state.part1_transcription + "\n" + st.session_state.part2_transcription
            st.session_state.full_summary = st.session_state.part1_summary + "\n" + st.session_state.part2_summary

            # Save full class notes
            with open(os.path.join(Config.NOTES_DIR, "full_class_transcription.txt"), "w", encoding='utf-8') as f: f.write(st.session_state.full_transcription)
            with open(os.path.join(Config.NOTES_DIR, "full_class_summary.txt"), "w", encoding='utf-8') as f: f.write(st.session_state.full_summary)

            st.success("All lecture parts processed and combined!")
            st.session_state.lecture_stage = 4 # Move to final stage
            st.rerun()

        elif st.session_state.lecture_stage == 4:
            st.subheader("🎉 Class Over! Here are your generated notes:")
            st.write(st.session_state.full_summary)

            st.markdown("---")
            st.subheader("Key Concepts & External Enrichment")
            key_concepts_output = nlp_service.extract_key_concepts_and_definitions(st.session_state.full_summary)
            st.text_area("Key Concepts:", key_concepts_output, height=150)
            with open(os.path.join(Config.NOTES_DIR, "key_concepts.txt"), "w", encoding='utf-8') as f: f.write(key_concepts_output)

            enrichment_info = nlp_service.enrich_content(st.session_state.full_summary)
            st.text_area("Enrichment Info:", enrichment_info, height=150)
            with open(os.path.join(Config.NOTES_DIR, "enriched_info.txt"), "w", encoding='utf-8') as f: f.write(enrichment_info)

            st.markdown("---")
            st.subheader("Interactive Q&A Chatbot")
            st.write("Ask questions about the lecture content. (Chat history clears on app rerun)")
            
            # Display chat history
            for chat_entry in st.session_state.chat_history:
                if chat_entry['role'] == "You":
                    st.markdown(f"**You:** {chat_entry['content']}")
                else:
                    st.markdown(f"**AI Assistant:** {chat_entry['content']}")

            # Chatbot input form
            with st.form(key="chatbot_form"):
                user_question = st.text_input("Type your question here:", key="current_chat_input", placeholder="e.g., 'Can you explain the main theories?'")
                submit_chat_button = st.form_submit_button(label="Ask AI Assistant")

                if submit_chat_button and user_question:
                    st.session_state.chat_history.append({"role": "You", "content": user_question})
                    ai_answer = nlp_service.answer_question(user_question, st.session_state.full_summary)
                    data_manager.log_qa_interaction(user_question, ai_answer)
                    st.session_state.chat_history.append({"role": "AI Assistant", "content": ai_answer})
                    st.rerun() # Rerun to update chat history display and clear the input box
                elif submit_chat_button and not user_question:
                    st.warning("Please type a question before clicking 'Ask AI Assistant'.")

            st.markdown("---")
            st.subheader("Quiz Time!")
            if not st.session_state.quiz_started:
                if st.button("Start Quiz Now"):
                    st.session_state.quiz_started = True
                    st.session_state.current_quiz_q = 0 # Start from first question
                    st.rerun() # Use st.rerun()
            else:
                quiz_manager.run_mcq_quiz(st.session_state.full_summary)
                
            st.sidebar.markdown("---")
            st.sidebar.subheader("Your Gamification Progress")
            st.sidebar.write(f"**Points:** {quiz_manager.user_data['points']}")
            st.sidebar.write(f"**Badges:** {', '.join(quiz_manager.user_data['badges']) if quiz_manager.user_data['badges'] else 'None'}")


    # --- Teacher Dashboard Page ---
    elif page == "Teacher Dashboard":
        data_manager.generate_teacher_report()

    # --- About Page ---
    elif page == "About":
        st.header("About the AI Lecture Assistant")
        st.write("""
        This application is designed to help students and teachers enhance the learning experience by automating lecture note-taking and assessment.
        
        **Features:**
        - **Automatic Transcription:** Converts spoken lectures into text.
        - **Summarization:** Condenses long lectures into key points.
        - **Brain Breaks:** Integrates short, fun videos to boost focus.
        - **Key Concept Extraction:** Identifies and defines crucial terms (powered by LLM).
        - **Cross-Referencing:** Provides external information on key topics (uses Wikipedia or LLM).
        - **Interactive Q&A Chatbot:** Answers student questions based on lecture content (powered by LLM).
        - **MCQ Quiz:** Tests comprehension at the end of the class.
        - **Gamification:** Rewards student engagement with points and badges.
        - **Teacher Dashboard:** Provides insights into class performance and common misconceptions.
        
        **Technology Used:**
        - **Streamlit:** For the interactive web user interface.
        - **Whisper:** For high-quality audio transcription.
        - **Hugging Face Transformers:** For summarization (T5-small) and question-answering (DistilBERT) as base models.
        - **Google Gemini API:** For advanced generative AI capabilities.
        - **Sounddevice & SciPy:** For audio recording and file handling.
        - **SQLite:** For robust data persistence of insights.
        - **Wikipedia API:** For external knowledge enrichment.
        """)
        st.info(f"Developed by: Nandhini M | Purani R | Rathipriya S") # Added team names
      

# --- Run the Streamlit Application ---
if __name__ == "__main__":
    # Ensure the notes directory exists before the app starts
    os.makedirs(Config.NOTES_DIR, exist_ok=True)
    
    # Initialize DB (will create if not exists)
    DataManager(Config.DB_PATH) 

    # Run the main Streamlit app function
    main_app()