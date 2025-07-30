from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
import requests
from flask import send_from_directory
import io
import zipfile
import tempfile
import shutil
import base64
import json
import csv
import re
from werkzeug.utils import secure_filename

# TTS API configuration
TTS_URL = "https://384ce62d09bd.ngrok-free.app"
API_SPEAKING_URL="https://b08b95d83396.ngrok-free.app"
API_WRITING_URL="https://baf7b1b76c01.ngrok-free.app"
LISTENING_URL="https://59af2917ad63.ngrok-free.app"

def extract_topic_from_question(question_text):
    """
    Extract topic from question text like "(Topic: Education) You have about 20 seconds..."
    """
    if not question_text:
        return None
    
    # Look for pattern like "(Topic: Education)" or "(Topic:Education)"
    topic_match = re.search(r'\(Topic:\s*([^)]+)\)', question_text)
    if topic_match:
        return topic_match.group(1).strip()
    return None

def save_topic_to_csv(topic, part):
    """
    Save topic and part to CSV file
    """
    csv_file = 'topics_used.csv'
    file_exists = os.path.exists(csv_file)
    
    try:
        with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow(['topic', 'part'])
            
            # Write the topic and part
            writer.writerow([topic, part])
            print(f"Saved topic '{topic}' for part {part} to CSV")
    except Exception as e:
        print(f"Error saving topic to CSV: {e}")

def split_text_into_batches(text, max_tokens=400):
    """
    Split text into batches of approximately max_tokens
    """
    # Ultra conservative estimation: 1 token ≈ 1.5 characters for English text
    # This ensures we stay well under the 400 token limit
    max_chars = max_tokens * 1.5  # 600 characters max per batch
    
    if len(text) <= max_chars:
        return [text]
    
    # Split by words first for more precise control
    words = text.split()
    batches = []
    current_batch = ""
    
    for word in words:
        # Check if adding this word would exceed the limit
        if len(current_batch) + len(word) + 1 > max_chars:
            if current_batch:
                batches.append(current_batch.strip())
                current_batch = word
            else:
                # Single word is too long, truncate it
                batches.append(word[:max_chars])
        else:
            current_batch += " " + word if current_batch else word
    
    # Add the last batch
    if current_batch:
        batches.append(current_batch.strip())
    
    # Final validation: ensure no batch exceeds the limit
    validated_batches = []
    for batch in batches:
        if len(batch) > max_chars:
            print(f"TTS Debug - Batch too long ({len(batch)} chars), splitting further")
            # Split this batch into smaller chunks
            words = batch.split()
            current_chunk = ""
            for word in words:
                if len(current_chunk) + len(word) + 1 > max_chars:
                    if current_chunk:
                        validated_batches.append(current_chunk.strip())
                        current_chunk = word
                    else:
                        # Single word is too long, truncate
                        validated_batches.append(word[:max_chars])
                else:
                    current_chunk += " " + word if current_chunk else word
            if current_chunk:
                validated_batches.append(current_chunk.strip())
        else:
            validated_batches.append(batch)
    
    # Log batch information for debugging
    for i, batch in enumerate(validated_batches):
        print(f"TTS Debug - Batch {i+1}: {len(batch)} characters")
    
    return validated_batches

def concatenate_audio_chunks(audio_chunks):
    """
    Concatenate multiple audio chunks into a single audio file
    """
    if not audio_chunks:
        return None
    
    if len(audio_chunks) == 1:
        return audio_chunks[0]
    
    try:
        import io
        import wave
        import numpy as np
        import struct
        import tempfile
        import os
        
        print(f"TTS Debug - Starting concatenation of {len(audio_chunks)} audio chunks")
        
        # Extract audio data from WAV files and concatenate properly
        audio_data_chunks = []
        wav_params = None
        
        for i, chunk in enumerate(audio_chunks):
            print(f"TTS Debug - Processing chunk {i+1} of {len(audio_chunks)}")
            
            # Create a temporary file for this chunk
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            try:
                # Read the WAV file and extract audio data
                with wave.open(temp_file_path, 'rb') as wav_file:
                    # Store parameters from first file
                    if wav_params is None:
                        wav_params = {
                            'nchannels': wav_file.getnchannels(),
                            'sampwidth': wav_file.getsampwidth(),
                            'framerate': wav_file.getframerate(),
                            'nframes': wav_file.getnframes()
                        }
                        print(f"TTS Debug - WAV parameters: {wav_params}")
                    
                    # Get audio frames
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_data_chunks.append(frames)
                    print(f"TTS Debug - Chunk {i+1}: {len(frames)} audio frames, {wav_file.getnframes()} frames")
                    
            except Exception as e:
                print(f"TTS Debug - Error reading chunk {i+1}: {e}")
                # If we can't read the WAV, try to use raw data
                audio_data_chunks.append(chunk)
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
        
        # Concatenate all audio data
        concatenated_audio_data = b''.join(audio_data_chunks)
        print(f"TTS Debug - Concatenated audio data size: {len(concatenated_audio_data)} bytes")
        
        # Calculate total frames
        total_frames = len(concatenated_audio_data) // (wav_params['nchannels'] * wav_params['sampwidth'])
        print(f"TTS Debug - Total frames: {total_frames}")
        
        # Create a new WAV file with the concatenated audio
        output_buffer = io.BytesIO()
        
        with wave.open(output_buffer, 'wb') as output_wav:
            output_wav.setnchannels(wav_params['nchannels'])
            output_wav.setsampwidth(wav_params['sampwidth'])
            output_wav.setframerate(wav_params['framerate'])
            output_wav.setnframes(total_frames)
            output_wav.writeframes(concatenated_audio_data)
        
        # Get the concatenated audio
        concatenated_audio = output_buffer.getvalue()
        output_buffer.close()
        
        print(f"TTS Debug - Successfully concatenated {len(audio_chunks)} audio chunks")
        print(f"TTS Debug - Total concatenated size: {len(concatenated_audio)} bytes")
        print(f"TTS Debug - Expected duration: {total_frames / wav_params['framerate']:.2f} seconds")
        
        return concatenated_audio
        
    except Exception as e:
        print(f"TTS Debug - Error concatenating audio: {e}")
        print(f"TTS Debug - Exception type: {type(e)}")
        import traceback
        print(f"TTS Debug - Traceback: {traceback.format_exc()}")
        # Fallback: return the first chunk if concatenation fails
        print(f"TTS Debug - Falling back to first chunk only")
        return audio_chunks[0]

def generate_tts_audio(text):
    """
    Generate TTS audio using the external API with retries
    """
    if not text or not text.strip():
        print("TTS: Empty or whitespace-only text provided")
        return None
    
    # Clean the text for TTS
    cleaned_text = clean_text_for_tts(text)
    print(f"TTS Debug - Original text: {text}")
    print(f"TTS Debug - Cleaned text: {cleaned_text}")
    print(f"TTS Debug - Text length: {len(cleaned_text)}")
    
    # Split text into batches if it's too long
    text_batches = split_text_into_batches(cleaned_text, max_tokens=400)
    print(f"TTS Debug - Split into {len(text_batches)} batches")
    
    if len(text_batches) > 1:
        print(f"TTS Debug - Processing {len(text_batches)} batches...")
        audio_chunks = []
        
        for i, batch in enumerate(text_batches):
            print(f"TTS Debug - Processing batch {i+1}/{len(text_batches)}: {batch[:100]}...")
            
            # Generate audio for this batch
            batch_audio = generate_tts_audio_single_batch(batch)
            if batch_audio:
                audio_chunks.append(batch_audio)
            else:
                print(f"TTS Debug - Failed to generate audio for batch {i+1}")
                return None
        
        # Return the audio chunks as a list instead of concatenating
        if audio_chunks:
            print(f"TTS Debug - Returning {len(audio_chunks)} audio chunks")
            return audio_chunks
        else:
            print("TTS Debug - No audio chunks generated")
            return None
    else:
        # Single batch, process normally
        return generate_tts_audio_single_batch(cleaned_text)

def generate_tts_audio_single_batch(text):
    """
    Generate TTS audio for a single text batch
    """
    print(f"TTS Debug - URL: {TTS_URL}/speak_male")
    
    for attempt in range(6):  # Retry up to 6 times
        try:
            print(f"TTS Debug - Attempt {attempt + 1}: Sending request...")
            
            # Prepare the request payload
            payload = {"text": text}
            print(f"TTS Debug - Payload length: {len(text)} characters")
            
            response = requests.post(
                f"{TTS_URL}/speak_male",
                json=payload,
                timeout=30  # Add timeout
            )
            
            print(f"TTS Debug - Response status: {response.status_code}")
            print(f"TTS Debug - Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                print(f"TTS Debug - Success! Response size: {len(response.content)} bytes")
                return response.content
            else:
                print(f"TTS Debug - Error response content: {response.text[:500]}")
                print(f"TTS attempt {attempt + 1}: Got status {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"TTS Debug - Attempt {attempt + 1}: Timeout error")
        except requests.exceptions.ConnectionError as e:
            print(f"TTS Debug - Attempt {attempt + 1}: Connection error: {e}")
        except Exception as e:
            print(f"TTS Debug - Attempt {attempt + 1}: Exception: {e}")
            print(f"TTS Debug - Exception type: {type(e)}")
        
        if attempt < 5:  # Don't sleep after the last attempt
            print(f"TTS Debug - Waiting 5 seconds before retry...")
            import time
            time.sleep(5)
    
    print("TTS Debug - Failed after 6 attempts")
    return None

def reorganize_grammar_feedback(grammar_feedback):
    """
    Reorganize grammar feedback JSON into natural language for TTS
    """
    try:
        # Try to parse as JSON
        if isinstance(grammar_feedback, str):
            import json
            data = json.loads(grammar_feedback)
        else:
            data = grammar_feedback
        
        # Check if it's the expected format
        if isinstance(data, dict) and 'result' in data and 'errors' in data:
            if data['result'] == 'correct':
                return "Your grammar is correct. Well done!"
            
            # Reorganize errors into natural language
            feedback_parts = []
            for error in data.get('errors', []):
                location = error.get('location', '')
                issue = error.get('issue', '')
                correction = error.get('correction', '')
                
                # Create natural language feedback
                if location and correction:
                    feedback_parts.append(f"The Mistake: {location}. {issue}. The Correction: {correction}")
                elif issue:
                    feedback_parts.append(f"Grammar issue: {issue}")
            
            if feedback_parts:
                return "Grammar feedback: " + ". ".join(feedback_parts)
            else:
                return "Some grammar issues were found. Please review your response."
        
        # If not JSON format, return as is
        return grammar_feedback
        
    except (json.JSONDecodeError, TypeError):
        # If not valid JSON, return as is
        return grammar_feedback

def clean_question_format(question_text):
    """
    Clean up the question format:
    1. Remove (Topic: X) from the beginning
    2. Move "You have about 20 seconds to answer." to the end
    3. Remove "Examiner:" prefix
    """
    if not question_text:
        return question_text
    
    # Remove topic prefix like "(Topic: Public Transport)"
    import re
    question_text = re.sub(r'^\(Topic:\s*[^)]+\)\s*', '', question_text)
    
    # Remove "Examiner:" prefix
    question_text = re.sub(r'^Examiner:\s*', '', question_text)
    
    # Move timing text to the end if it exists
    timing_pattern = r'(You have about \d+ seconds? to answer\.)'
    timing_match = re.search(timing_pattern, question_text)
    
    if timing_match:
        timing_text = timing_match.group(1)
        # Remove timing text from current position
        question_text = re.sub(timing_pattern, '', question_text).strip()
        # Add timing text to the end
        question_text = f"{question_text} {timing_text}"
    
    return question_text.strip()

def clean_text_for_tts(text):
    """
    Clean text for TTS by removing punctuation marks and special characters
    """
    if not text:
        return text
    
    import re
    
    # Replace bullet points with commas
    cleaned_text = text.replace('•', ',')
    
    # Remove special characters but keep basic punctuation and colons
    # Keep letters, numbers, spaces, and basic punctuation: . , ! ? - :
    cleaned_text = re.sub(r'[^\w\s\.\,\!\?\-\:]', '', cleaned_text)
    
    # Remove multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Remove leading/trailing spaces
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'audios/'
app.config['ERROR_CLIPS_FOLDER'] = 'error_clips/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['ERROR_CLIPS_FOLDER']):
    os.makedirs(app.config['ERROR_CLIPS_FOLDER'])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/speaking', methods=['GET', 'POST'])
def speaking():
    return render_template('speaking.html')

@app.route('/listening', methods=['GET', 'POST'])
def listening():
    audio_url = None
    if request.method == 'POST':
        topic = request.form.get('topic')
        part = request.form.get('part', '1')  # Default to part 1 if not specified
        
        if topic:
            # Call the listening API with both topic and part
            api_url = f"{LISTENING_URL}/generate_speech"
            
            # Use GET request with query parameters like the original working version
            api_url = f"{LISTENING_URL}/generate_speech?topic={topic}&part={part}"
            
            print(f"Listening request to: {api_url}")
            print(f"Topic: {topic}")
            print(f"Part: {part}")
            
            try:
                response = requests.get(api_url)
                
                print(f"Listening response status: {response.status_code}")
                
                if response.status_code == 200:
                    # Try to parse as JSON first (in case it's a transcript response)
                    try:
                        result = response.json()
                        print(f"Listening JSON response: {result}")
                        
                        if 'transcript' in result and 'warning' in result:
                            # Speech server failed, but we have transcript
                            print(f"Speech server failed, but transcript available: {result['transcript']}")
                            
                            # Clean the transcript for TTS
                            cleaned_transcript = result['transcript']
                            # Remove <think> tags and their content
                            import re
                            cleaned_transcript = re.sub(r'<think>.*?</think>', '', cleaned_transcript, flags=re.DOTALL)
                            # Remove extra whitespace and normalize
                            cleaned_transcript = re.sub(r'\n\s*\n', '\n', cleaned_transcript)
                            cleaned_transcript = cleaned_transcript.strip()
                            
                            # Parse transcript into speaker dictionary
                            speaker_dict = {}
                            lines = cleaned_transcript.split('\n')
                            
                            print(f"Total lines to process: {len(lines)}")
                            
                            for i, line in enumerate(lines):
                                line = line.strip()
                                if not line:
                                    continue
                                
                                print(f"Processing line {i+1}: {line[:100]}...")
                                
                                # Extract speaker and text - handle special characters
                                # Normalize the line to handle \xa0 and other special characters
                                normalized_line = line.replace('\xa0', ' ')
                                
                                if normalized_line.startswith('[Examiner]'):
                                    speaker = '[Examiner]'
                                    text = normalized_line.replace('[Examiner]', '').strip()
                                elif normalized_line.startswith('[Speaker 1]'):
                                    speaker = '[Speaker 1]'
                                    text = normalized_line.replace('[Speaker 1]', '').strip()
                                elif normalized_line.startswith('[Speaker 2]'):
                                    speaker = '[Speaker 2]'
                                    text = normalized_line.replace('[Speaker 2]', '').strip()
                                elif normalized_line.startswith('[Speaker 3]'):
                                    speaker = '[Speaker 3]'
                                    text = normalized_line.replace('[Speaker 3]', '').strip()
                                else:
                                    # If no speaker tag, skip this line
                                    print(f"Skipping line {i+1} - no speaker tag found: {line[:50]}...")
                                    continue
                                
                                print(f"Found speaker: {speaker}, text: {text[:50]}...")
                                
                                if text:  # Only add if there's actual text
                                    # Concatenate text if speaker already exists
                                    if speaker in speaker_dict:
                                        speaker_dict[speaker] += ' ' + text
                                        print(f"Appended to existing {speaker}")
                                    else:
                                        speaker_dict[speaker] = text
                                        print(f"Created new entry for {speaker}")
                                else:
                                    print(f"No text found for {speaker}")
                            
                            print(f"Speaker dictionary: {speaker_dict}")
                            
                            # Try to generate audio from transcript using the new TTS endpoint
                            try:
                                print(f"Sending TTS request to: {TTS_URL}/speak_text")
                                print(f"TTS payload: {speaker_dict}")
                                
                                # Convert speaker dictionary back to transcript format for the API
                                transcript_text = ""
                                for speaker, text in speaker_dict.items():
                                    if speaker == "[Examiner]":
                                        transcript_text += f"[Examiner] {text}\n"
                                    elif speaker == "[Speaker 1]":
                                        transcript_text += f"[Speaker 1] {text}\n"
                                    elif speaker == "[Speaker 2]":
                                        transcript_text += f"[Speaker 2] {text}\n"
                                    elif speaker == "[Speaker 3]":
                                        transcript_text += f"[Speaker 3] {text}\n"
                                
                                print(f"Converted transcript text: {transcript_text[:200]}...")
                                
                                # Try the speak_text endpoint first
                                try:
                                    # First try with a shorter version to test if the endpoint works
                                    short_transcript = ""
                                    for speaker, text in speaker_dict.items():
                                        if speaker == "[Examiner]":
                                            short_transcript += f"[Examiner] {text}\n"
                                        elif speaker == "[Speaker 1]":
                                            short_transcript += f"[Speaker 1] {text[:500]}\n"
                                        elif speaker == "[Speaker 2]":
                                            short_transcript += f"[Speaker 2] {text[:500]}\n"
                                        elif speaker == "[Speaker 3]":
                                            short_transcript += f"[Speaker 3] {text[:500]}\n"
                                    
                                    print(f"Trying short transcript first: {short_transcript[:200]}...")
                                    
                                    tts_response = requests.post(
                                        f"{TTS_URL}/speak_text",
                                        json={"transcript": short_transcript},
                                        timeout=120  # Increased timeout for longer transcripts
                                    )
                                    
                                    if tts_response.status_code == 200:
                                        # Success with speak_text
                                        filename = f"ielts_listening_part{part}_{topic.replace(' ', '_')}.wav"
                                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                                        with open(filepath, 'wb') as f:
                                            f.write(tts_response.content)
                                        audio_url = url_for('serve_audio', filename=filename)
                                        print(f"TTS audio generated from transcript: {filename}")
                                        return render_template('listening.html', audio_url=audio_url, error_message=None)
                                    else:
                                        print(f"speak_text failed: {tts_response.status_code}")
                                        print(f"speak_text error: {tts_response.text}")
                                        print(f"speak_text response headers: {dict(tts_response.headers)}")
                                        print(f"speak_text response content length: {len(tts_response.content)}")
                                        # Fallback to speak_male with just the examiner text
                                        raise Exception("speak_text failed, trying fallback")
                                        
                                except Exception as e:
                                    print(f"speak_text error: {e}")
                                    # Fallback: use speak_male with a shorter version of the transcript
                                    try:
                                        # Create a shorter version by taking first 200 characters from each speaker
                                        fallback_text = ""
                                        for speaker, text in speaker_dict.items():
                                            if speaker == "[Examiner]":
                                                fallback_text += f"{text} "
                                            else:
                                                # Take first 1000 characters from each speaker (increased from 200)
                                                short_text = text[:1000] + "..." if len(text) > 1000 else text
                                                fallback_text += f"{short_text} "
                                        
                                        fallback_text = fallback_text.strip()
                                        print(f"Trying fallback with shortened text: {fallback_text[:100]}...")
                                        fallback_response = requests.post(
                                            f"{TTS_URL}/speak_male",
                                            json={"text": fallback_text},
                                            timeout=60
                                        )
                                        
                                        if fallback_response.status_code == 200:
                                            filename = f"ielts_listening_part{part}_{topic.replace(' ', '_')}.wav"
                                            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                                            with open(filepath, 'wb') as f:
                                                f.write(fallback_response.content)
                                            audio_url = url_for('serve_audio', filename=filename)
                                            print(f"Fallback TTS audio generated: {filename}")
                                            return render_template('listening.html', audio_url=audio_url, error_message=None)
                                        else:
                                            print(f"Fallback failed: {fallback_response.status_code}")
                                            print(f"Fallback error: {fallback_response.text}")
                                    except Exception as fallback_e:
                                        print(f"Fallback error: {fallback_e}")
                                    
                                    # If both attempts fail, return transcript
                                    return render_template('listening.html', 
                                                        audio_url=None, 
                                                        transcript=result['transcript'],
                                                        warning=result['warning'])
                                
                            except Exception as e:
                                print(f"TTS generation error: {e}")
                                return render_template('listening.html', 
                                                    audio_url=None, 
                                                    transcript=result['transcript'],
                                                    warning=result['warning'])
                        else:
                            # Normal audio response
                            filename = f"ielts_listening_part{part}_{topic.replace(' ', '_')}.wav"
                            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            audio_url = url_for('serve_audio', filename=filename)
                            print(f"Listening audio saved: {filename}")
                            
                    except json.JSONDecodeError:
                        # Not JSON, treat as audio file
                        filename = f"ielts_listening_part{part}_{topic.replace(' ', '_')}.wav"
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        audio_url = url_for('serve_audio', filename=filename)
                        print(f"Listening audio saved: {filename}")
                        
                else:
                    print(f"Listening API error: {response.status_code}")
                    print(f"Response content: {response.text}")
                    # Return error message to frontend
                    return render_template('listening.html', audio_url=None, error_message=f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                print("Listening request timed out")
                return render_template('listening.html', audio_url=None, error_message="Request timed out. Please try again.")
            except requests.exceptions.ConnectionError as e:
                print(f"Listening connection error: {e}")
                return render_template('listening.html', audio_url=None, error_message="Connection error. Please check your internet connection.")
            except Exception as e:
                print(f"Listening error: {e}")
                return render_template('listening.html', audio_url=None, error_message=f"Error: {str(e)}")
        else:
            return render_template('listening.html', audio_url=None, error_message="Please enter a topic.")
    return render_template('listening.html', audio_url=audio_url, error_message=None)

# Serve audio files
@app.route('/audios/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/writing', methods=['GET', 'POST'])
def writing():
    return render_template('writing.html')

# New speech analysis route
@app.route('/analyze_speech', methods=['POST'])
def analyze_speech():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    try:
        # Save the uploaded audio temporarily
        temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_recording.wav')
        audio_file.save(temp_audio_path)
        
        # Send to external API for analysis
        api_url =  f"{API_SPEAKING_URL}/analyze"
        
        with open(temp_audio_path, 'rb') as f:
            files = {'audio': f}
            response = requests.post(api_url, files=files)
        
        if response.status_code == 200:
            analysis_data = response.json()
            
            # Download and extract error clips if available
            if 'zip_url' in analysis_data and analysis_data['zip_url']:
                zip_url = f"https://3f55758067e5.ngrok-free.app{analysis_data['zip_url']}"
                session_id = analysis_data['session_id']
                
                # Download the zip file
                zip_response = requests.get(zip_url)
                if zip_response.status_code == 200:
                    # Save zip file temporarily
                    temp_zip_path = os.path.join(app.config['ERROR_CLIPS_FOLDER'], f"{session_id}.zip")
                    with open(temp_zip_path, 'wb') as f:
                        f.write(zip_response.content)
                    
                    # Extract error clips
                    extract_error_clips(temp_zip_path, session_id)
                    
                    # Clean up zip file
                    os.remove(temp_zip_path)
            
            # Clean up temporary audio file
            os.remove(temp_audio_path)
            
            return jsonify(analysis_data)
        else:
            # Clean up temporary audio file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return jsonify({'error': 'Analysis failed'}), 500
            
    except Exception as e:
        # Clean up temporary audio file
        temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_recording.wav')
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return jsonify({'error': str(e)}), 500

def extract_error_clips(zip_path, session_id):
    """Extract error clips from the downloaded zip file"""
    try:
        session_folder = os.path.join(app.config['ERROR_CLIPS_FOLDER'], session_id)
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Look for error_clips folder in the zip
            for file_info in zip_ref.filelist:
                if 'error_clips' in file_info.filename:
                    # Extract the file
                    zip_ref.extract(file_info, session_folder)
                    
    except Exception as e:
        print(f"Error extracting clips: {e}")

# Route to serve error clips
@app.route('/play_error_clip/<session_id>/<word>/<int:index>')
def play_error_clip(session_id, word, index):
    try:
        session_folder = os.path.join(app.config['ERROR_CLIPS_FOLDER'], session_id)
        error_clips_folder = os.path.join(session_folder, 'error_clips')
        
        if os.path.exists(error_clips_folder):
            # Look for the specific error clip file
            # The naming pattern is typically like "03_would.wav"
            for filename in os.listdir(error_clips_folder):
                if filename.endswith('.wav') and word.lower() in filename.lower():
                    file_path = os.path.join(error_clips_folder, filename)
                    return send_file(file_path, mimetype='audio/wav')
        
        # If specific file not found, return a default message
        return jsonify({'error': 'Error clip not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New IELTS Speaking Simulation Routes
@app.route('/get_ielts_question', methods=['POST'])
def get_ielts_question():
    """Get the next IELTS examiner question"""
    try:
        data = request.get_json()
        part = data.get('part', 1)
        topic = data.get('topic')
        memory = data.get('memory', {"history": []})
        
        # Call the external API
        api_url = f"{API_SPEAKING_URL}/get_ielts_question"
        response = requests.post(api_url, json={
            "part": part,
            "topic": topic,
            "memory": memory
        })
        
        if response.status_code == 200:
            try:
                result = response.json()
                
                # Extract topic from question if user didn't provide a custom topic
                question_text = result.get('question', '')
                if isinstance(question_text, list):
                    question_text = question_text[0]  # Take the first element if it's a list
                
                # Extract topic before cleaning the question format
                user_topic = request.json.get('topic')
                if not user_topic:  # Only save if user didn't select a custom topic
                    extracted_topic = extract_topic_from_question(question_text)
                    if extracted_topic:
                        part = request.json.get('part', 1)
                        save_topic_to_csv(extracted_topic, part)
                
                # Clean up the question format
                question_text = clean_question_format(question_text)
                result['question'] = question_text
                
                # Generate TTS audio for the question
                if question_text and question_text != 'Thank you for your response.':
                    print("Generating TTS audio for question...")
                    question_audio = generate_tts_audio(question_text)
                    if question_audio:
                        if isinstance(question_audio, list):
                            # Multiple audio chunks
                            question_audio_chunks_b64 = []
                            for i, chunk in enumerate(question_audio):
                                chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                                question_audio_chunks_b64.append(chunk_b64)
                            result['question_audio_chunks'] = question_audio_chunks_b64
                            print(f"Question TTS audio generated successfully - {len(question_audio)} chunks")
                        else:
                            # Single audio chunk
                            question_audio_b64 = base64.b64encode(question_audio).decode('utf-8')
                            result['question_audio'] = question_audio_b64
                            print("Question TTS audio generated successfully")
                    else:
                        print("Failed to generate question TTS audio")
                        result['question_audio'] = None
                
                return jsonify(result)
            except Exception as e:
                print(f"Error processing question response: {e}")
                return jsonify({'error': 'Failed to process question'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_ielts_response', methods=['POST'])
def analyze_ielts_response():
    """Analyze user's IELTS speaking response"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        part = int(request.form.get('part', 1))
        topic = request.form.get('topic')
        memory_str = request.form.get('memory', '{}')
        
        # Parse memory
        try:
            memory = json.loads(memory_str)
        except:
            memory = {"history": []}
        
        # Convert audio to WAV format if needed
        import tempfile
        import os
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
            temp_wav_path = temp_wav.name
        
        try:
            # Save the uploaded audio to temporary file
            audio_file.save(temp_wav_path)
            
            # Call the external API
            api_url = f"{API_SPEAKING_URL}/analyze_ielts_response"
            
            # Read the file content and create a new file object
            with open(temp_wav_path, 'rb') as wav_file:
                wav_content = wav_file.read()
            
            # Create a new file object for the request
            files = {'audio': ('response.wav', wav_content, 'audio/wav')}
            data = {
                'part': part,
                'topic': topic,
                'memory': json.dumps(memory)
            }
        
            print(f"Sending request to {api_url}")
            response = requests.post(api_url, files=files, data=data)
            print(f"Response status: {response.status_code}")
            
            # Clean up temporary file
            os.unlink(temp_wav_path)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"Response keys: {list(result.keys())}")
                    
                    # Generate TTS audio for grammar feedback if available
                    grammar_audio_b64 = None
                    if result.get('grammar_feedback') and result['grammar_feedback'] != 'Analysis failed':
                        print("Generating TTS audio for grammar feedback...")
                        
                        # Reorganize grammar feedback for better TTS
                        reorganized_grammar = reorganize_grammar_feedback(result['grammar_feedback'])
                        print(f"Original grammar feedback: {result['grammar_feedback']}")
                        print(f"Reorganized grammar feedback: {reorganized_grammar}")
                        
                        grammar_audio = generate_tts_audio(reorganized_grammar)
                        if grammar_audio:
                            if isinstance(grammar_audio, list):
                                # Multiple audio chunks
                                grammar_audio_chunks_b64 = []
                                for i, chunk in enumerate(grammar_audio):
                                    chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                                    grammar_audio_chunks_b64.append(chunk_b64)
                                result['grammar_audio_chunks'] = grammar_audio_chunks_b64
                                print(f"Grammar feedback TTS audio generated successfully - {len(grammar_audio)} chunks")
                            else:
                                # Single audio chunk
                                grammar_audio_b64 = base64.b64encode(grammar_audio).decode('utf-8')
                                result['grammar_audio'] = grammar_audio_b64
                                print(f"Grammar feedback TTS audio generated successfully - Size: {len(grammar_audio)} bytes")
                                print(f"Grammar feedback TTS audio base64 length: {len(grammar_audio_b64)}")
                        else:
                            print("Failed to generate grammar feedback TTS audio")
                    
                    # Generate TTS audio for next question if available
                    next_question_audio_b64 = None
                    next_question_text = result.get('next_question', '')
                    
                    # Handle case where next_question is a list
                    if isinstance(next_question_text, list):
                        next_question_text = next_question_text[0] if next_question_text else ''
                    
                    # Clean up the next question format
                    next_question_text = clean_question_format(next_question_text)
                    result['next_question'] = next_question_text
                    
                    if next_question_text and next_question_text != 'Thank you for your response.':
                        print("Generating TTS audio for next question...")
                        next_question_audio = generate_tts_audio(next_question_text)
                        if next_question_audio:
                            if isinstance(next_question_audio, list):
                                # Multiple audio chunks
                                next_question_audio_chunks_b64 = []
                                for i, chunk in enumerate(next_question_audio):
                                    chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                                    next_question_audio_chunks_b64.append(chunk_b64)
                                result['next_question_audio_chunks'] = next_question_audio_chunks_b64
                                print(f"Next question TTS audio generated successfully - {len(next_question_audio)} chunks")
                            else:
                                # Single audio chunk
                                next_question_audio_b64 = base64.b64encode(next_question_audio).decode('utf-8')
                                result['next_question_audio'] = next_question_audio_b64
                                print(f"Next question TTS audio generated successfully - Size: {len(next_question_audio)} bytes")
                                print(f"Next question TTS audio base64 length: {len(next_question_audio_b64)}")
                        else:
                            print("Failed to generate next question TTS audio")
                    
                    # Update result with generated audio
                    if 'grammar_audio_chunks' in result:
                        result['grammar_audio'] = None  # Clear single audio if chunks exist
                    elif 'grammar_audio' in result:
                        pass  # Keep single audio
                    
                    if 'next_question_audio_chunks' in result:
                        result['next_question_audio'] = None  # Clear single audio if chunks exist
                    elif 'next_question_audio' in result:
                        pass  # Keep single audio
                    
                    # Handle mispronounced words if available
                    if result.get('mispronounced'):
                        print(f"Found {len(result['mispronounced'])} mispronounced words")
                        # Add pronunciation feedback to the result
                        pronunciation_feedback = []
                        for word_data in result['mispronounced']:
                            word = word_data.get('word', '')
                            ref_pronunciation = word_data.get('ref', '')
                            rec_pronunciation = word_data.get('rec', '')
                            start_ms = word_data.get('clip_start_ms', 0)
                            end_ms = word_data.get('clip_end_ms', 0)
                            
                            feedback = f"Word: '{word}' - Expected: {ref_pronunciation}, Heard: {rec_pronunciation}"
                            pronunciation_feedback.append({
                                'word': word,
                                'expected': ref_pronunciation,
                                'heard': rec_pronunciation,
                                'start_ms': start_ms,
                                'end_ms': end_ms,
                                'feedback': feedback
                            })
                        
                        result['pronunciation_feedback'] = pronunciation_feedback
                        print(f"Added pronunciation feedback for {len(pronunciation_feedback)} words")
                    
                    # Handle zip_url for error clips if available
                    if result.get('zip_url'):
                        print(f"Found zip_url: {result['zip_url']}")
                        # The zip_url is already in the result, so it will be passed to frontend
                        # Frontend can use this to download error clips if needed
                        result['error_clips_available'] = True
                    else:
                        result['error_clips_available'] = False
                    
                    # Convert base64 audio back to blob for frontend (if available)
                    if result.get('grammar_audio'):
                        try:
                            grammar_audio_bytes = base64.b64decode(result['grammar_audio'])
                            # Convert bytes to base64 string for JSON serialization
                            result['grammar_audio_blob'] = base64.b64encode(grammar_audio_bytes).decode('utf-8')
                            print("Grammar audio converted successfully")
                        except Exception as e:
                            print(f"Grammar audio conversion error: {e}")
                            result['grammar_audio_blob'] = None
                    
                    if result.get('next_question_audio'):
                        try:
                            next_question_audio_bytes = base64.b64decode(result['next_question_audio'])
                            # Convert bytes to base64 string for JSON serialization
                            result['next_question_audio_blob'] = base64.b64encode(next_question_audio_bytes).decode('utf-8')
                            print("Next question audio converted successfully")
                        except Exception as e:
                            print(f"Next question audio conversion error: {e}")
                            result['next_question_audio_blob'] = None
                    
                    return jsonify(result)
                except Exception as e:
                    print(f"JSON parsing error: {e}")
                    print(f"Response content: {response.text[:500]}")
                    return jsonify({'error': f'JSON parsing failed: {str(e)}'}), 500
            else:
                print(f"API returned error: {response.status_code}")
                print(f"Response content: {response.text}")
                return jsonify({'error': f'Analysis failed: {response.status_code}'}), 500
                
        except Exception as e:
            print(f"Audio processing error: {e}")
            # Clean up temporary file if it exists
            if 'temp_wav_path' in locals():
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass
            return jsonify({'error': f'Audio processing failed: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Overall error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/best_answer', methods=['POST'])
def get_best_answer():
    """Get a model-generated best answer for the given question"""
    try:
        data = request.get_json()
        question = data.get('question')
        part = data.get('part', 1)
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Call the external API for best answer
        api_url = f"{API_SPEAKING_URL}/best_answer"
        response = requests.post(api_url, json={
            "question": question,
            "part": part
        })
        
        if response.status_code == 200:
            try:
                result = response.json()
                return jsonify(result)
            except Exception as e:
                print(f"Error processing best answer response: {e}")
                return jsonify({'error': 'Failed to process best answer'}), 500
        else:
            print(f"API returned error: {response.status_code}")
            print(f"Response content: {response.text}")
            return jsonify({'error': f'Best answer request failed: {response.status_code}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_tts', methods=['POST'])
def generate_tts():
    """Generate TTS audio for the given text"""
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Generate TTS audio using the existing function
        audio_content = generate_tts_audio(text)
        
        if audio_content:
            if isinstance(audio_content, list):
                # Multiple audio chunks
                audio_chunks_b64 = []
                for chunk in audio_content:
                    chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                    audio_chunks_b64.append(chunk_b64)
                return jsonify({'audio_chunks': audio_chunks_b64})
            else:
                # Single audio chunk
                audio_b64 = base64.b64encode(audio_content).decode('utf-8')
                return jsonify({'audio': audio_b64})
        else:
            return jsonify({'error': 'Failed to generate TTS audio'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_writing', methods=['POST'])
def evaluate_writing():
    """Evaluate writing using the external API"""
    try:
        data = request.get_json()
        topic = data.get('topic')
        essay = data.get('essay')
        
        if not topic or not essay:
            return jsonify({'error': 'Topic and essay are required'}), 400
        
        # Call the external writing evaluation API
        api_url = f"{API_WRITING_URL}/evaluate"
        
        payload = {
            "prompt": topic,
            "essay": essay
        }
        
        print(f"Writing evaluation request to: {api_url}")
        print(f"Topic: {topic}")
        print(f"Essay length: {len(essay)} characters")
        
        response = requests.post(api_url, json=payload, timeout=60)
        
        print(f"Writing evaluation response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Writing evaluation response: {result}")
                
                # Convert the API response to match frontend expectations
                if 'evaluation' in result:
                    return jsonify({'feedback': result['evaluation']})
                else:
                    return jsonify(result)
            except Exception as e:
                print(f"Error parsing writing evaluation response: {e}")
                return jsonify({'error': 'Failed to parse evaluation response'}), 500
        else:
            print(f"Writing evaluation API error: {response.status_code}")
            print(f"Response content: {response.text}")
            return jsonify({'error': f'Evaluation failed: {response.status_code}'}), 500
            
    except requests.exceptions.Timeout:
        print("Writing evaluation request timed out")
        return jsonify({'error': 'Evaluation request timed out'}), 500
    except requests.exceptions.ConnectionError as e:
        print(f"Writing evaluation connection error: {e}")
        return jsonify({'error': 'Connection error to evaluation service'}), 500
    except Exception as e:
        print(f"Writing evaluation error: {e}")
        return jsonify({'error': str(e)}), 500

# Existing upload route for speaking (can be updated later)
@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return redirect(url_for('speaking'))
    file = request.files['audio']
    if file.filename == '':
        return redirect(url_for('speaking'))
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return f"Audio uploaded: {file.filename}"
    return redirect(url_for('speaking'))

@app.route('/download/<filename>')
def download_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True) 