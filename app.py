from flask import Flask, render_template, request, jsonify
import os
import requests
import random
import time
import uuid
from gtts import gTTS
from pydub import AudioSegment
from werkzeug.utils import secure_filename
import google.generativeai as genai
from markdown import markdown
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting


app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": ""}  

# Initialize Vertex AI
vertexai.init(project="commanding-ring-441505-g7", location="us-central1")

def generate(video_data):
    model = GenerativeModel("gemini-1.5-flash-002")

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
    ]

    video_part = Part.from_data(mime_type="video/mp4", data=video_data)

    responses = model.generate_content(
        [video_part, """Please analyze the video to assess whether the individual exhibits any characteristics commonly associated with autism spectrum disorder (ASD). Focus on signs such as difficulties in social interactions (e.g., challenges with eye contact, conversational reciprocity, or understanding social cues), repetitive behaviors (e.g., hand-flapping, strict adherence to routines, or intense focus on specific interests), sensory sensitivities (e.g., discomfort with loud noises, certain textures, or bright lights), and limited social reciprocity (e.g., difficulty understanding or responding to others' emotions). Based on your observations, determine if any of these traits are present and whether they align with common ASD behaviors."""],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    # Collect all responses and combine them into one result
    result = ""
    
    for response in responses:
        result += response.text

    return result


# Hugging Face API and get transcription
def query_huggingface_api(audio_file_path):
    with open(audio_file_path, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

# Store start time globally (or per session if required)
start_time = 0

@app.route('/')
def index():
    global start_time
    # Display text for user to read and record
    sentences = [
        # Paragraph 1
        "The labyrinthine corridors of the ancient library were shrouded in a heavy silence, save for the occasional rustling of yellowed pages being turned by a diligent scholar. Dust motes danced in the shafts of light that pierced through the tall, arched windows, illuminating rows upon rows of tomes that held the secrets of civilizations long past. Each step echoed softly, as though the very walls were absorbing the sound, adding to the atmosphere of quiet reverence.",
        
        # Paragraph 2
        "As the storm raged outside, the ship's crew worked tirelessly to secure the sails and stabilize the vessel against the unforgiving waves. The sea, a furious mass of churning water, seemed determined to pull them under. The captain, standing at the helm, barked orders to his crew, his voice barely audible above the howling wind. Each man knew that in this battle against nature, even the smallest mistake could mean disaster.",
        
        # Paragraph 3
        "The theory of relativity, proposed by Albert Einstein in the early 20th century, revolutionized our understanding of space, time, and gravity. According to this theory, space and time are not fixed entities but are intertwined in a four-dimensional fabric called spacetime. Massive objects, such as stars and planets, distort this fabric, creating what we perceive as gravity. This profound concept challenged centuries of Newtonian physics and opened the door to new discoveries about the nature of the universe.",
        
        # Paragraph 4
        "In the deep recesses of the rainforest, where the dense canopy blocked out the sun's rays, life thrived in hidden and unexpected ways. Giant ferns and twisting vines clung to ancient trees, while colorful birds and insects darted through the underbrush. The air was thick with the scent of damp earth and decaying foliage, a testament to the cycle of life that continued in this remote and untouched part of the world. Every step taken revealed new wonders, from tiny frogs no larger than a thumbnail to orchids with blossoms as wide as a hand.",
        
        # Paragraph 5
        "The intricacies of quantum mechanics defy our common-sense understanding of the world. At the quantum level, particles behave in ways that seem impossible when compared to the classical laws of physics. For example, electrons can exist in a state of superposition, meaning they can be in multiple places at once until they are observed. This phenomenon, famously illustrated by the thought experiment of Schrödinger's cat, suggests that reality itself may be far more complex and mysterious than we once thought.",
        
        # Paragraph 6
        "As the artist stood before the massive canvas, brush in hand, they were overcome by a sense of both trepidation and exhilaration. Each stroke of the brush was a deliberate act, a merging of color and form that would eventually coalesce into a representation of their inner world. The blank canvas, once intimidating, became a playground for experimentation, where the boundaries of reality and imagination blurred. With each new layer of paint, the image took on a life of its own, becoming something greater than the sum of its parts.",
        
        # Paragraph 7
        "The ethical implications of artificial intelligence have been a topic of intense debate among scientists, ethicists, and lawmakers. While AI holds the potential to revolutionize industries such as healthcare, finance, and transportation, it also raises concerns about privacy, security, and autonomy. As machines become more capable of making decisions and learning from data, the line between human and machine intelligence begins to blur. This has led to questions about responsibility and accountability in a world where AI plays an increasingly prominent role.",
        
        # Paragraph 8
        "The expansive desert stretched out before the travelers, a seemingly endless sea of sand dunes that undulated like waves frozen in time. The heat was oppressive, the sun beating down relentlessly on the barren landscape. Despite the desolation, there was a strange beauty to the desert, with its ever-shifting sands and the occasional oasis providing a glimpse of life in an otherwise unforgiving environment. The travelers knew that to survive, they would need to rely on both their physical endurance and their wits, for the desert was as treacherous as it was mesmerizing.",
        
        # Paragraph 9
        "In the realm of neuroscience, the human brain remains one of the most complex and least understood structures in existence. Comprising billions of neurons and trillions of synaptic connections, the brain is responsible for every thought, emotion, and action we experience. Despite advancements in brain imaging technology, many of the brain's functions remain a mystery, and scientists continue to grapple with questions about consciousness, memory, and perception. The study of the brain is, in many ways, the final frontier of human knowledge.",
        
        # Paragraph 10
        "Beneath the surface of the ocean lies a world that is as alien as any found in science fiction. In the deep sea, where sunlight cannot penetrate, creatures have evolved in strange and fascinating ways. Bioluminescent organisms glow in the darkness, using light to attract prey or communicate with one another. Gigantic squids with eyes the size of dinner plates swim silently through the inky depths, while bizarre fish with translucent bodies and rows of sharp teeth lurk in the shadows. This hidden world, largely unexplored, holds the potential to unlock new discoveries about life on Earth."
    ]


    index = random.randint(0, len(sentences)-1)
    sample_text = sentences[index]
    
    # Record the start time
    start_time = time.time()
    
    return render_template('index.html', sample_text=sample_text)

@app.route('/upload', methods=['POST'])
def upload_audio():
    global start_time
    if 'audio' not in request.files or 'original_text' not in request.form:
        return jsonify({'error': 'No audio file or original text found'}), 400

    # Save uploaded audio file
    audio_file = request.files['audio']
    audio_path = os.path.join("uploads", audio_file.filename)
    audio_file.save(audio_path)

    # Calculate the reading duration
    end_time = time.time()
    reading_duration = end_time - start_time

    # original text from the form data
    original_text = request.form['original_text']

    # Check the validity of reading time based on sentence length
    is_valid_time, min_time, max_time = validate_reading_time(original_text, reading_duration)

    # Send the audio file to Hugging Face API for transcription
    transcription_result = query_huggingface_api(audio_path)

    # Extract transcription from the response
    transcription = transcription_result.get('text', '')

    # Compare the transcribed text with the original
    comparison_score = compare_texts(original_text, transcription)
    comparison_score = 100 - comparison_score

    # Check if the reading is too fast or too slow
    autism_flag = not is_valid_time

    # Return the results as JSON
    return jsonify({
        'original_text': original_text,
        'transcribed_text': transcription,
        'score': comparison_score,
        'autism_flag': autism_flag,
        'reading_duration': round(reading_duration, 2),
        'expected_min_time': round(min_time, 2),
        'expected_max_time': round(max_time, 2),
        'autism_message': "Reading time suggests autism" if autism_flag else "Reading time is normal"
    })

def validate_reading_time(original_text, reading_duration):
    """
    Validates whether the reading time is within an acceptable range based on the sentence length.
    If the reading duration is too short or too long, the function returns False along with the expected time range.
    """
    words = original_text.split()
    num_words = len(words)

    # Define a reasonable reading speed range (in seconds per word)
    min_speed_per_word = 0.3  # minimum time (seconds) per word
    max_speed_per_word = 1.0  # maximum time (seconds) per word

    # Calculate acceptable time range based on number of words
    min_time = num_words * min_speed_per_word
    max_time = num_words * max_speed_per_word

    # Check if reading duration is within the acceptable range
    if min_time <= reading_duration <= max_time:
        return True, min_time, max_time  # Reading time is valid
    else:
        return False, min_time, max_time  # Reading time is too fast or too slow

def compare_texts(original_text, transcribed_text):
    original_words = set(original_text.lower().split())
    transcribed_words = set(transcribed_text.lower().split())
    common_words = original_words.intersection(transcribed_words)

    score = 1 - len(common_words) / len(original_words)
    return round(score * 100, 2)

@app.route('/convert_text_to_speech', methods=['POST'])
def convert_text_to_speech():
    text = request.form['text_to_convert']

    # Ensure the 'static' directory exists
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    output_file = os.path.join(static_dir, "output.mp3")

    # Generate the speech with gTTS
    tts = gTTS(text=text, lang='en', slow=True)
    tts.save(output_file)

    # Load the generated audio file
    audio = AudioSegment.from_mp3(output_file)

    # Slow down the audio further (adjust the multiplier as needed)
    slowed_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * 0.75)})
    slowed_audio = slowed_audio.set_frame_rate(audio.frame_rate)

    # Save the slowed audio to a new file
    slowed_output_file = os.path.join(static_dir, f"{str(uuid.uuid4())}-output.mp3")
    slowed_audio.export(slowed_output_file, format="mp3")

    return jsonify({'audio_url': slowed_output_file})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file found'}), 400

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read video file as binary data
    video_data = video_file.read()

    try:
        # Call the generate function to analyze the video
        print("Generating analysis using Vertex AI...")
        analysis_result = generate(video_data)
        #analysis_result = markdown(analysis_result)
        return jsonify({'analysis': analysis_result})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Video processing failed'}), 500
    

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=False)
