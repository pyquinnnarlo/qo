import speech_recognition as sr
import requests
import os
import sys
import json

# --- CONFIGURATION ---
# "qwen2.5:0.5b" is 5x faster than phi3 on Raspberry Pi
# Run: ollama pull qwen2.5:0.5b
OLLAMA_MODEL = "qwen2.5:0.5b"  
MIC_INDEX = 1               

def speak(text):
    """Speaks text immediately using espeak"""
    print(f"Robot: {text}")
    # Escape quotes to prevent errors
    safe_text = text.replace("'", "").replace('"', "").strip()
    if not safe_text: return
    # Speak fast (-s160)
    os.system(f'espeak -ven+m3 -s160 "{safe_text}" 2>/dev/null')

def listen():
    """Listens for audio"""
    r = sr.Recognizer()
    # Use the specific index if you know it, or leave blank for default
    mic = sr.Microphone(device_index=MIC_INDEX)
    
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=0.5) # Fast adapt
        r.dynamic_energy_threshold = True
        
        print("\nListening...")
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=6)
            print("Processing...")
            text = r.recognize_google(audio).lower()
            print(f"You: {text}")
            return text
        except Exception:
            return None

def stream_and_speak(prompt):
    """Streams the response from Ollama and speaks sentence-by-sentence"""
    url = "http://localhost:11434/api/generate"
    
    # We tell the AI to be brief to speed up generation
    full_prompt = f"You are a helpful voice assistant. Keep your answer to 1 or 2 short sentences. User said: {prompt}"
    
    data = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": True  # <--- THIS IS THE KEY
    }
    
    print("Thinking...", end="", flush=True)
    
    try:
        response = requests.post(url, json=data, stream=True)
        
        buffer = ""
        for line in response.iter_lines():
            if line:
                # Decode the chunk
                decoded_line = line.decode('utf-8')
                json_chunk = json.loads(decoded_line)
                
                if 'response' in json_chunk:
                    word = json_chunk['response']
                    buffer += word
                    
                    # If we hit a punctuation mark, speak the buffer immediately
                    if word in ['.', '?', '!', '\n']:
                        speak(buffer)
                        buffer = ""  # Clear buffer for next sentence
        
        # Speak any remaining text
        if buffer.strip():
            speak(buffer)
            
    except Exception as e:
        print(f"\nError: {e}")

# --- MAIN LOOP ---
if __name__ == "__main__":
    speak("I am ready.")
    
    while True:
        user_input = listen()
        
        if user_input:
            if "stop" in user_input or "exit" in user_input:
                speak("Goodbye.")
                sys.exit()
                
            # Use the new streaming function
            stream_and_speak(user_input)
