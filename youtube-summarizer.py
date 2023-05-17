from youtube_transcript_api import YouTubeTranscriptApi
import openai
import requests
import textwrap

# Set your OpenAI API key
openai.api_key = 'your-openai-key'

def youtube_transcript(video_url):
    # Extract video_id from the URL
    video_id = video_url.split('watch?v=')[1]

    # Get the video title
    r = requests.get(video_url)
    start = r.text.find('<title>') + len('<title>')
    end = r.text.find('</title>')
    video_title = r.text[start:end].replace(' - YouTube', '')
    
    # Get the transcript
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

    # Concatenate the transcript texts
    full_transcript = " ".join([t['text'] for t in transcript])
    
    return video_title, full_transcript


def gpt_chat(prompt, transcript):
    # Split the transcript into chunks of 8000 tokens
    transcript_chunks = textwrap.wrap(transcript, 8000)

    # Initialize chat models and response
    response = ''

    # Process each chunk
    for chunk in transcript_chunks:
        chat_models = openai.ChatCompletion.create(
          model="gpt-4",
          messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chunk},
            ]
        )
        
        # Append the assistant's reply
        response += chat_models['choices'][0]['message']['content']

    # Return the assistant's reply
    return response

# Read file with YouTube URLs
with open('youtube_links.txt', 'r') as f:
    video_urls = [line.strip() for line in f.readlines()]

# Input prompt
prompt = "Here is a transcript from a video. Can you summarize the main points?"

# Process each video
for video_url in video_urls:
    title, transcript = youtube_transcript(video_url)
    if transcript:
        response = gpt_chat(prompt, transcript)
        print(f"Title: {title}\nURL: {video_url}\nResponse: {response}\n")
