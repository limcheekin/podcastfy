# %% [markdown]
# # Podcastfy 
# Transforming Multimodal Content into Captivating Multilingual Audio Conversations with GenAI:

# %% [markdown]
# In this Notebook, we will go through the main features of Podcastfy via its Python package:
# 
# - Support multiple input sources (text, images, websites, YouTube, and PDFs).
# - Generate short (2-5 minutes) or longform (30+ minutes) podcasts.
# - Customize transcript and audio generation (e.g., style, language, structure).
# - Generate transcripts using 100+ cloud-basedLLM models (OpenAI, Anthropic, Google etc).
# - Leverage local LLMs for transcript generation for increased privacy and control.
# - Integrate with advanced text-to-speech models (OpenAI, ElevenLabs, Microsoft Edge, Google single and multispeaker).
# - Provide multi-language support for global content creation.

# %% [markdown]
# ## Table of Contents
# 
# - Setup
# - Getting Started
# - Generate a podcast from text content
#   - Generate podcast from URLs
#   - Selecting TTS models
#   - Generate longform podcasts
#   - Generate transcript only
#   - Generate audio from transcript
#   - Generate podcast from pdf
#   - Raw text as input
#   - Podcast from topic / grounded web search
# - Generate podcast from images
# - Conversation Customization
# - Multilingual Support
#   - French (fr)
#   - Portugue (pt-br)
# - Custom LLM Support

# %% [markdown]
# ## Setup
# 
# Firstly, please make sure you have installed the podcastfy module, its dependencies and associated API keys. [See Setup](README.md#setup).

# %% [markdown]
# ## Getting Started

# %% [markdown]
# Podcast generation is done through the `generate_podcast` function.

# %%
from podcastfy.client import generate_podcast

# %%
def generate_and_embed_podcast(**kwargs):
    """
    Generates a podcast and embeds it in the notebook.
    """
    print(kwargs)
    transcript_only = kwargs.pop('transcript_only', False)
    transcript_file = generate_podcast(
        llm_model_name='llama-3.2-3b-instruct-q4_k_m', 
        api_key_label="OPENAI_API_KEY",
        transcript_only=True,
        **kwargs
    )
    if transcript_only:
        return transcript_file
    else:    
        audio_file = generate_podcast(
            transcript_file=transcript_file,
            tts_model="openai"
        )
        print(audio_file)
        return audio_file

# %% [markdown]
# ## Generate podcast from URL
# ## Generate longform podcasts
# 
# 
# By default, Podcastfy generates shortform podcasts (2-5 minutes). However, users can generate longform podcasts (20-30+ minutes) by setting the `longform` parameter to `True`. Note: Images are not yet supported for longform podcast generation.
# 
# 

# %% [markdown]
# In this example, we generate a longform podcast from the book "The Autobiography of Benjamin Franklin":

# %%
generate_and_embed_podcast(urls=["https://www.gutenberg.org/cache/epub/20203/pg20203.txt"], 
                           longform=True)

# %% [markdown]
# LLMs have a limited ability to output long text responses. Most LLMs have a `max_output_tokens` of around 4096 and 8192 tokens. Hence, long-form podcast transcript generation is challeging. We have implemented a technique I call "Content Chunking with Contextual Linking" to enable long-form podcast generation by breaking down the input content into smaller chunks and generating a conversation for each chunk while ensuring the combined transcript is coherent and linked to the original input.
# 
# ### Adjusting longform podcast length
# 
# Users may adjust lonform podcast length by setting the following parameters in your customization params (see later section "Conversation Customization"):
# - `max_num_chunks` (default: 7): Sets maximum number of rounds of discussions.
# - `min_chunk_size` (default: 600): Sets minimum number of characters to generate a round of discussion.
# 
# We define "round of discussion" as the output transcript obtained from a single LLM call. The higher the `max_num_chunks` and the lower the `min_chunk_size`, the longer the generated podcast will be.
# Today, this technique allows users to generate long-form podcasts of any length if input content is long enough. However, the conversation quality may decrease and its length may converge to a maximum if `max_num_chunks`/`min_chunk_size` is to high/low particularly if input content length is limited.
# 
# Recommendation:
# - If input content is short (1-10 paragraphs), generate shortform podcast (`longform=False`, which is the default).
# - If input content is long (10+ paragraphs), generate longform podcast (`longform=True`).
# - If input content is very long (e.g. long pdfs, books, series of websites), consider increasing `max_num_chunks` from default 7 to e.g. 10 or 15.
# 

# %% [markdown]
# ## Generate transcript
# 
# Users have the option to generate the transcript only from input urls, i.e. without audio generation. In that way, users may edit/process transcripts before further downstream audio generation.

# %%
# Generate transcript only
transcript_file = generate_and_embed_podcast(
	urls=["https://github.com/souzatharsis/podcastfy/blob/main/README.md"],
	transcript_only=True
)

# %%

print(f"Transcript generated and saved as: {transcript_file}")
# Read and print the first 20 characters from the transcript file
with open(transcript_file, 'r') as file:
	transcript_content = file.read(100)
	print(f"First 100 characters of the transcript: {transcript_content}")

# %% [markdown]
# ## Generate podcast from pdf
# 
# One or many pdfs can be processed in the same way as urls by simply passing a corresponding file path.

# %%
audio_file_from_pdf = generate_and_embed_podcast(urls=["./data/pdf/s41598-024-58826-w.pdf"])

# %% [markdown]
# This is a Scientific Reports article about climate change in France. Let's listen to this short-form podcast:

# %% [markdown]
# ## Generate podcast from raw text
# 
# Users can generate a podcast from raw text input.
# 

# %%
raw_text = "The wonderful world of LLMs."
generate_and_embed_podcast(text=raw_text)

# %% [markdown]
# Note that if input text is short, the generated podcast may be too short to be interesting. Further, generating a longform podcast from short input text may lead to low-quality conversations.

# %% [markdown]
# ## Generate podcast from topic
# 
# Users can also generate a podcast from a specific topic of interest, e.g. "Latest News in U.S. Politics" or "Modern art in the 1920s". Podcastfy will generate a podcast based on *grounded* real-time information about the most recent content published on the web about the topic.

# %%
audio_file_from_topic = generate_and_embed_podcast(topic="Latest news about OpenAI")

# %% [markdown]
# The generate conversation captures the rapid pace of OpenAI's developments as of today (11/16/2024) including leaked e-mail between Elon Musk and Sam Altman.

# %% [markdown]
# The difference between generating a podcast from a topic and from raw text is that a topic-based podcast is more likely to be grounded in real-time events and news, whereas a raw text-based podcast may not be as current or relevant. Note that a topic-based podcast won't necessarily generate a conversation about most recent events, instead it will consider the most relevant results from a web search. If the user would like to generate a conversation about recent events, please add such such information to properly instruct the LLM, for instance by adding "Latest News on..." to your query.

# %% [markdown]
# ## Generate podcast from images
# 
# Images can be provided as input to generate a podcast. This feature is currently only supported for shortform podcasts. 
# 
# This can be useful when users want to generate a podcast from images such as works of art, physical spaces, historical events, etc. One or many images can be provided as input. The following example generates a podcast from two images: Senecio, 1922 (Paul Klee) and Connection of Civilizations (2017) by Gheorghe Virtosu.
# 

# %%
# Generate podcast from input images
image_paths = [
        "https://raw.githubusercontent.com/souzatharsis/podcastfy/refs/heads/main/data/images/Senecio.jpeg",
        "https://raw.githubusercontent.com/souzatharsis/podcastfy/refs/heads/main/data/images/connection.jpg",
]

audio_file_from_images = generate_and_embed_podcast(image_paths=image_paths)

print("Podcast generated from images:", audio_file_from_images)

# %% [markdown]
# Here is the generated podcast, which we have pre-saved in the data directory.

# %% [markdown]
# ## Customization
# 
# Podcastfy offers a range of customization options to tailor your AI-generated podcasts. Whether you're creating educational content, storytelling experiences, or anything in between, these configuration options allow you to fine-tune your podcast's tone, style, and format.
# See [Conversation Configuration](usage/conversation_custom.md) for more details.
# 

# %%
# Example: In-depth Tech Debate Podcast

# Define a custom conversation config for a tech debate podcast
tech_debate_config = {
    'conversation_style': ['Engaging', 'Fast-paced', 'Enthusiastic', 'Educational'], 
    'roles_person1': 'Interviewer', 
    'roles_person2': 'Subject matter expert', 
    'dialogue_structure': ['Topic Introduction', 'Summary of Key Points', 'Discussions', 'Q&A Session', 'Farewell Messages'], 
    'podcast_name': 'Supernova Podcast', 
    'podcast_tagline': 'The future of intelligence', 
    'output_language': 'English', 
    'user_instructions': 'Make if fun and engaging', 
    'engagement_techniques': ['Rhetorical Questions', 'Personal Testimonials', 'Quotes', 'Anecdotes', 'Analogies', 'Humor'], 
    'creativity': 0.75
}

# Generate a tech debate podcast about artificial intelligence
tech_debate_podcast = generate_and_embed_podcast(
    urls=["https://en.wikipedia.org/wiki/Artificial_intelligence", 
          "https://en.wikipedia.org/wiki/Ethics_of_artificial_intelligence"],
    conversation_config=tech_debate_config,
)

print("Tech Debate Podcast generated:", tech_debate_podcast)

