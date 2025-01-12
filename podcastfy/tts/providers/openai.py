"""OpenAI TTS provider implementation."""

import openai
from openai import OpenAI
from typing import List, Optional
from ..base import TTSProvider
import logging
import os

logger = logging.getLogger(__name__)

class OpenAITTS(TTSProvider):
    """OpenAI Text-to-Speech provider."""
    
    # Provider-specific SSML tags
    PROVIDER_SSML_TAGS: List[str] = ['break', 'emphasis']
    
    def __init__(self, api_key: Optional[str] = None, model: str = "tts-1-hd"):
        """
        Initialize OpenAI TTS provider.
        
        Args:
            api_key: OpenAI API key. If None, expects OPENAI_API_KEY env variable
            model: Model name to use. Defaults to "tts-1-hd"
        """
        if api_key:
            openai.api_key = api_key
        elif not openai.api_key:
            raise ValueError("OpenAI API key must be provided or set in environment")
        self.model = model
            
    def get_supported_tags(self) -> List[str]:
        """Get all supported SSML tags including provider-specific ones."""
        return self.PROVIDER_SSML_TAGS
        
    def generate_audio(self, text: str, voice: str, model: str, voice2: str = None) -> bytes:
        """Generate audio using OpenAI API."""
        self.validate_parameters(text, voice, model)
        timeout = int(os.getenv('OPENAI_API_TTS_TIMEOUT', 3600))
        voice_as_model = os.getenv('OPENAI_TTS_VOICE_AS_MODEL', None)
        base_url = os.getenv('OPENAI_TTS_BASE_URL', None)
        logger.info(f"timeout {timeout}, voice_as_model {voice_as_model}")
        if voice_as_model:
            extra_body = {"backend": model}
            model = voice
        else:
            extra_body = {}    
        
        client = OpenAI(timeout=timeout, base_url=base_url)

        try:
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                extra_body=extra_body
            )
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate audio: {str(e)}") from e