"""
Simple Voice Handler for manual recording control
Provides start/stop recording interface for Streamlit UI
"""
import queue
import time
import sounddevice as sd
import numpy as np
from loguru import logger

from .asr_module import ASRClient, pcm16_to_wav_bytes
from .llm_module import LLMClient
from .tts_module import KokoroTTSClient
from .state_manager import ConversationState


class SimpleVoiceHandler:
    """
    Simplified voice handler with manual recording controls
    """
    def __init__(self, persona: dict, callbacks: dict = None):
        self.persona = persona
        self.callbacks = callbacks or {}
        
        # Initialize components
        self.asr_client = ASRClient()
        self.llm_client = LLMClient()
        self.tts_client = KokoroTTSClient()
        
        # State
        self.state = ConversationState(
            session_id="streamlit_session",
            persona_name=persona.get("name", "Assistant")
        )
        
        # Recording
        self.sample_rate = 16000
        self.audio_buffer = queue.Queue()
        self.stream = None
        self.is_recording = False
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream"""
        if self.is_recording:
            self.audio_buffer.put(indata.copy())
    
    def start_recording(self):
        """Start recording audio"""
        try:
            # Clear buffer
            while not self.audio_buffer.empty():
                self.audio_buffer.get()
            
            # Start stream
            self.is_recording = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                callback=self._audio_callback
            )
            self.stream.start()
            
            if self.callbacks.get('status'):
                self.callbacks['status']("🎤 Recording...")
                
            logger.info("Recording started")
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            if self.callbacks.get('error'):
                self.callbacks['error'](str(e))
    
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        logger.info("Recording stopped")
    
    def process_voice_input(self):
        """
        Process recorded audio through ASR -> LLM -> TTS pipeline
        Returns metrics dict
        """
        metrics = {}
        start_total = time.time()
        
        try:
            # Stop recording
            self.stop_recording()
            
            # Collect audio data
            if self.callbacks.get('status'):
                self.callbacks['status']("🎯 Transcribing...")
            
            audio_chunks = []
            while not self.audio_buffer.empty():
                audio_chunks.append(self.audio_buffer.get())
            
            if not audio_chunks:
                return {"error": "No audio recorded"}
            
            # Combine audio
            audio_data = np.concatenate(audio_chunks, axis=0)
            
            # ASR: Convert to text
            start_asr = time.time()
            pcm_bytes = audio_data.flatten().tobytes()
            wav_bytes = pcm16_to_wav_bytes(pcm_bytes, self.sample_rate)
            user_text, asr_ms = self.asr_client.transcribe_wav_bytes(wav_bytes)
            metrics['asr_ms'] = asr_ms
            
            if not user_text or not user_text.strip():
                return {"error": "No speech detected"}
            
            logger.info(f"User said: {user_text}")
            
            # Update state
            self.state.add_turn("user", user_text)
            
            # Notify user text
            if self.callbacks.get('user_text'):
                self.callbacks['user_text'](user_text)
            
            # LLM: Generate response
            if self.callbacks.get('status'):
                self.callbacks['status']("🤖 Thinking...")
            
            if self.callbacks.get('llm_start'):
                self.callbacks['llm_start'](user_text)
            
            start_llm = time.time()
            messages = self.state.as_messages(self.persona.get("system_prompt", "You are a helpful assistant."))
            
            # Generate LLM response
            assistant_text, llm_ms, usage = self.llm_client.complete(messages)
            metrics['llm_ms'] = llm_ms
            
            if not assistant_text.strip():
                return {"error": "No response generated"}
            
            logger.info(f"Assistant said: {assistant_text}")
            
            # Update state
            self.state.add_turn("assistant", assistant_text)
            
            # Notify assistant text
            if self.callbacks.get('assistant_text'):
                self.callbacks['assistant_text'](assistant_text)
            
            # TTS: Convert to speech
            if self.callbacks.get('status'):
                self.callbacks['status']("🔊 Speaking...")
            
            start_tts = time.time()
            # Split into sentences and speak
            import re
            sentences = re.split(r'(?<=[.!?])\s+', assistant_text)
            for sentence in sentences:
                if sentence.strip():
                    wav_bytes = self.tts_client.synthesize_sentence(sentence)
                    self.tts_client.playback.play_wav(wav_bytes)
            tts_ms = (time.time() - start_tts) * 1000
            metrics['tts_ms'] = tts_ms
            
            # Total time
            total_ms = (time.time() - start_total) * 1000
            metrics['total_ms'] = total_ms
            
            logger.info(f"Metrics: ASR={asr_ms:.0f}ms, LLM={llm_ms:.0f}ms, TTS={tts_ms:.0f}ms, Total={total_ms:.0f}ms")
            
            # Notify metrics
            if self.callbacks.get('metrics'):
                self.callbacks['metrics'](metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error processing voice input: {e}")
            if self.callbacks.get('error'):
                self.callbacks['error'](str(e))
            return {"error": str(e)}
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.state.turns = []
        logger.info("Conversation reset")
    
    def cleanup(self):
        """Clean up resources"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        # Stop any playback
        self.tts_client.playback.stop()
        logger.info("Cleanup complete")
