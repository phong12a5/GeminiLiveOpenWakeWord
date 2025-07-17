"""
Gemini Live Audio Demo - Phiên bản hoàn chỉnh
Sử dụng API đúng và đã được test
"""

import asyncio
import logging
from google import genai
from google.genai import types
from dotenv import dotenv_values
import numpy as np
import sys
from inputdevice import AudioInputDevice

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiLiveDemo:
    def __init__(self, input_device_index=None):
        # Load API key
        env_values = dotenv_values(".env")
        self.api_key = env_values.get("GOOGLE_API_KEY") or env_values.get("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ API key not found in .env file")
        
        logger.info(f"🔑 Using API key: {self.api_key[:10]}...")
        
        # Audio setup với AudioInputDevice
        self.audio_device = AudioInputDevice(
            input_device_index=input_device_index,
            save_recording=True  # Lưu recording để debug
        )
    
        
        # Gemini setup
        self.client = genai.Client(
            http_options={"api_version": "v1alpha"},
            api_key=self.api_key
        )
        
        self.config = types.LiveConnectConfig(
            response_modalities=["TEXT"],  # Chỉ text trước để test
            speech_config =types.SpeechConfig(
                language_code="en-US",
                voice_config =types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
                )
            ),
            system_instruction="You are a helpful AI assistant. Respond briefly in English.",
            input_audio_transcription={},
        )
        
        self.session = None
    
    async def send_audio_loop(self):
        """Send audio to Gemini Live"""
        try:
            while self.session is None:
                await asyncio.sleep(0.1)
            
            logger.info("📡 Audio streaming started")
            sent_count = 0
            error_count = 0
            
            while self.audio_device.is_recording:
                try:
                    # Lấy audio data từ AudioInputDevice
                    audio_data = self.audio_device.get_audio_for_streaming()
                    
                    if audio_data is not None and len(audio_data) > 0:
                        # Convert numpy array to bytes for Gemini Live API
                        # Convert float32 to int16 PCM format
                        audio_int16 = (audio_data * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()
                        
                        # Send audio with correct format
                        await self.session.send_realtime_input(
                            audio={"data": audio_bytes, "mime_type": "audio/pcm;rate=16000"}
                        )
                        sent_count += 1
                        
                        if sent_count % 50 == 0:
                            logger.info(f"📤 Sent {sent_count} audio chunks")
                            
                except Exception as e:
                    error_count += 1
                    if error_count % 10 == 1:  # Log error mỗi 10 lần
                        logger.warning(f"⚠️ Audio send error: {e}")
                    await asyncio.sleep(0.1)
                    continue
                
                await asyncio.sleep(0.05)  # Optimal delay
                
        except asyncio.CancelledError:
            logger.info("📡 Audio streaming cancelled")
        except Exception as e:
            logger.error(f"❌ Audio streaming error: {e}")
    
    async def receive_loop(self):
        """Receive responses from Gemini Live"""
        try:
            while self.session is None:
                await asyncio.sleep(0.1)
            
            logger.info("📡 Response receiving started")
            
            while True:
                turn = self.session.receive()
                
                user_text = ""
                ai_text = ""
                
                async for response in turn:
                    if response.server_content:
                        # Handle transcriptions only
                        if response.server_content.input_transcription:
                            user_text += response.server_content.input_transcription.text
                            
                        if response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.text:
                                    ai_text += part.text

                        if response.server_content.interrupted is True:
                            logger.info("🔄 Response interrupted, waiting for next turn...")
                            break
                    
                
                # Log transcriptions
                if user_text.strip():
                    logger.info(f"👤 You: {user_text.strip()}")
                if ai_text.strip():
                    logger.info(f"🤖 AI: {ai_text.strip()}")
                    
        except asyncio.CancelledError:
            logger.info("📡 Response receiving cancelled")
        except Exception as e:
            logger.error(f"❌ Response receiving error: {e}")
    
    async def run(self):
        """Run the demo"""
        logger.info("🚀 Gemini Live Audio Demo")
        logger.info("🎯 Speak in English, AI will transcribe and respond with text")
        logger.info("⏹️  Press Ctrl+C to stop")
        print()
        
        tasks = []  # Khởi tạo tasks ở đây
        
        try:
            async with self.client.aio.live.connect(
                model="models/gemini-2.0-flash-live-001",
                config=self.config
            ) as session:
                self.session = session
                logger.info("✅ Connected to Gemini Live successfully!")
                
                # Start audio recording với AudioInputDevice
                self.audio_device.start_recording()
                
                # Create async tasks
                tasks = [
                    asyncio.create_task(self.send_audio_loop()),
                    asyncio.create_task(self.receive_loop()),
                ]
                
                logger.info("🎤 Ready! Start speaking...")
                
                # Run until interrupted
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            logger.error("💡 Please check:")
            logger.error("   1. API key is correct")
            logger.error("   2. Internet connection")
            logger.error("   3. Microphone permissions")
        finally:
            self.session = None
            # Stop audio recording
            self.audio_device.stop_recording()
            
            # Cancel tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

async def main():
    import sys
    
    # Parse command line argument for device selection
    input_device_index = None
    if len(sys.argv) > 1:
        try:
            input_device_index = int(sys.argv[1])
            print(f"🎤 Using device index: {input_device_index}")
        except ValueError:
            print("❌ Invalid device index. Use: python main3.py [device_number]")
            return
    
    try:
        demo = GeminiLiveDemo(input_device_index=input_device_index)
        await demo.run()
    except ValueError as e:
        print(f"{e}")
        print("💡 Create .env file with: GOOGLE_API_KEY=your_api_key")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        logger.info("👋 Demo completed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Stopped by user!")

