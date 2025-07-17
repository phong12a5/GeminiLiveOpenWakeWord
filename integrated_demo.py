"""
Integrated Wakeword + Gemini Live Demo
Tích hợp wakeword detection với Gemini Live theo logic timeout
"""

import asyncio
import logging
import time
import numpy as np
from google import genai
from google.genai import types
from dotenv import dotenv_values
from wakeword import WakeWordDetector
from inputdevice import AudioInputDevice

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DEFAUL_DEVICE_INDEX = 1
WAKEWORD_TIMEOUT = 10  # seconds

class IntegratedDemo:
    def __init__(self, input_device_index=DEFAUL_DEVICE_INDEX):
        # Load API key
        env_values = dotenv_values(".env")
        self.api_key = env_values.get("GOOGLE_API_KEY") or env_values.get("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ API key not found in .env file")
        
        logger.info(f"🔑 Using API key: {self.api_key[:10]}...")
        
        # Audio setup using shared AudioInputDevice
        self.input_device_index = input_device_index
        self.is_running = False
        
        # Create shared AudioInputDevice for both wakeword and Gemini
        self.audio_device = AudioInputDevice(
            input_device_index=input_device_index,
            save_recording=False  # Don't save recordings for this demo
        )
        
        # Wakeword setup using shared AudioInputDevice
        self.wakeword_detector = WakeWordDetector(
            # wakeword_models=["hey_jarvis"],
            #  wakeword_models=["hey_kai.tflite"],
             wakeword_models=["alexa"],
            audio_device=self.audio_device,  # Share the same AudioInputDevice
            threshold=0.5
        )
        
        # Gemini setup
        self.client = genai.Client(
            http_options={"api_version": "v1alpha"},
            api_key=self.api_key
        )
        
        self.config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            speech_config=types.SpeechConfig(
                language_code="en-US",
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
                )
            ),
            system_instruction="You are a helpful AI assistant. Respond briefly in English.",
            input_audio_transcription={},
        )
        
        # State management
        self.mode = "wakeword"  # "wakeword" or "gemini"
        self.gemini_session = None
        self.last_user_text_time = None
        self.user_activity_detected = False
    
    def start_audio_stream(self):
        """Start audio stream using shared AudioInputDevice"""
        if not self.audio_device.is_recording:
            self.audio_device.start_recording()
        self.is_running = True
        logger.info("🎤 Shared AudioInputDevice started")
    
    def stop_audio_stream(self):
        """Stop audio stream"""
        self.is_running = False
        if self.audio_device.is_recording:
            self.audio_device.stop_recording()
        logger.info("🔇 Shared AudioInputDevice stopped")
    
    def get_audio_frame(self):
        """Get audio frame from shared AudioInputDevice"""
        return self.audio_device.get_audio_frame_int16()
    
    async def wakeword_detection_loop(self):
        """Wakeword detection loop using shared AudioInputDevice"""
        logger.info("👂 Wakeword detection started - say 'hey jarvis'")
        
        # AudioInputDevice is already started by start_audio_stream()
        # WakeWordDetector will use the shared AudioInputDevice
        
        while self.is_running:
            if self.mode == "wakeword":
                try:
                    # Check for wakeword detection
                    detections = self.wakeword_detector.detect_wakeword()
                    
                    if detections:
                        for wakeword, confidence in detections.items():
                            logger.info(f"🔥 Wakeword detected: {wakeword} (confidence: {confidence:.3f})")
                            # Switch to Gemini mode
                            await self.switch_to_gemini_mode()
                            break
                    
                except Exception as e:
                    logger.error(f"❌ Wakeword detection error: {e}")
                    await asyncio.sleep(0.1)
            else:
                # In Gemini mode, just wait
                await asyncio.sleep(0.1)
    
    async def switch_to_gemini_mode(self):
        """Switch from wakeword to Gemini mode"""
        logger.info("🔄 Switching to Gemini mode...")
        self.mode = "gemini"
        self.last_user_text_time = time.time()
        self.user_activity_detected = False
        
        # Start Gemini session
        await self.start_gemini_session()
    
    async def switch_to_wakeword_mode(self):
        """Switch from Gemini to wakeword mode"""
        logger.info("🔄 Switching back to wakeword mode...")
        self.mode = "wakeword"
        
        # Stop Gemini session
        await self.stop_gemini_session()
        
        # Clear any pending audio in AudioInputDevice
        self.audio_device.clear_queue()
        logger.info("🔄 Audio queue cleared for wakeword mode")
    
    async def start_gemini_session(self):
        """Start Gemini Live session and audio loops"""
        try:
            async with self.client.aio.live.connect(
                model="models/gemini-2.0-flash-live-001",
                config=self.config
            ) as session:
                self.gemini_session = session
                logger.info("✅ Gemini Live session started")
                
                # Start Gemini audio and receive loops within the session context
                gemini_tasks = [
                    asyncio.create_task(self.gemini_audio_loop()),
                    asyncio.create_task(self.gemini_receive_loop()),
                ]
                
                # Wait until mode switches back to wakeword
                while self.mode == "gemini" and self.is_running:
                    await asyncio.sleep(0.1)
                
                # Cancel Gemini tasks when exiting Gemini mode
                for task in gemini_tasks:
                    if not task.done():
                        task.cancel()
                
                # Wait for tasks to complete cancellation
                await asyncio.gather(*gemini_tasks, return_exceptions=True)
                
                logger.info("🔇 Gemini Live session ended")

        except Exception as e:
            logger.error(f"❌ Failed to start Gemini session: {e}")
        finally:
            self.gemini_session = None
    
    async def stop_gemini_session(self):
        """Stop Gemini Live session"""
        # Session will be stopped automatically when the async context exits
        # This method now just ensures the mode is set correctly
        pass
    
    async def gemini_audio_loop(self):
        """Send audio to Gemini Live - runs within session context"""
        logger.info("📡 Gemini audio streaming started")
        try:
            while True:  # Loop until cancelled
                try:
                    # Get audio data for streaming (float32 format)
                    audio_data = self.audio_device.get_audio_for_streaming()
                    
                    if audio_data is not None and len(audio_data) > 0:
                        audio_int16 = (audio_data * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()
                        
                        # Send audio with correct format
                        await self.gemini_session.send_realtime_input(
                            audio={"data": audio_bytes, "mime_type": "audio/pcm;rate=16000"}
                        )
                    
                    await asyncio.sleep(0.05)  # Optimal delay
                    
                except Exception as e:
                    # Bỏ qua các lỗi keep-alive không quan trọng
                    if "sent 1000 (OK); then received 1000 (OK)" not in str(e):
                        logger.warning(f"⚠️ Gemini audio send error: {e}")
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.info("📡 Gemini audio streaming stopped")
    
    async def gemini_receive_loop(self):
        """Receive responses from Gemini Live - runs within session context"""
        logger.info("📡 Gemini response receiving started")
        try:
            while True:  # Loop until cancelled
                try:
                    turn = self.gemini_session.receive()
                    print(f"Received turn: {turn}")
                    
                    user_text = ""
                    ai_text = ""
                    
                    async for response in turn:
                        if response.server_content:
                            # Handle transcriptions
                            if response.server_content.input_transcription:
                                user_text += response.server_content.input_transcription.text
                                
                            if response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.text:
                                        ai_text += part.text
                            
                            if response.server_content.interrupted is True:
                                break
                    
                    # Process transcriptions
                    if user_text.strip():
                        logger.info(f"👤 You: {user_text.strip()}")
                        # Reset timer when user speaks
                        self.last_user_text_time = time.time()
                        self.user_activity_detected = True
                        
                    if ai_text.strip():
                        logger.info(f"🤖 AI: {ai_text.strip()}")
                    
                except Exception as e:
                    # Bỏ qua các lỗi keep-alive không quan trọng
                    if "sent 1000 (OK); then received 1000 (OK)" not in str(e):
                        logger.warning(f"⚠️ Gemini receive error: {e}")
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.info("📡 Gemini response receiving stopped")
    
    async def timeout_monitor_loop(self):
        """Monitor timeout in Gemini mode"""
        while self.is_running:
            if self.mode == "gemini" and self.last_user_text_time:
                current_time = time.time()
                elapsed = current_time - self.last_user_text_time
                
                if elapsed >= WAKEWORD_TIMEOUT:
                    if self.user_activity_detected:
                        logger.info(f"⏰ Timeout after user activity ({WAKEWORD_TIMEOUT}s) - switching back to wakeword mode")
                    else:
                        logger.info(f"⏰ Timeout with no user activity ({WAKEWORD_TIMEOUT}s) - switching back to wakeword mode")
                    
                    await self.switch_to_wakeword_mode()
            
            await asyncio.sleep(0.5)  # Check every 500ms
    
    async def run(self):
        """Run the integrated demo"""
        logger.info("🚀 Integrated Wakeword + Gemini Live Demo")
        logger.info("👂 Say 'hey jarvis' to activate")
        logger.info(f"⏰ Auto-timeout after {WAKEWORD_TIMEOUT}s of inactivity")
        logger.info("⏹️  Press Ctrl+C to stop")
        print()
        
        tasks = []
        
        try:
            # Start audio stream
            self.start_audio_stream()
            
            # Create async tasks - only wakeword and timeout monitor
            # gemini_audio_loop and gemini_receive_loop are started within start_gemini_session
            tasks = [
                asyncio.create_task(self.wakeword_detection_loop()),
                asyncio.create_task(self.timeout_monitor_loop()),
            ]
            
            logger.info("🎤 Ready! Say 'hey jarvis' to start...")
            
            # Run until interrupted
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"❌ Demo error: {e}")
        finally:
            # Cleanup
            await self.stop_gemini_session()
            self.stop_audio_stream()
            
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
            print("❌ Invalid device index. Use: python integrated_demo.py [device_number]")
            return
    
    try:
        demo = IntegratedDemo(input_device_index=input_device_index)
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
