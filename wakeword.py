import openwakeword
from openwakeword.model import Model
import logging
import time
from inputdevice import init_global_audio_device, get_audio_frame, cleanup_global_audio_device

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize wakeword model
model = Model(
    wakeword_models=["hey_jarvis"],  # can also leave this argument empty to load all of the included pre-trained models
)

def start_wakeword_detection():
    """Bắt đầu wakeword detection"""
    logger.info("🎤 Starting wakeword detection...")
    
    # Initialize global audio device
    audio_device = init_global_audio_device(device_index=0, save_recording=False)
    
    try:
        logger.info("👂 Listening for wakeword 'hey jarvis'...")
        
        while True:
            # Get audio data containing 16-bit 16khz PCM audio data from microphone
            frame = audio_device.get_audio_frame_int16()
            
            if frame is not None:
                # Run wakeword detection
                prediction = model.predict(frame)
                
                # Check if wakeword detected
                for wakeword in prediction:
                    confidence = prediction[wakeword]
                    if confidence > 0.3:  # Threshold
                        logger.info(f"🔥 Wakeword detected: {wakeword} (confidence: {confidence:.3f})")
                        return True  # Wakeword detected!
            
            time.sleep(0.01)  # Small delay
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Wakeword detection stopped by user")
        return False
    finally:
        cleanup_global_audio_device()

# Example usage
if __name__ == "__main__":
    start_wakeword_detection()
