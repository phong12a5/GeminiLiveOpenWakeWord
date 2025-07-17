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
    wakeword_models=["hey_kai.tflite"],  # can also leave this argument empty to load all of the included pre-trained models
)

def start_wakeword_detection():
    """Bắt đầu wakeword detection"""
    logger.info("🎤 Starting wakeword detection...")
    
    # Initialize global audio device
    audio_device = init_global_audio_device(device_index=1, save_recording=False)
    
    try:
        logger.info("👂 Listening for wakeword 'hey kai'...")
        
        while True:
            # Get audio data containing 16-bit 16khz PCM audio data from microphone
            frame = get_audio_frame()
            
            if frame is not None:
                # Run wakeword detection
                prediction = model.predict(frame)
                
                # Check if wakeword detected
                for wakeword in prediction:
                    if prediction[wakeword] > 0.5:  # Threshold
                        logger.info(f"🔥 Wakeword detected: {wakeword} (confidence: {prediction[wakeword]:.2f})")
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
