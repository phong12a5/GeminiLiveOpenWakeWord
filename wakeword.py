import openwakeword
from openwakeword.model import Model
import logging
import time
import numpy as np
from inputdevice import AudioInputDevice

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    WakeWord Detector class wrapping openwakeword functionality
    Uses AudioInputDevice for audio handling
    """
    
    def __init__(self, wakeword_models=None, audio_device=None, threshold=0.5):
        """
        Initialize WakeWord Detector
        
        Args:
            wakeword_models: List of wakeword model names (default: ["hey_jarvis"])
            audio_device: AudioInputDevice instance (if None, will create new one)
            threshold: Detection confidence threshold (0.0 - 1.0)
        """
        self.wakeword_models = wakeword_models or ["hey_jarvis"]
        self.threshold = threshold
        
        # Audio setup - use provided AudioInputDevice or create new one
        if audio_device is not None:
            self.audio_device = audio_device
            self.own_audio_device = False  # Don't manage the device lifecycle
        else:
            self.audio_device = AudioInputDevice(save_recording=False)
            self.own_audio_device = True  # We manage the device lifecycle
        
        # Wakeword setup
        self._initialize_model()
        
        logger.info(f"✅ WakeWordDetector initialized with models: {self.wakeword_models}")
    
    def _initialize_model(self):
        """Initialize openwakeword model"""
        try:
            # Download models if needed
            openwakeword.utils.download_models()
            
            # Initialize model
            self.model = Model(wakeword_models=self.wakeword_models)
            logger.info("✅ Wakeword model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize wakeword model: {e}")
            raise
    
    def start_listening(self):
        """Start audio stream and wakeword detection"""
        if self.own_audio_device and not self.audio_device.is_recording:
            self.audio_device.start_recording()
            logger.info("🎤 WakeWord detection started")
        elif not self.own_audio_device:
            logger.info("🎤 Using shared AudioInputDevice for WakeWord detection")
    
    def stop_listening(self):
        """Stop audio stream and wakeword detection"""
        if self.own_audio_device:
            self.audio_device.stop_recording()
            logger.info("🔇 WakeWord detection stopped")
    
    def get_audio_frame(self):
        """Get audio frame from AudioInputDevice"""
        return self.audio_device.get_audio_frame_int16()
    
    def detect_wakeword(self, audio_frame=None):
        """
        Detect wakeword in audio frame
        
        Args:
            audio_frame: Audio data (int16), if None will get from AudioInputDevice
            
        Returns:
            dict: {wakeword_name: confidence} or None if no detection
        """
        if audio_frame is None:
            audio_frame = self.get_audio_frame()
        
        if audio_frame is None:
            return None
        
        try:
            # Run wakeword detection
            prediction = self.model.predict(audio_frame)
            
            # Check for detections above threshold
            detections = {}
            for wakeword in prediction:
                confidence = prediction[wakeword]
                if confidence > self.threshold:
                    detections[wakeword] = confidence
                    self.model.reset()
            
            return detections if detections else None
            
        except Exception as e:
            logger.error(f"❌ Wakeword detection error: {e}")
            return None
    
    def listen_for_wakeword(self, callback=None):
        """
        Continuously listen for wakeword
        
        Args:
            callback: Function to call when wakeword detected (wakeword, confidence)
                     If None, will return when first wakeword detected
        
        Returns:
            tuple: (wakeword_name, confidence) if callback is None
        """
        if not self.audio_device.is_recording:
            self.start_listening()
        
        logger.info(f"👂 Listening for wakewords: {self.wakeword_models}")
        
        try:
            while self.audio_device.is_recording:
                detections = self.detect_wakeword()
                
                if detections:
                    for wakeword, confidence in detections.items():
                        logger.info(f"🔥 Wakeword detected: {wakeword} (confidence: {confidence:.3f})")
                        
                        if callback:
                            callback(wakeword, confidence)
                        else:
                            return wakeword, confidence
                
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Wakeword detection stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in wakeword detection loop: {e}")
        finally:
            if not callback:  # Only stop if not using callback mode
                self.stop_listening()
        
        return None
    
    def __enter__(self):
        """Context manager support"""
        self.start_listening()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop_listening()


# Example usage
if __name__ == "__main__":
    # Method 1: Simple usage with own AudioInputDevice
    detector = WakeWordDetector(threshold=0.3)
    
    try:
        result = detector.listen_for_wakeword()
        if result:
            wakeword, confidence = result
            print(f"✅ Detected: {wakeword} with confidence {confidence:.3f}")
    finally:
        detector.stop_listening()
    
    # Method 2: Context manager
    # with WakeWordDetector(threshold=0.3) as detector:
    #     result = detector.listen_for_wakeword()
    #     if result:
    #         wakeword, confidence = result
    #         print(f"✅ Detected: {wakeword} with confidence {confidence:.3f}")
    
    # Method 3: Using shared AudioInputDevice
    # from inputdevice import AudioInputDevice
    # with AudioInputDevice() as audio_device:
    #     detector = WakeWordDetector(audio_device=audio_device, threshold=0.3)
    #     result = detector.listen_for_wakeword()
    #     if result:
    #         wakeword, confidence = result
    #         print(f"✅ Detected: {wakeword} with confidence {confidence:.3f}")
