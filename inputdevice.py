"""
Audio Input Device Handler
Xử lý microphone và audio recording cho các module khác sử dụng
"""

import sounddevice as sd
import logging
import queue
import numpy as np
import soundfile as sf
import os
from datetime import datetime
import threading
import time

# Setup logging
logger = logging.getLogger(__name__)

# Audio config
SAMPLE_RATE = 16000  # Standard for speech recognition
CHANNELS = 1
CHUNK_SIZE = 1024
DTYPE = 'float32'


class AudioInputDevice:
    def __init__(self, input_device_index=None, save_recording=False):
        """
        Initialize audio input device
        
        Args:
            input_device_index: Index của device để sử dụng (None = default)
            save_recording: Có lưu audio thành file WAV không
        """
        self.input_device_index = input_device_index
        self.save_recording = save_recording
        
        # Audio setup
        self.stream = None
        self.audio_queue = queue.Queue(maxsize=100)
        self.is_recording = False
        
        # Recording buffer (nếu cần lưu file)
        self.audio_buffer = []
        self.recording_filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        # Frame callback (cho wakeword detection)
        self.frame_callback = None
        
        self._setup_audio_device()
    
    def _setup_audio_device(self):
        """Setup audio device và kiểm tra"""
        try:
            # Set environment for WSL
            os.environ['PULSE_SERVER'] = 'unix:/mnt/wslg/PulseServer'
            
            # List available devices
            logger.info("🎤 Available input devices:")
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    logger.info(f"   Device {i}: {device['name']} (Channels: {device['max_input_channels']})")
            
            # Get device info
            if self.input_device_index is not None:
                device_info = sd.query_devices(self.input_device_index)
                logger.info(f"🎤 Using device {self.input_device_index}: {device_info['name']}")
            else:
                self.input_device_index = sd.default.device[0]
                device_info = sd.query_devices(self.input_device_index)
                logger.info(f"🎤 Using default input device: {device_info['name']} (Index: {self.input_device_index})")
                
        except Exception as e:
            logger.error(f"❌ Audio device setup error: {e}")
            raise
    
    def audio_callback(self, indata, frames, time, status):
        """Audio input callback"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        if self.is_recording:
            # Put in queue for streaming
            if not self.audio_queue.full():
                self.audio_queue.put(indata.copy())
            
            # Save to buffer nếu cần
            if self.save_recording:
                self.audio_buffer.append(indata.copy())
            
            # Call frame callback cho wakeword detection
            if self.frame_callback:
                # Convert float32 to int16 cho wakeword
                audio_int16 = (indata * 32767).astype(np.int16)
                self.frame_callback(audio_int16)
    
    def start_recording(self):
        """Bắt đầu recording"""
        try:
            # Create stream
            try:
                self.stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    device=self.input_device_index,
                    blocksize=CHUNK_SIZE,
                    callback=self.audio_callback
                )
            except Exception as stream_error:
                logger.warning(f"⚠️ Failed with specified device, trying default: {stream_error}")
                self.stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    blocksize=CHUNK_SIZE,
                    callback=self.audio_callback
                )
            
            self.is_recording = True
            self.stream.start()
            logger.info("🎤 Audio recording started")
            
            if self.save_recording:
                logger.info(f"🎵 Recording will be saved as: {self.recording_filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start audio recording: {e}")
            raise
    
    def stop_recording(self):
        """Dừng recording"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Lưu file WAV nếu cần
        if self.save_recording and len(self.audio_buffer) > 0:
            self._save_audio_to_wav()
        
        logger.info("🔇 Audio recording stopped")
    
    def _save_audio_to_wav(self):
        """Lưu audio buffer thành file WAV"""
        try:
            # Kết hợp tất cả audio chunks
            audio_data = np.concatenate(self.audio_buffer, axis=0)
            
            # Lưu thành file WAV
            sf.write(self.recording_filename, audio_data, SAMPLE_RATE)
            logger.info(f"💾 Audio saved to: {self.recording_filename}")
            
        except Exception as e:
            logger.error(f"❌ Error saving audio file: {e}")
    
    def get_audio_frame(self):
        """
        Lấy 1 frame audio cho wakeword detection
        Returns: numpy array (float32) hoặc None nếu không có data
        """
        try:
            if not self.audio_queue.empty():
                return self.audio_queue.get_nowait()
            return None
        except queue.Empty:
            return None
    
    def get_audio_frame_int16(self):
        """
        Lấy 1 frame audio format int16 cho wakeword detection
        Returns: numpy array (int16) hoặc None nếu không có data
        """
        frame = self.get_audio_frame()
        if frame is not None:
            # Convert float32 to int16
            return (frame * 32767).astype(np.int16)
        return None
    
    def set_frame_callback(self, callback_func):
        """
        Set callback function để xử lý real-time audio frames
        callback_func(audio_data_int16) sẽ được gọi cho mỗi frame
        """
        self.frame_callback = callback_func
    
    def get_audio_for_streaming(self):
        """
        Lấy audio data để streaming (cho Gemini Live)
        Returns: numpy array (float32) hoặc None
        """
        return self.get_audio_frame()
    
    def clear_queue(self):
        """Clear audio queue"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def __enter__(self):
        """Context manager support"""
        self.start_recording()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop_recording()


# Convenience function cho wakeword.py
def get_audio_frame():
    """
    Global function để lấy audio frame - tương thích với wakeword.py
    Cần có một instance global của AudioInputDevice
    """
    global _global_audio_device
    
    if '_global_audio_device' not in globals():
        logger.warning("⚠️ Audio device chưa được khởi tạo. Gọi init_global_audio_device() trước.")
        return None
    
    return _global_audio_device.get_audio_frame_int16()


def init_global_audio_device(device_index=None, save_recording=False):
    """
    Khởi tạo global audio device cho wakeword.py
    """
    global _global_audio_device
    
    _global_audio_device = AudioInputDevice(
        input_device_index=device_index,
        save_recording=save_recording
    )
    _global_audio_device.start_recording()
    logger.info("🎤 Global audio device initialized")
    
    return _global_audio_device


def cleanup_global_audio_device():
    """
    Cleanup global audio device
    """
    global _global_audio_device
    
    if '_global_audio_device' in globals():
        _global_audio_device.stop_recording()
        del _global_audio_device
        logger.info("🔇 Global audio device cleaned up")


# Example usage
if __name__ == "__main__":
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    print("🎤 Testing Audio Input Device...")
    
    # Test 1: Basic usage
    with AudioInputDevice(save_recording=True) as audio_device:
        print("Recording for 5 seconds...")
        time.sleep(5)
        
        # Test getting frames
        frame_count = 0
        for i in range(10):
            frame = audio_device.get_audio_frame_int16()
            if frame is not None:
                frame_count += 1
            time.sleep(0.1)
        
        print(f"Got {frame_count} audio frames")
    
    print("✅ Test completed!")
