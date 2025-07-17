import sounddevice as sd
import numpy as np
import soundfile as sf
import sys
import os

def select_input_device():
    """Hiển thị danh sách các thiết bị đầu vào và cho phép người dùng chọn một."""
    print("Đang truy vấn các thiết bị âm thanh...")
    try:
        devices = sd.query_devices()
        input_devices = [device for device in devices if device['max_input_channels'] > 0]
        if not input_devices:
            print("Không tìm thấy thiết bị đầu vào nào.")
            return None
        print("Các thiết bị đầu vào có sẵn:")
        for i, device in enumerate(input_devices):
            default_marker = " (mặc định)" if i == sd.default.device[0] else ""
            print(f"{i}: {device['name']}{default_marker} - {device['max_input_channels']} kênh")
        while True:
            try:
                device_index = int(input(f"Chọn thiết bị đầu vào (0-{len(input_devices) - 1}): "))
                if 0 <= device_index < len(input_devices):
                    selected_device = input_devices[device_index]
                    print(f"Đã chọn: {selected_device['name']}")
                    return selected_device['name']
                else:
                    print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
            except ValueError:
                print("Vui lòng nhập một số nguyên.")
    except Exception as e:
        print(f"Lỗi khi truy vấn thiết bị: {e}")
        return None

def record_audio_stream(filename="output.wav", device=None, samplerate=16000, channels=1, blocksize=512):
    """Ghi âm từ micro và lưu vào file WAV, với khả năng chẩn đoán lỗi buffer."""
    recorded_data = []

    def audio_callback(indata, frames, time, status):
        """Hàm callback được gọi cho mỗi audio buffer.
        
        Kiểm tra cờ status để phát hiện các vấn đề như tràn bộ đệm.
        """
        if status:
            # In ra cảnh báo một cách rõ ràng hơn
            if status.input_overflow:
                print("CẢNH BÁO: Đã xảy ra tràn bộ đệm đầu vào! Dữ liệu có thể bị mất.", file=sys.stderr)
            if status.input_underflow:
                print("CẢNH BÁO: Đã xảy ra thiếu dữ liệu bộ đệm đầu vào!", file=sys.stderr)
            # Bạn có thể thêm các kiểm tra khác nếu cần
            # print(status, file=sys.stderr)
        
        # Debug: in ra để kiểm tra xem callback có được gọi không
        print(f"Đã ghi {len(recorded_data)} khối dữ liệu...", end='\r')
            
        recorded_data.append(indata.copy())

    print(f"Đang khởi tạo luồng âm thanh với thiết bị: {device}")
    print(f"Tham số: samplerate={samplerate}, channels={channels}, blocksize={blocksize}")
    
    try:
        with sd.InputStream(samplerate=samplerate, device=device,
                            channels=channels, callback=audio_callback,
                            blocksize=blocksize, dtype='float32'):
            print('#' * 80)
            print(f"Bắt đầu ghi âm. Sẽ lưu vào file '{filename}'")
            print('Nhấn Enter để dừng ghi âm')
            print('#' * 80)
            input()

        print(f'\nĐã dừng ghi âm. Số khối dữ liệu đã ghi: {len(recorded_data)}')
        print('Đang lưu file...')
        
        if recorded_data:
            # Ghép tất cả dữ liệu âm thanh lại với nhau
            audio_data = np.concatenate(recorded_data, axis=0)
            print(f"Tổng số mẫu âm thanh: {len(audio_data)}")
            
            # Đảm bảo filename không có khoảng trắng ở đầu/cuối và sử dụng đường dẫn tuyệt đối
            filename = filename.strip()
            if not os.path.isabs(filename):
                filename = os.path.join(os.getcwd(), filename)
            
            # Lưu trực tiếp vào file đích
            sf.write(filename, audio_data, samplerate)
            print(f"Bản ghi đã được lưu thành công vào: {filename}")
            print(f"Độ dài: {len(audio_data)/samplerate:.2f} giây")
        else:
            print("Không có dữ liệu âm thanh nào được ghi lại!")
            print("Có thể thiết bị âm thanh không hoạt động hoặc không có quyền truy cập micro.")

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("\nĐã dừng ghi âm bởi người dùng.")
        if recorded_data:
            # Vẫn cố gắng lưu dữ liệu đã ghi được
            try:
                audio_data = np.concatenate(recorded_data, axis=0)
                filename = filename.strip()
                if not os.path.isabs(filename):
                    filename = os.path.join(os.getcwd(), filename)
                sf.write(filename, audio_data, samplerate)
                print(f"Dữ liệu đã được lưu vào: {filename}")
            except Exception as save_error:
                print(f"Không thể lưu file: {save_error}")

if __name__ == "__main__":
    try:
        selected_device = select_input_device()
        if selected_device:
            file_name = input("Nhập tên file để lưu (ví dụ: my_record.wav, mặc định: output.wav): ").strip()
            if not file_name:
                file_name = "output.wav"
            if not file_name.lower().endswith('.wav'):
                file_name += '.wav'
            
            # Bạn có thể thay đổi tần số lấy mẫu nếu muốn, 44100Hz là tiêu chuẩn CD
            record_audio_stream(filename=file_name, device=selected_device, samplerate=16000, blocksize=512)
    except KeyboardInterrupt:
        print("\nĐã hủy bởi người dùng.")
        sys.exit(0)