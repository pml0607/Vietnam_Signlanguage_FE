import pandas as pd
import os

# Đọc CSV gốc
input_csv = "val.corpus.csv"
df = pd.read_csv(input_csv)

def convert_to_video_path(npy_path):
    # Chuẩn hóa path
    npy_path = npy_path.replace('/', os.sep).replace('\\', os.sep)

    # Phân tích các phần trong đường dẫn
    parts = npy_path.split(os.sep)
    
    # Tìm chỉ số của 'npy' và tên thư mục động tác
    try:
        idx = parts.index('npy')
        action_folder = parts[idx + 1]  # ví dụ: A52P7
    except ValueError:
        raise ValueError(f"❌ Không tìm thấy thư mục 'npy' trong đường dẫn: {npy_path}")
    
    # Tạo đường dẫn mới
    root_parts = parts[:idx]  # everything before 'npy'
    filename = parts[-1].replace('_landmarked', '').replace('.npy', '.avi')
    
    new_path = os.path.join(
        *root_parts, 'mediapipe', action_folder, 'rgb', filename
    )
    
    return new_path

# Tạo cột video_path mới
df['video_path'] = df['file_path'].apply(convert_to_video_path)

# Đưa video_path lên đầu
df = df[['video_path'] + [col for col in df.columns if col != 'video_path']]

# Lưu lại file mới
output_csv = "cnn_val.corpus.csv"
df.to_csv(output_csv, index=False)

print("✅ Đã tạo file CSV với đường dẫn video chính xác.")
