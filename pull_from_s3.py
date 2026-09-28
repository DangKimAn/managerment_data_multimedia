import os
import boto3
import concurrent.futures
from io import BytesIO
from PIL import Image

# ==========================================
# 1. KHỞI TẠO S3 CLIENT
# ==========================================
s3_client = boto3.client(
    's3',
    region_name=os.getenv("AWS_REGION", "ap-southeast-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "ten-bucket-cua-ban")

# ==========================================
# 2. HÀM TẢI VÀ CHUYỂN ĐỔI ẢNH THÀNH CHUỖI BITS
# ==========================================
def load_data_from_s3(bucket, file_key):
    """
    Tải file từ S3 và trả về 3 định dạng để bạn tiện sử dụng cho các thuật toán khác nhau:
    1. img_obj: Đối tượng ảnh Pillow (Dùng cho OpenCV, trích xuất màu)
    2. raw_bytes: Mảng byte gốc của file BMP (Dùng cho nén thực tế)
    3. bit_string: Chuỗi ký tự '010101...' (Dùng để test thuật toán nén cơ bản)
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=file_key)
        # 1. Lấy dữ liệu dạng mảng Byte gốc (Thường dùng nhất để nén)
        raw_bytes = response['Body'].read()
        
        # 2. Chuyển đổi thành chuỗi ký tự '0' và '1' (Nếu thuật toán của bạn yêu cầu string)
        # Lưu ý: Hàm format '08b' sẽ chuyển 1 byte (vd: 255) thành '11111111'
        # bit_string = ''.join(format(byte, '08b') for byte in raw_bytes)
        
        # Tạm tắt bit_string để tránh ngốn RAM nếu bạn chưa xài tới. 
        # Nếu thuật toán của bạn nhận vào chuỗi '0101...', hãy bỏ comment dòng trên.
        bit_string = None 
        
        # 3. Tạo đối tượng Image cho các thuật toán khác
        img_obj = Image.open(BytesIO(raw_bytes))
        
        return img_obj, raw_bytes, bit_string

    except Exception as e:
        print(f"Lỗi khi tải ảnh {file_key}: {e}")
        return None, None, None

# ==========================================
# 3. CÁC THUẬT TOÁN XỬ LÝ
# ==========================================
def thuat_toan_1_nen_anh(raw_bytes, bit_string):
    """
    Ví dụ: Thuật toán nén của bạn nhận vào dữ liệu thô.
    """
    print(f" - [Nén] Đang xử lý file dung lượng: {len(raw_bytes)} bytes")
    
    # CÁCH 1: Nếu thuật toán của bạn chạy trên mảng Byte (Thực tế & Tối ưu)
    # nén_data = my_huffman_compress(raw_bytes)
    
    # CÁCH 2: Nếu thuật toán của bạn chạy trên chuỗi '0' và '1' (Thường dùng trong học thuật)
    # if bit_string:
    #     print(f" - [Nén] Chuỗi bit bắt đầu bằng: {bit_string[:64]}...")
    #     nén_data = my_binary_string_compress(bit_string)
    pass

def thuat_toan_2_khac(img_obj):
    print(f" - [Xử lý màu] Kích thước ảnh: {img_obj.size}")
    pass

# ==========================================
# 4. HÀM PIPELINE ĐA LUỒNG
# ==========================================
def process_single_image(file_key):
    print(f"\nBắt đầu tải: {file_key}")
    
    # Kéo từ S3 về và lấy ra cả Object Ảnh lẫn Chuỗi Bits
    img_obj, raw_bytes, bit_string = load_data_from_s3(BUCKET_NAME, file_key)
    
    if img_obj is not None and raw_bytes is not None:
        try:
            # Truyền mảng byte / bit string vào thuật toán nén
            thuat_toan_1_nen_anh(raw_bytes, bit_string)
            
            # Truyền đối tượng ảnh vào thuật toán khác
            thuat_toan_2_khac(img_obj)
        finally:
            # Giải phóng bộ nhớ RAM
            img_obj.close()