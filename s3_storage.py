import os
import threading
from io import BytesIO
from PIL import Image
import boto3
from botocore.exceptions import ClientError
from icrawler.storage import BaseStorage


class S3Storage(BaseStorage):
    """
    Storage backend đẩy/kéo trực tiếp dữ liệu từ RAM lên/về S3 bucket.
    Không ghi bất kỳ byte nào xuống ổ cứng máy chủ.
    """
    def __init__(self, bucket_name, prefix="", endpoint_url=None, region_name="us-east-1",
                 aws_access_key_id=None, aws_secret_access_key=None, s3_client=None):
        self.bucket_name = bucket_name
        self.prefix = prefix.strip("/")
        
        if s3_client is not None:
            self.s3 = s3_client
        else:
            client_kwargs = {}
            if endpoint_url:
                client_kwargs["endpoint_url"] = endpoint_url
            if region_name:
                client_kwargs["region_name"] = region_name
            if aws_access_key_id and aws_secret_access_key and "YOUR_" not in aws_access_key_id:
                client_kwargs["aws_access_key_id"] = aws_access_key_id
                client_kwargs["aws_secret_access_key"] = aws_secret_access_key
            self.s3 = boto3.client("s3", **client_kwargs)

        self._lock = threading.Lock()
        self._current_max_idx = 0
        self._existing_keys = set()
        self._initialized = False

    def sync_existing(self):
        """Quét danh sách file hiện có trên S3 để hỗ trợ tự động đếm & tiếp tục (resume) mà không tải trùng"""
        with self._lock:
            if self._initialized:
                return
            self._existing_keys.clear()
            max_idx = 0
            paginator = self.s3.get_paginator("list_objects_v2")
            prefix = f"{self.prefix}/" if self.prefix else ""
            try:
                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        self._existing_keys.add(key)
                        filename = os.path.basename(key)
                        try:
                            idx = int(os.path.splitext(filename)[0])
                            if idx > max_idx:
                                max_idx = idx
                        except ValueError:
                            pass
            except ClientError as e:
                print(f"[CẢNH BÁO S3] Lỗi quét danh sách file trên S3: {e}")

            self._current_max_idx = max_idx
            self._initialized = True

    def write(self, id, data):
        """Ghi dữ liệu (bytes) lên S3"""
        key = f"{self.prefix}/{id}".lstrip("/") if self.prefix else id
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
            ContentType="image/bmp"
        )
        with self._lock:
            self._existing_keys.add(key)
            try:
                idx = int(os.path.splitext(id)[0])
                if idx > self._current_max_idx:
                    self._current_max_idx = idx
            except ValueError:
                pass

    def read(self, id):
        """
        Tải raw bytes của 1 file từ S3.
        """
        key = f"{self.prefix}/{id}".lstrip("/") if self.prefix and not id.startswith(self.prefix) else id
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except ClientError as e:
            print(f"[LỖI S3] Lỗi khi đọc file {key}: {e}")
            return None
        except Exception as e:
            print(f"Lỗi khi tải file {key} từ S3: {e}")
            return None

    def load_data(self, file_key):
        """
        Tải file từ S3 và trả về 3 định dạng để bạn tiện sử dụng cho các thuật toán khác nhau:
        1. img_obj: Đối tượng ảnh Pillow (Dùng cho OpenCV, trích xuất màu)
        2. raw_bytes: Mảng byte gốc của file BMP (Dùng cho nén thực tế)
        3. bit_string: Chuỗi ký tự '010101...' (Dùng để test thuật toán nén cơ bản)
        """
        try:
            raw_bytes = self.read(file_key)
            if raw_bytes is None:
                return None, None, None

            # 2. Chuyển đổi thành chuỗi ký tự '0' và '1' (Nếu thuật toán của bạn yêu cầu string)
            # bit_string = ''.join(format(byte, '08b') for byte in raw_bytes)
            bit_string = None

            # 3. Tạo đối tượng Image cho các thuật toán khác
            img_obj = Image.open(BytesIO(raw_bytes))

            return img_obj, raw_bytes, bit_string
        except Exception as e:
            print(f"Lỗi khi xử lý dữ liệu ảnh {file_key}: {e}")
            return None, None, None

    def exists(self, id):
        self.sync_existing()
        key = f"{self.prefix}/{id}".lstrip("/") if self.prefix else id
        with self._lock:
            return key in self._existing_keys

    def max_file_idx(self):
        self.sync_existing()
        with self._lock:
            return self._current_max_idx

    def count(self):
        self.sync_existing()
        with self._lock:
            return len([k for k in self._existing_keys if k.lower().endswith('.bmp')])
