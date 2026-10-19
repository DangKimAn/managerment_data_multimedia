import os
import re
import html
import time
import threading
from io import BytesIO
from PIL import Image
import numpy as np
import boto3
from botocore.exceptions import ClientError
from icrawler.storage import BaseStorage
from icrawler.builtin import BingImageCrawler
from icrawler.builtin.bing import BingParser
from icrawler.downloader import ImageDownloader

# Tự động đọc file .env nếu có trong thư mục
env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_file):
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# ==============================================================================
# 1. CẤU HÌNH AWS S3 (HOẶC MINIO / CLOUDFLARE R2 / WASABI)
# ==============================================================================
# Tên S3 Bucket (Bắt buộc)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "YOUR_BUCKET_NAME")

# Thư mục gốc trên S3 (prefix), ví dụ: "Project_Data" -> s3://bucket/Project_Data/1_Natural_Complex/...
S3_BASE_PREFIX = os.getenv("S3_BASE_PREFIX", "Project_Data")

# AWS Region (ví dụ: ap-southeast-1, us-east-1)
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

# AWS Credentials (Có thể điền trực tiếp ở đây hoặc gán vào biến môi trường hệ thống)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET_ACCESS_KEY")

# Endpoint URL tùy chọn (Chỉ cần dùng nếu dùng MinIO, Cloudflare R2, Wasabi,... Để None nếu dùng AWS chuẩn)
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", None)

# ==============================================================================
# 2. CẤU HÌNH CRAWL DATASET
# ==============================================================================
# Số lượng ảnh mục tiêu cho mỗi nhóm (100 ảnh/nhóm, tổng 4 nhóm = 400 ảnh)
TARGET_IMAGES_PER_GROUP = int(os.getenv("TARGET_IMAGES_PER_GROUP", 100))

# Số lượng ảnh tối đa tải cho mỗi từ khóa (ví dụ: 25 ảnh/từ khóa -> ~4 từ khóa đạt đủ 100 ảnh đa dạng)
MAX_IMAGES_PER_KEYWORD = int(os.getenv("MAX_IMAGES_PER_KEYWORD", 25))

# Giới hạn cạnh lớn nhất của ảnh (pixel) để tránh file BMP 4K/8K quá nặng khi tải lên S3.
# Độ phân giải 1280px đảm bảo chất lượng cực cao, sắc nét cho mọi thuật toán nén/xử lý ảnh.
# Đặt None nếu muốn giữ nguyên 100% kích thước gốc nguyên bản.
MAX_IMAGE_DIM = 1280

# Số luồng tải và phân tích song song
DOWNLOADER_THREADS = 12
PARSER_THREADS = 4


from s3_storage import S3Storage



# ==============================================================================
# 4. PARSER & DOWNLOADER CHUYỂN ĐỔI BMP TRỰC TIẾP TRONG BỘ NHỚ
# ==============================================================================
class CustomBingParser(BingParser):
    """Bắt toàn bộ link ảnh chất lượng cao từ thuộc tính 'murl' của Bing"""
    def parse(self, response):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content.decode("utf-8", "ignore"), "lxml")
        image_divs = soup.find_all("div", class_="imgpt")
        pattern = re.compile(r'"murl":"(https?://[^"]+)"')
        for div in image_divs:
            try:
                href_str = html.unescape(div.a["m"])
            except (KeyError, TypeError, AttributeError):
                continue
            match = pattern.search(href_str)
            if match:
                yield dict(file_url=match.group(1))


class BmpImageDownloader(ImageDownloader):
    """Tải ảnh về RAM, dùng Pillow convert sang chuẩn BMP và gọi storage.write để đẩy lên S3"""
    def get_filename(self, task, default_ext="bmp"):
        file_idx = self.fetched_num + self.file_idx_offset
        return f"{file_idx:06d}.bmp"

    def download(self, task, default_ext="bmp", timeout=8, max_retry=2, overwrite=False, **kwargs):
        file_url = task["file_url"]
        task["success"] = False
        task["filename"] = None
        retry = max_retry

        if not overwrite:
            with self.lock:
                self.fetched_num += 1
                filename = self.get_filename(task, default_ext)
                if self.storage.exists(filename):
                    self.logger.info("skip downloading file %s", filename)
                    return
                self.fetched_num -= 1

        while retry > 0 and not self.signal.get("reach_max_num"):
            try:
                response = self.session.get(file_url, timeout=timeout)
            except Exception as e:
                self.logger.debug("Lỗi tải link %s: %s", file_url, e)
            else:
                if self.reach_max_num():
                    self.signal.set(reach_max_num=True)
                    break
                elif response.status_code != 200:
                    break
                elif not self.keep_file(task, response, **kwargs):
                    break

                # Chuyển đổi sang BMP chuẩn RGB hoàn toàn trong bộ nhớ (RAM)
                try:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    if MAX_IMAGE_DIM is not None:
                        if max(img.size) > MAX_IMAGE_DIM:
                            img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
                    bmp_buffer = BytesIO()
                    img.save(bmp_buffer, format="BMP")
                    bmp_data = bmp_buffer.getvalue()
                except Exception as e:
                    self.logger.debug("Lỗi convert BMP %s: %s", file_url, e)
                    break

                with self.lock:
                    self.fetched_num += 1
                    filename = self.get_filename(task, default_ext)
                
                # Ghi trực tiếp lên S3 (S3Storage.write)
                self.storage.write(filename, bmp_data)
                self.logger.info("Đẩy lên S3 thành công #%s: %s", self.fetched_num, filename)
                task["success"] = True
                task["filename"] = filename
                break
            finally:
                retry -= 1


# ==============================================================================
# 5. HÀM CRAWL TỪNG NHÓM ĐẨY LÊN S3
# ==============================================================================
def crawl_images_for_group_s3(group_name, keywords, target_images=TARGET_IMAGES_PER_GROUP):
    s3_prefix = f"{S3_BASE_PREFIX}/{group_name}".strip("/")

    # Khởi tạo S3Storage cho nhóm hiện tại
    storage = S3Storage(
        bucket_name=S3_BUCKET_NAME,
        prefix=s3_prefix,
        endpoint_url=S3_ENDPOINT_URL,
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    storage.sync_existing()

    print(f"\n{'='*75}")
    print(f"========== BẮT ĐẦU CRAWL NHÓM: {group_name} ==========")
    print(f"Mục tiêu: {target_images} ảnh BMP | S3 URL: s3://{S3_BUCKET_NAME}/{s3_prefix}/")
    print(f"Hiện có sẵn trên S3: {storage.count()} ảnh")
    print(f"{'='*75}")

    for idx, base_keyword in enumerate(keywords, 1):
        current_count = storage.count()
        if current_count >= target_images:
            print(f"\n[HOÀN THÀNH] Nhóm '{group_name}' đã đạt đủ mục tiêu {current_count}/{target_images} ảnh trên S3!")
            break

        needed = target_images - current_count
        batch_num = min(MAX_IMAGES_PER_KEYWORD, needed)

        print(f"\n[{idx}/{len(keywords)}] Từ khóa: '{base_keyword}' | Tiến độ S3: {current_count}/{target_images} ảnh (Cần thêm: {needed})")

        crawler = BingImageCrawler(
            parser_cls=CustomBingParser,
            downloader_cls=BmpImageDownloader,
            storage=storage,
            feeder_threads=1,
            parser_threads=PARSER_THREADS,
            downloader_threads=DOWNLOADER_THREADS
        )

        filters = {'size': 'large'}

        try:
            crawler.crawl(
                keyword=base_keyword,
                filters=filters,
                max_num=batch_num,
                file_idx_offset='auto'
            )
        except Exception as e:
            print(f"Lỗi khi crawl từ khóa '{base_keyword}': {e}")

        new_count = storage.count()
        print(f"-> Sau từ khóa '{base_keyword}': Đang có {new_count}/{target_images} ảnh trên S3 ({(new_count/target_images)*100:.1f}%)")

        time.sleep(2)

    total_downloaded = storage.count()
    print(f"\n{'='*75}")
    print(f"========== KẾT THÚC NHÓM: {group_name} ({total_downloaded}/{target_images} ảnh trên S3) ==========")
    print(f"{'='*75}\n")


def generate_true_noise_to_s3(
    group_name: str = "3_Textures_Noisy",
    target_images: int = TARGET_IMAGES_PER_GROUP,
    width: int = 1280,
    height: int = 720,
    overwrite: bool = True
) -> None:
    """
    Sinh ảnh nhiễu ngẫu nhiên toán học thuần túy (True Mathematical Noise) bằng NumPy
    và đẩy trực tiếp lên S3 (không lưu file cục bộ).

    Cơ sở khoa học (Shannon Source Coding Theorem):
    - Mỗi byte pixel phân bố đều độc lập U(0, 255) => Entropy cực đại H(X) = 8.00 bpp.
    - Tương quan không gian = 0 (Spatial correlation r = 0).
    - Bộ lọc dự đoán PNG/WebP và từ điển LZ77/LZW hoàn toàn vô hiệu.
    - Kết quả: Tỷ lệ nén tiệm cận chặt chẽ 1.00x, chứng minh giới hạn nén lý thuyết.
    """
    s3_prefix = f"{S3_BASE_PREFIX}/{group_name}".strip("/")
    storage = S3Storage(
        bucket_name=S3_BUCKET_NAME,
        prefix=s3_prefix,
        endpoint_url=S3_ENDPOINT_URL,
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    storage.sync_existing()

    print(f"\n{'='*75}")
    print(f"========== BẮT ĐẦU SINH ẢNH NHIỄU TOÁN HỌC (TRUE NOISE): {group_name} ==========")
    print(f"Mục tiêu: {target_images} ảnh BMP ({width}x{height}) | S3: s3://{S3_BUCKET_NAME}/{s3_prefix}/")
    print(f"Chế độ ghi đè (Overwrite): {overwrite}")
    print(f"{'='*75}")

    created = 0
    for i in range(1, target_images + 1):
        filename = f"{i:06d}.bmp"
        if not overwrite and storage.exists(filename):
            print(f"[BỎ QUA] Đã tồn tại: {filename}")
            continue

        # 1. Sinh ma trận điểm ảnh ngẫu nhiên tuyệt đối U(0, 255) cho RGB
        noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        # 2. Chuyển thành ảnh BMP trong bộ nhớ RAM
        img = Image.fromarray(noise_array, "RGB")
        bmp_buffer = BytesIO()
        img.save(bmp_buffer, format="BMP")
        bmp_data = bmp_buffer.getvalue()

        # 3. Đẩy trực tiếp lên S3
        storage.write(filename, bmp_data)
        created += 1
        if created % 10 == 0 or created == target_images:
            print(f"-> Đã sinh & đẩy lên S3: [{created}/{target_images}] ảnh nhiễu chuẩn ({filename})")

    print(f"\n{'='*75}")
    print(f"========== HOÀN TẤT SINH {created} ẢNH NHIỄU TOÁN HỌC LÊN S3 ==========")
    print(f"{'='*75}\n")


# ==============================================================================
# 6. BỘ TỪ KHÓA PHONG PHÚ CHO 4 NHÓM
# ==============================================================================
DATASET_PLAN = {
    # NHÓM 1: Cảnh quan, thiên nhiên phức tạp (Natural & Complex Scenes)
    "1_Natural_Complex": [
        "landscape photography", "nature forest trees", "mountain peak sunset",
        "ocean waves sea", "underwater coral reef", "national park scenic",
        "autumn forest foliage", "desert sand dunes", "aurora borealis northern lights",
        "starry night milky way", "tropical rainforest jungle", "waterfall cascading rocks",
        "arctic glacier ice", "grand canyon aerial", "wildflower blooming meadow",
        "misty morning lake reflection", "rocky coastline cliffs", "thunderstorm clouds lightning",
        "snowy winter mountain pines", "volcanic landscape lava", "sunrise valley fog",
        "river canyon rapids", "bamboo forest pathway", "scenic rolling hills green",
        "african savanna wildlife sunset", "deep cavern rock formations", "sunbeams enchanted forest",
        "dramatic sea fjord norway", "rocky beach sunset long exposure", "alpine mountain lake mirror",
        "lush tropical paradise island", "autumn golden leaves path", "cloud forest mountains",
        "mossy green river canyon", "sandstone desert arches utah", "winter frost frozen forest",
        "mountain hiking ridge vista", "tropical lagoon clear water", "scenic countryside vineyard",
        "frozen winter waterfall ice", "deep ocean marine life", "dramatic ocean storm waves",
        "coastal sea caves cliffs", "coniferous pine forest fog", "everglades swamp wetland",
        "yellowstone geothermal pool", "dolomites mountain peaks", "lavender field sunrise provence",
        "rice terraces valley landscape", "cherry blossom spring river", "sunflower field blue sky",
        "coastal sunset lighthouse", "boreal taiga forest lake", "tundra wilderness landscape",
        "glacier bay icebergs", "sahara desert oasis dunes", "serengeti wildlife landscape",
        "appalachian mountains mist", "patagonia mountain peaks", "ha long bay karst landscape",
        "victoria falls spray rainbow", "matterhorn mountain peak", "banff national park lake",
        "sequoia national park trees", "redwood forest giants", "fjords scandinavia aerial",
        "antarctica iceberg sea", "atacama desert starry night", "monument valley rock towers",
        "plush moss forest cascade"
    ],

    # NHÓM 2: Đồ họa mảng phẳng, vector, biểu tượng (Flat Graphics & Vector Art)
    "2_Flat_Graphics": [
        "vector flat illustration", "minimalist app interface ui", "flat company logo design",
        "2d vector anime character", "infographic vector elements", "flat design icon set",
        "simple geometric background vector", "isometric 2d vector art", "clean minimal vector layout",
        "colorful flat abstract shapes", "flat character illustration people", "vector cartoon animal illustration",
        "flat landscape vector wallpaper", "minimal line art vector", "flat ui kit web mobile",
        "vector badge emblem vintage", "2d game sprite asset", "flat graphic design poster",
        "seamless vector pattern background", "flat mobile app dashboard", "pop art vector illustration",
        "flat retro 80s synthwave vector", "low poly vector illustration", "flat infographic banner presentation",
        "minimalist vector desktop wallpaper", "vector sticker badge set", "flat food drink illustration",
        "flat travel tourism poster", "flat modern workspace illustration", "abstract vector composition shapes",
        "flat business team illustration", "2d vector transportation car", "flat city skyline vector illustration",
        "minimalist continuous line drawing", "flat technology modern concept", "creative vector graphic illustration",
        "flat healthcare medical vector", "flat holiday celebration vector", "vector pictogram symbols set",
        "geometric bauhaus vector poster", "flat fitness gym illustration", "flat web banner template",
        "flat science space vector", "cute flat mascot character", "flat ecommerce shopping icons",
        "flat analytics dashboard interface", "modern minimalist vector logo", "flat avatar user profile icons",
        "flat botanical leaves illustration", "flat music audio vector", "flat education school illustration",
        "flat security cyber vector", "flat environment green eco vector", "flat weather climate icons",
        "flat finance investment vector", "flat delivery logistics vector", "flat real estate house vector",
        "flat pet veterinary illustration", "flat gaming esport vector", "flat social media vector banner",
        "vector outline icons set", "flat gradient modern illustration", "memphis pattern vector graphic",
        "flat digital marketing vector", "flat cryptocurrency blockchain vector", "flat coffee shop vector art",
        "flat gardening plants vector", "flat fashion clothing illustration", "flat book reading vector",
        "flat cooking kitchen illustration"
    ],

    # NHÓM 3: Vân bề mặt, chất liệu, nhiễu tần số cao (Textures & Noisy Patterns)
    "3_Textures_Noisy": [
        # --- Cực kỳ nhiễu (High Entropy / Sensor & Synthetic Noise) ---
        "tv static white noise background", "analog television static snow noise",
        "gaussian noise pattern visual", "salt and pepper noise texture",
        "high iso camera noise texture", "extreme high iso noise grain",
        "film grain texture overlay 35mm", "heavy 35mm film grain texture",
        "random white noise texture background", "chrominance color noise digital sensor",
        "vhs tape glitch noise grain", "speckle noise pattern texture",
        "crt monitor pixel noise texture", "uniform digital noise pattern",
        # --- Vân hạt vi mô và chất liệu thô ráp tự nhiên (High-frequency Textures) ---
        "coarse abrasive sandpaper texture", "natural sea salt crystals texture",
        "crumpled aluminum foil metallic texture", "seamless stone wall texture",
        "animal fur texture macro", "sand dune texture surface",
        "rough concrete texture wall", "fabric wool texture close up", "wood grain texture natural",
        "rusty metal texture oxidized", "brick wall texture seamless", "marble stone texture slab",
        "gravel ground texture pebbles", "asphalt pavement road texture", "weathered rustic wood plank texture",
        "rough tree bark macro texture", "crumpled paper texture background", "cracked dry mud soil ground",
        "denim jean textile fabric texture", "brushed metal steel surface texture", "woven linen burlap fabric texture",
        "cobblestone street pavement texture", "rough granite rock surface texture", "leather texture close up seamless",
        "corrugated cardboard paper texture", "mosaic ceramic tile wall texture", "rough burlap sack cloth texture",
        "heavy rusted iron metal texture", "porous sponge surface macro texture", "peeling distressed paint texture wall",
        "moss covered stone rock texture", "rough stucco plaster wall texture", "knitted wool sweater texture pattern",
        "cracked aged plaster wall texture", "dry parched earth soil texture", "woven rattan wicker basket texture",
        "natural slate stone tile texture", "cork board bulletin texture surface", "coarse abrasive sandpaper texture",
        "crumpled aluminum foil metallic texture", "chipped weathered paint wood texture", "cracked dry lake bed desert texture",
        "reptile snake skin scales texture", "weathered copper patina verdigris texture", "artist linen canvas texture background",
        "distressed concrete cement texture", "tree stump growth rings wood texture", "crushed velvet fabric texture soft",
        "coarse volcanic rock basalt texture", "ancient papyrus parchment paper texture", "woven carbon fiber texture pattern",
        "rough terrazzo stone floor texture", "sponge porous natural texture", "rough sandstone rock cliff texture",
        "cracked paint on canvas texture", "felt fabric craft material texture", "tweed wool fabric texture houndstooth",
        "oxidized bronze metal surface texture", "rough timber bark texture", "natural sea salt crystals texture",
        "crushed stone ballast track texture", "corduroy ribbed cotton fabric texture", "weathered ship wood timber texture",
        "rough limestone rock texture", "coarse wool carpet rug texture", "distressed leather jacket surface texture",
        "weathered corrugated sheet metal texture", "dense green moss forest floor texture", "rough handmade recycled paper texture",
        "grungy scratched metal texture", "perforated metal sheet mesh texture", "woven bamboo mat texture pattern",
        "charcoal carbon burned wood texture"
    ],

    # NHÓM 4: Tài liệu, văn bản, bản vẽ kỹ thuật (Documents, Text & Line Diagrams)
    "4_Documents_Text": [
        "scanned book page text", "vintage encyclopedia page print", "handwritten manuscript cursive letter",
        "architectural blueprint technical drawing", "printed newspaper column article", "vintage paper receipt invoice",
        "legal contract document paper", "mathematical equations handwritten paper", "antique handwritten letter calligraphy",
        "engineering schematic wiring diagram", "vintage sheet music piano score", "dictionary page vocabulary text",
        "old typewriter typed letter page", "vintage certificate diploma document", "scanned magazine article text",
        "ancient parchment scroll calligraphy", "comic book page inked panels text", "scientific research paper printed page",
        "financial report balance sheet tables", "calligraphy brush lettering practice paper", "vintage postal telegram message",
        "typography font specimen poster print", "vintage restaurant food menu typography", "electronic circuit diagram schematic",
        "old classic novel open book text", "handwritten journal diary notebook page", "printed crossword puzzle newspaper",
        "official notarized stamped document paper", "vintage hardware catalog page print", "ancient stone hieroglyphics inscription",
        "cursive handwriting exercise notebook", "printed library book page typography", "vintage antique world map labels",
        "patent application drawing blueprint", "accounting ledger book columns numbers", "scanned official certificate document",
        "vintage yellowed newspaper clipping", "orchestral conductor sheet music score", "programming computer code printed sheet",
        "vintage handwritten recipe card", "formal calligraphy invitation card letter", "mechanical CAD engineering draft",
        "printed legal brief court document", "medical lab test report paper", "vintage postage stamps collection sheet",
        "antique handwritten travel diary notebook", "schematic electronics circuit diagram", "vintage manga comic book text page",
        "typewritten poetry manuscript paper", "printed dictionary columns definitions", "handwritten algebra calculus notes",
        "old bible page gothic font text", "vintage travel passport stamped pages", "printed airline boarding pass ticket paper",
        "botanical illustration vintage text label", "vintage stock share certificate bond", "scanned user manual instruction diagram",
        "vintage postal postcard handwriting", "printed periodic table chemistry poster", "patent technical diagram typography",
        "vintage newspaper front page headline", "handwritten laboratory notebook science", "old library catalog card printed text",
        "printed musical notation score paper", "vintage medical anatomy book text", "handwritten recipe book cursive text",
        "printed exam test paper questionnaire", "vintage bank check draft paper", "printed tax form document paper",
        "handwritten personal letter envelope text"
    ]
}


# ==============================================================================
# 7. MAIN FUNCTION
# ==============================================================================
if __name__ == '__main__':
    # Kiểm tra cấu hình S3 trước khi chạy
    if S3_BUCKET_NAME == "YOUR_BUCKET_NAME":
        print("\n" + "="*75)
        print("[LỖI CHƯA CẤU HÌNH S3 BUCKET]")
        print("Vui lòng mở file crawl_dataset.py và điền thông tin S3 tại dòng 18 - 30:")
        print("  - S3_BUCKET_NAME = 'tên_bucket_của_bạn'")
        print("  - AWS_ACCESS_KEY_ID = 'access_key_của_bạn'")
        print("  - AWS_SECRET_ACCESS_KEY = 'secret_key_của_bạn'")
        print("  - AWS_REGION = 'khu_vực_bucket (ví dụ ap-southeast-1)'")
        print("\nHoặc export biến môi trường trong terminal:")
        print("  export S3_BUCKET_NAME='my-bucket'")
        print("  export AWS_ACCESS_KEY_ID='AKIA...'")
        print("  export AWS_SECRET_ACCESS_KEY='secret...'")
        print("  export AWS_REGION='ap-southeast-1'")
        print("="*75 + "\n")
        exit(1)

    print(f"\n{'='*75}")
    print(f"CHƯƠNG TRÌNH CRAWL 4 NHÓM x {TARGET_IMAGES_PER_GROUP:,} ẢNH BMP ĐẨY TRỰC TIẾP LÊN S3")
    print(f"Bucket đích: s3://{S3_BUCKET_NAME}/{S3_BASE_PREFIX}/")
    print(f"Trạng thái: KHÔNG LƯU BẤT KỲ FILE NÀO TRÊN Ổ CỨNG MÁY CHỦ")
    print(f"{'='*75}")

    import argparse
    parser = argparse.ArgumentParser(description="Crawl ảnh hoặc sinh ảnh nhiễu chuẩn đẩy lên S3.")
    parser.add_argument("--noise-only", action="store_true", help="Chỉ sinh 100 ảnh nhiễu toán học chuẩn cho nhóm 3_Textures_Noisy.")
    parser.add_argument("--no-overwrite", action="store_true", help="Không ghi đè nếu ảnh đã tồn tại trên S3.")
    cli_args, _ = parser.parse_known_args()

    if cli_args.noise_only:
        generate_true_noise_to_s3(
            group_name="3_Textures_Noisy",
            target_images=TARGET_IMAGES_PER_GROUP,
            overwrite=not cli_args.no_overwrite
        )
        print("\n[CHÚC MỪNG] ĐÃ HOÀN TẤT SINH ẢNH NHIỄU TOÁN HỌC CHO NHÓM 3 LÊN S3!")
        sys.exit(0)

    # Chạy lần lượt từng nhóm
    for group_name, kw_list in DATASET_PLAN.items():
        if group_name == "3_Textures_Noisy":
            generate_true_noise_to_s3(
                group_name=group_name,
                target_images=TARGET_IMAGES_PER_GROUP,
                overwrite=not cli_args.no_overwrite
            )
        else:
            crawl_images_for_group_s3(
                group_name=group_name,
                keywords=kw_list,
                target_images=TARGET_IMAGES_PER_GROUP
            )
        print("Nghỉ 5 giây trước khi chuyển sang nhóm tiếp theo...")
        time.sleep(5)

    print("\n[CHÚC MỪNG] ĐÃ HOÀN TẤT CRAWL VÀ UPLOAD 4 NHÓM LÊN S3!")