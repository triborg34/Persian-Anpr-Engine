
import json
import os
import plates  # از plates.py: canonical_letter, to_en_digits, parse استفاده می‌کنیم


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLATE_DATA_PATH = os.path.join(BASE_DIR, 'plate_data.json')
MAX_LOCATIONS = 3  # حداکثر تعداد شهر/استان برگشتی
class Parameters:

    def __init__(self):
        self.PLACES_BY_CODE_LETTER, self.PLACES_BY_CODE = self._build_indexes()

        self.charclassnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                               'D', 'S', 'A', 'B', 'X', 'CER', 'Se', 'J', 'De', 'Z',
                               'Sin', 'SH', 'SAD', 'T', 'ZA', 'EIN', 'F', 'GH', 'L', 'M',
                               'N', 'H', 'HX', 'V', 'P', 'ZHE', 'K', 'G', 'Y']

        self.persian_letter = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                               'D', 'S', 'الف', 'ب', 'ت', 'تشریفات', 'ث', 'ج', 'د', 'ز',
                               'س', 'ش', 'ص', 'ط', 'ظ', 'ع', 'ف', 'ق', 'ل', 'م', 'ن', 'ه', 'ه\u200c', 'و', 'پ', 'ژ (معلولین و جانبازان)', 'ک', 'گ', 'ی']

        self.persian_to_english_letter = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', 'D': 'D', 'S': 'S', 'الف': 'A', 'ب': 'B', 'ت': 'X', 'تشریفات': 'CER', 'ث': 'Se', 'ج': 'J', 'د': 'De',
                                          'ز': 'Z', 'س': 'Sin', 'ش': 'SH', 'ص': 'SAD', 'ط': 'T', 'ظ': 'ZA', 'ع': 'EIN', 'ف': 'F', 'ق': 'GH', 'ل': 'L', 'م': 'M', 'ن': 'N', 'ه': 'H', 'ه\u200c': 'HX', 'و': 'V', 'پ': 'P', 'ژ (معلولین و جانبازان)': 'ZHE', 'ک': 'K', 'گ': 'G', 'ی': 'Y'}

        self.english_to_persian_letter = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', 'D': 'D', 'S': 'S', 'A': 'الف', 'B': 'ب', 'X': 'ت', 'CER': 'تشریفات', 'Se': 'ث', 'J': 'ج', 'De': 'د',
                                          'Z': 'ز', 'Sin': 'س', 'SH': 'ش', 'SAD': 'ص', 'T': 'ط', 'ZA': 'ظ', 'EIN': 'ع', 'F': 'ف', 'GH': 'ق', 'L': 'ل', 'M': 'م', 'N': 'ن', 'H': 'ه', 'HX': 'ه\u200c', 'V': 'و', 'P': 'پ', 'ZHE': 'ژ (معلولین و جانبازان)', 'K': 'ک', 'G': 'گ', 'Y': 'ی'}

    def persian_to_english(self, plate: str) -> str:
        """Convert Persian plate characters to database-friendly English."""
        for persian, english in self.persian_to_english_letter.items():
            plate = plate.replace(persian, english)

        return plate

    def english_to_persian(self, plate: str) -> str:
        """Convert database-friendly English plate characters to Persian."""
        # Sort by length so codes like SH, TH, ZH are handled before
        # their individual characters.
        for english in sorted(self.english_to_persian_letter, key=len, reverse=True):
            plate = plate.replace(
                english, self.english_to_persian_letter[english])

        return plate

# ── ایندکس‌های plate_data.json (یک بار ساخته می‌شوند) ────────────────────────
    def _build_indexes(self):
        """
        از فایل plate_data.json دو ایندکس می‌سازد:
        * by_code_letter: کلید (کد ۲ رقمی, حرف) → لیست (استان, شهر)
        * by_code      : کلید کد ۲ رقمی → لیست (استان, شهر)
        """
        with open(PLATE_DATA_PATH, encoding='utf-8') as f:
            data = json.load(f)

        by_code_letter = {}
        by_code = {}
        for province, cities in data.get('carplates', {}).items():
            for city, codes in cities.items():
                for code, letters in codes.items():
                    place = (province, city)
                    by_code.setdefault(code, []).append(place)
                    if letters:
                        for raw in letters:
                            key = (code, plates.canonical_letter(raw))
                            by_code_letter.setdefault(key, []).append(place)
                    else:
                        # کد بدون محدودیت حرف: برای همه حروف معتبر است
                        by_code_letter.setdefault((code, None), []).append(place)
        return by_code_letter, by_code

    def lookup_city(self,letter, suffix):
        """
        تعیین شهر و استان بر اساس حرف و کد ۲ رقمی پلاک.

        Args:
            letter: حرف پلاک (مثلاً 'ن')
            suffix: کد ۲ رقمی انتهای پلاک (مثلاً '66')

        Returns:
            لیستی از {'province': استان, 'city': شهر} — حداکثر MAX_LOCATIONS مورد.
        """
        letter = plates.canonical_letter(letter)
        suffix = plates.to_en_digits(suffix)

        matches = []
        seen = set()
        # اول: جستجو با (کد, حرف) دقیق، بعد (کد, بدون حرف)
        for key in ((suffix, letter), (suffix, None)):
            for place in self.PLACES_BY_CODE_LETTER.get(key, ()):
                if place not in seen:
                    seen.add(place)
                    matches.append(place)

        # اگر چیزی پیدا نشد، فقط بر اساس کد جستجو کن (مثل نسخه اصلی)
        if not matches:
            for place in self.PLACES_BY_CODE.get(suffix, ()):
                if place not in seen:
                    seen.add(place)
                    matches.append(place)

        return [{'province': p, 'city': c} for p, c in matches[:MAX_LOCATIONS]]


    def plate_city(self,raw_plate_text):
        """
        تعیین شهر/استان از روی متن کامل پلاک.

        Args:
            raw_plate_text: هر نوشتاری از پلاک (مثلاً "24ن144-66" یا "۲۴ن۱۴۴۶۶")

        Returns:
            {'province': استان, 'city': شهر} اول و لیست همه موارد، یا None اگر پلاک معتبر نباشد.
        """
        parsed = plates.parse(raw_plate_text)
        if not parsed:
            return None

        locations = self.lookup_city(parsed['letter'], parsed['suffix'])
        return {
            'canonical': parsed['canonical'],
            'letter': parsed['letter'],
            'suffix': parsed['suffix'],
            'locations': locations,
            'province': locations[0]['province'] if locations else '',
            'city': locations[0]['city'] if locations else '',
        }

# def detectChars(img):


#     chars = []
#     confidences = []

#     char_res=char_model.predict(img,verbose=False)[0]
#     boxes=char_res.boxes


#     if boxes is not None and len(boxes) > 0:

#         order = boxes.xyxy[:, 0].argsort()

#         for idx in order:
#             cls_id = int(boxes.cls[idx])
#             conf = float(boxes.conf[idx])

#             chars.append(char_model.names[cls_id])
#             confidences.append(conf)

#     char_result = ''.join(chars)

#     char_conf_avg = round(statistics.mean(confidences) * 100) if confidences else 0
#     return char_result,char_conf_avg
