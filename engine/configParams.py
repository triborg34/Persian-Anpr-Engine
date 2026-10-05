


class Parameters:

    def __init__(self):

        self.charclassnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'D', 'S', 'A', 'B', 'X', 'CER', 'Se', 'J', 'De', 'Z',
                  'Sin', 'SH', 'SAD', 'T', 'ZA', 'EIN', 'F', 'GH', 'L', 'M',
                  'N', 'H', 'H2', 'V', 'P', 'ZHE', 'K', 'G', 'Y']

        self.persian_letter = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        'D', 'S', 'الف', 'ب', 'ت', 'تشریفات', 'ث', 'ج', 'د', 'ز',
                        'س', 'ش', 'ص', 'ط', 'ظ', 'ع', 'ف', 'ق', 'ل', 'م', 'ن', 'ه', 'ه\u200c', 'و', 'پ', 'ژ (معلولین و جانبازان)', 'ک', 'گ', 'ی']

        self.persian_to_english_letter={'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', 'D': 'D', 'S': 'S', 'الف': 'A', 'ب': 'B', 'ت': 'X', 'تشریفات': 'CER', 'ث': 'Se', 'ج': 'J', 'د': 'De', 'ز': 'Z', 'س': 'Sin', 'ش': 'SH', 'ص': 'SAD', 'ط': 'T', 'ظ': 'ZA', 'ع': 'EIN', 'ف': 'F', 'ق': 'GH', 'ل': 'L', 'م': 'M', 'ن': 'N', 'ه': 'H', 'ه\u200c': 'H2', 'و': 'V', 'پ': 'P', 'ژ (معلولین و جانبازان)': 'ZHE', 'ک': 'K', 'گ': 'G', 'ی': 'Y'}


        self.english_to_persian_letter={'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', 'D': 'D', 'S': 'S', 'A': 'الف', 'B': 'ب', 'X': 'ت', 'CER': 'تشریفات', 'Se': 'ث', 'J': 'ج', 'De': 'د', 'Z': 'ز', 'Sin': 'س', 'SH': 'ش', 'SAD': 'ص', 'T': 'ط', 'ZA': 'ظ', 'EIN': 'ع', 'F': 'ف', 'GH': 'ق', 'L': 'ل', 'M': 'م', 'N': 'ن', 'H': 'ه', 'H2': 'ه\u200c', 'V': 'و', 'P': 'پ', 'ZHE': 'ژ (معلولین و جانبازان)', 'K': 'ک', 'G': 'گ', 'Y': 'ی'}

    def persian_to_english(self,plate: str) -> str:
            """Convert Persian plate characters to database-friendly English."""
            for persian, english in self.persian_to_english_letter.items():
                plate = plate.replace(persian, english)

            return plate


    def english_to_persian(self,plate: str) -> str:
            """Convert database-friendly English plate characters to Persian."""
            # Sort by length so codes like SH, TH, ZH are handled before
            # their individual characters.
            for english in sorted(self.english_to_persian_letter, key=len, reverse=True):
                plate = plate.replace(english, self.english_to_persian_letter[english])

            return plate


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
