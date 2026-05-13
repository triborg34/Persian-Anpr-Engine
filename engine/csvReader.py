import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pandas as pd
import datetime
import requests

# ----------------------------------------------------------------------
# Dictionaries for Persian/English plate conversion (exactly as provided)
alphabetP1 = {
    "A": "آ",
    "B": "ب",
    "D": "د",
    "Gh": "ق",
    "H": "ه",
    "J": "ج",
    "L": "ل",
    "M": "م",
    "N": "ن",
    "P": "پ",
    "PuV": "ع",
    "PwD": "ژ",
    "Sad": "ص",
    "Sin": "س",
    "T": "ط",
    "Taxi": "ت",
    "V": "و",
    "Y": "ی",
}

alphabetP2 = {
    "۰": "0",
    "۱": "1",
    "۲": "2",
    "۳": "3",
    "۴": "4",
    "۵": "5",
    "۶": "6",
    "۷": "7",
    "۸": "8",
    "۹": "9",
    "آ": "A",
    "ب": "B",
    "د": "D",
    "ق": "Gh",
    "ه": "H",
    "ج": "J",
    "ل": "L",
    "م": "M",
    "ن": "N",
    "پ": "P",
    "ع": "PuV",
    "ژ": "PwD",
    "ص": "Sad",
    "س": "Sin",
    "ط": "T",
    "ت": "Taxi",
    "و": "V",
    "ی": "Y",
}

URL = 'http://127.0.0.1:8090/api/collections/registredDb/records'


def parse_plate(plate_english: str):
    """
    Parse an English‑converted Iranian plate.
    Format: 2 digits + letter_code + 5 digits (3 middle + 2 last)
    Returns a dict with keys: firstTwoDigit, englishAlphabet, threeDigit, lastTwoDigit
    Returns None if pattern fails.
    """
    import re
    # Pattern: 2 digits, then letters (A-Za-z), then 5 digits
    match = re.match(r'^(\d{2})([A-Za-z]+)(\d{5})$', plate_english)
    if not match:
        return None
    first_two = match.group(1)
    letter_code = match.group(2)
    rest_digits = match.group(3)   # 5 digits
    three_digit = rest_digits[:3]
    last_two = rest_digits[3:]
    return {
        "firstTwoDigit": first_two,
        "englishAlphabet": letter_code,
        "threeDigit": three_digit,
        "lastTwoDigit": last_two
    }
# ----------------------------------------------------------------------
class ExcelUploaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Excel File Upload & Process")
        self.root.geometry("700x550")
        self.root.resizable(True, True)

        # Path selection
        path_frame = ttk.Frame(root)
        path_frame.pack(pady=10, padx=10, fill=tk.X)

        ttk.Label(path_frame, text="Excel File:").pack(side=tk.LEFT)
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(path_frame, textvariable=self.file_path_var, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=5)
        self.browse_btn = ttk.Button(path_frame, text="Browse", command=self.browse_file)
        self.browse_btn.pack(side=tk.LEFT)

        # Upload / Process button
        self.process_btn = ttk.Button(root, text="Upload & Process", command=self.start_processing_thread)
        self.process_btn.pack(pady=10)

        # Log area
        log_frame = ttk.LabelFrame(root, text="Processing Log")
        log_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=20)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Excel file",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)

    def log_message(self, msg, level="INFO"):
        """Insert a message into the log text widget and scroll to bottom."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {msg}\n"
        self.log_text.insert(tk.END, formatted)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_processing_thread(self):
        """Run the processing in a separate thread to keep GUI responsive."""
        file_path = self.file_path_var.get().strip()
        if not file_path:
            messagebox.showerror("Error", "Please select an Excel file first.")
            return

        # Disable buttons during processing
        self.process_btn.config(state=tk.DISABLED)
        self.browse_btn.config(state=tk.DISABLED)
        self.log_text.delete(1.0, tk.END)
        self.status_var.set("Processing...")

        thread = threading.Thread(target=self.process_excel_file, args=(file_path,), daemon=True)
        thread.start()

    def process_excel_file(self, file_path):
        """Main processing logic (runs in background thread)."""
        try:
            self.log_message(f"Reading Excel file: {file_path}")
            df = pd.read_excel(file_path)

            # Validate required columns
            required_cols = ['name', 'platenumber', 'carName', 'role']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                self.log_message(f"Missing columns: {missing}", "ERROR")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Missing columns: {missing}"))
                return

            # Optional 'arvand' column
            if 'arvand' not in df.columns:
                df['arvand'] = False

            count = df.shape[0]
            self.log_message(f"Loaded {count} rows. Starting upload to {URL}")

            # Prepare values that are the same for all records (date/time once)
            current_date = datetime.datetime.now().date()
            current_time = datetime.datetime.now().time()

            # Extract data as lists
            names = df['name'].tolist()
            plateFarsi = df['platenumber'].tolist()
            carname = df['carName'].tolist()
            role = df['role'].tolist()
            arvand = df['arvand'].tolist()

            # Convert Persian plates to English representation
            plateEnglish = []
            for plate in plateFarsi:
                translated = ''.join(alphabetP2.get(ch, ch) for ch in str(plate))
                plateEnglish.append(translated)

            success_count = 0
            fail_count = 0

            for i in range(count):
                self.log_message(f"Processing row {i+1}/{count}: {names[i]}")

                # Build payload according to original logic
                if arvand[i]:
                    isarvand = 'arvand'
                    firstTwoDigit = threeDigit = lastTwoDigit = englishAlphabet = persinalAlphabet = ""
                else:
                    isarvand = 'notarvand'
                    parsed = parse_plate(plateEnglish[i])
                    if parsed is None:
                        self.log_message(f"Skipping row {i+1}: invalid plate format '{plateEnglish[i]}'", "ERROR")
                        fail_count += 1
                        continue
                    firstTwoDigit = parsed["firstTwoDigit"]
                    englishAlphabet = parsed["englishAlphabet"]
                    threeDigit = parsed["threeDigit"]
                    lastTwoDigit = parsed["lastTwoDigit"]
                    persinalAlphabet = alphabetP1.get(englishAlphabet)  # may be None if unknown code

                body = {
                    "name": names[i],
                    "carName": carname[i],
                    "eDate": current_date.isoformat(),
                    "eTime": current_time.strftime("%H:%M"),
                    "role": role[i],
                    "rtpath": "/rt1",
                    "plateNumber": plateEnglish[i],
                    "isarvand": isarvand,
                    "firstTwoDigit": firstTwoDigit,
                    "threeDigit": threeDigit,
                    "lastTwoDigit": lastTwoDigit,
                    "englishAlphabet": englishAlphabet,
                    "persinalAlphabet": persinalAlphabet
                }

                try:
                    response = requests.post(URL, data=body, timeout=10)
                    if response.status_code == 200:
                        record_id = response.json().get('id', 'N/A')
                        self.log_message(f"  -> Success. ID: {record_id}")
                        success_count += 1
                    else:
                        self.log_message(f"  -> HTTP {response.status_code}: {response.text[:100]}", "WARNING")
                        fail_count += 1
                except Exception as e:
                    self.log_message(f"  -> Request error: {str(e)}", "ERROR")
                    fail_count += 1

            # Final report
            self.log_message(f"\n=== Processing finished ===")
            self.log_message(f"Successful: {success_count}, Failed: {fail_count}")

            self.root.after(0, lambda: messagebox.showinfo("Done", f"Upload completed.\nSuccess: {success_count}\nFailed: {fail_count}"))

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.log_message(error_msg, "ERROR")
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        finally:
            # Re-enable UI buttons (must be done in main thread)
            self.root.after(0, self.enable_buttons)

    def enable_buttons(self):
        self.process_btn.config(state=tk.NORMAL)
        self.browse_btn.config(state=tk.NORMAL)
        self.status_var.set("Ready")

# ----------------------------------------------------------------------


if __name__ == "__main__":
    root = tk.Tk()
    app = ExcelUploaderApp(root)
    root.mainloop()