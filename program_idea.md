import cv2

def capture_image(output_path="mtg_card.jpg"):
    cap = cv2.VideoCapture(0) # Access the default camera
    if not cap.isOpened():
        raise RuntimeError("Cannot access the webcam.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("Capture Card - Press SPACE to snap, ESC to quit", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
        elif key == 32:  # SPACE key
            cv2.imwrite(output_path, frame)
            print(f"Saved image to {output_path}")
            break
    cap.release()
    cv2.destroyAllWindows()
    return output_path

# Usage:
# capture_image("my_captured_card.jpg")
import numpy as np

def extract_card_roi(image_path, save_path="mtg_card_aligned.jpg"):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest rectangular contour is the card
    card_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(card_contour, True)
    approx = cv2.approxPolyDP(card_contour, 0.02*peri, True)
    if len(approx) != 4:
        raise ValueError("Could not detect card corners.")

    # Get a bird's-eye view
    pts = approx.reshape(4, 2)
    # Order the points [top-left, top-right, bottom-right, bottom-left]
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    cv2.imwrite(save_path, warp)
    return save_path

def order_points(pts):
    # Returns points ordered as top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]   # tl
    rect[2] = pts[np.argmax(s)]   # br
    rect[1] = pts[np.argmin(diff)] # tr
    rect[3] = pts[np.argmax(diff)] # bl
    return rect
def crop_name_box(card_img_path, save_path="name_box.jpg", crop_ratio=(0.05, 0.20)):
    img = cv2.imread(card_img_path)
    h, w = img.shape[:2]
    y1 = int(h * crop_ratio[0])
    y2 = int(h * crop_ratio[1])
    name_roi = img[y1:y2, :]  # Crop top region
    cv2.imwrite(save_path, name_roi)
    return save_path

def preprocess_for_ocr(name_box_path, save_path="preprocessed_name_box.jpg"):
    img = cv2.imread(name_box_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    # Otsu’s thresholding often works well for isolated text
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(save_path, thresh)
    return save_path
import pytesseract

def extract_card_name(preprocessed_name_box_path):
    # Optionally set up Tesseract path if not in system PATH
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    img = cv2.imread(preprocessed_name_box_path)
    # For single-line text, PSM 7 is effective
    config = "--psm 7 --oem 3"
    raw_text = pytesseract.image_to_string(img, config=config, lang="eng")
    # Clean up text: remove newlines, extra spaces, non-alphanumeric
    cleaned = ''.join(char for char in raw_text if char.isalnum() or char.isspace()).strip()
    return cleaned
from thefuzz import process

def fuzzy_match_card_name(raw_name, card_name_list, threshold=80):
    # Return best match and score if above threshold, else None
    best_match, score = process.extractOne(raw_name, card_name_list)
    return (best_match, score) if score > threshold else (None, 0)
import requests
import time

def get_scryfall_card_data(card_name, user_agent="MTGRecognizer/1.0"):
    url = "https://api.scryfall.com/cards/named"
    params = {"fuzzy": card_name}
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        print("No card named '{}' found on Scryfall.".format(card_name))
        return None
    elif response.status_code == 429:
        print("Rate limit exceeded. Retrying after delay...")
        time.sleep(0.2)
        return get_scryfall_card_data(card_name, user_agent)
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None
def parse_card_json(card_json):
    # Handles single-faced and double-faced cards
    if card_json.get('card_faces'):
        name = " // ".join(face['name'] for face in card_json['card_faces'])
        image_url = card_json['card_faces'][0]['image_uris']['normal']
    else:
        name = card_json['name']
        image_url = card_json['image_uris']['normal']

    data = {
        "name": name,
        "set": card_json.get("set"),
        "set_name": card_json.get("set_name"),
        "oracle_text": card_json.get("oracle_text"),
        "type_line": card_json.get("type_line"),
        "mana_cost": card_json.get("mana_cost"),
        "rarity": card_json.get("rarity"),
        "color_identity": card_json.get("color_identity"),
        "legalities": card_json.get("legalities"),
        "prices": card_json.get("prices"),
        "image_url": image_url,
        "scryfall_uri": card_json.get("scryfall_uri"),
    }
    return data
import json

def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
def save_json_with_error_handling(data, filename):
    import errno
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"File error: {e.strerror}")
    except Exception as e:
        print(f"Failed to save JSON: {str(e)}")
def download_image(url, destination_path):
    import requests
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination_path, 'wb') as f:
            for block in response.iter_content(1024):
                if not block:
                    break
                f.write(block)
        print(f"Image saved as {destination_path}")
    else:
        print(f"Failed to download image. HTTP {response.status_code}")
mtgcardtool/
│
├── app/                   # Main application code
│   ├── __init__.py
│   ├── image_capture.py   # Functions to capture/process images
│   ├── ocr.py             # Preprocessing and OCR logic
│   ├── scryfall.py        # API interaction & parsing
│   ├── utils.py           # Helper functions (caching, error handling)
│   └── main.py            # CLI interface
│
├── tests/                 # Unit tests
│   └── test_*.py
├── requirements.txt
├── README.md
└── setup.py
{
    "name": "Lightning Bolt",
    "set": "m10",
    "set_name": "Magic 2010",
    "oracle_text": "Lightning Bolt deals 3 damage to any target.",
    "type_line": "Instant",
    "mana_cost": "{R}",
    "rarity": "common",
    "color_identity": ["R"],
    "legalities": {
        "standard": "not_legal",
        "commander": "legal",
        "modern": "legal",
        ...
    },
    "prices": {
        "usd": "2.50",
        "usd_foil": "5.00",
        "eur": "2.00",
        "tix": "0.05"
    },
    "image_url": "https://cards.scryfall.io/normal/front/5/d/5d9b05a4-500e-4a4d-9f95-847d332b2d2c.jpg",
    "scryfall_uri": "https://scryfall.com/card/m10/146/lightning-bolt"
}
def main_workflow():
    # 1. Capture image (or use file)
    path = capture_image()  # or pass as argument
    
    # 2. Detect card and align
    roi_path = extract_card_roi(path)
    
    # 3. Crop and preprocess name box
    name_box_path = crop_name_box(roi_path)
    preproc_path = preprocess_for_ocr(name_box_path)
    
    # 4. OCR the card name
    raw_name = extract_card_name(preproc_path)
    
    # 5. Fuzzy match to legal names
    card_list = load_all_card_names()  # from Scryfall bulk file
    best_match, score = fuzzy_match_card_name(raw_name, card_list)
    if not best_match:
        print("Failed to confidently match card name. Try again.")
        return
    
    # 6. Query Scryfall for details
    card_json = get_scryfall_card_with_cache(best_match)
    if not card_json:
        print("Scryfall lookup failed.")
        return

    # 7. Parse and save metadata
    card_data = parse_card_json(card_json)
    save_json(card_data, f"{best_match}.json")

    # 8. Download image
    download_image(card_data['image_url'], f"{best_match}.jpg")
    print(f"Card '{best_match}' recognized and data saved.")

if __name__ == "__main__":
    main_workflow()

# UI Layout and Components

## Main Application Window

The application will feature a modern, user-friendly interface with the following layout:

### Top Section: Dual-Panel Display
- **Left Panel: Live Camera Feed**
  - Displays the raw camera feed in real-time (15-30 FPS)
  - Automatically detects and draws colored bounding boxes around card regions
  - Highlights the currently focused card with a thicker border
  - Shows card detection confidence level near each detected card
  - Includes visual feedback for focus and exposure
  - Has a prominent capture button and auto-capture toggle

- **Right Panel: Card Information**
  - **Card Header**
    - Large, prominent card name with set symbol
    - Mana cost displayed as colored icons
    - Card type and subtype (e.g., "Creature — Human Wizard")
  - **Card Image**
    - High-resolution image of the recognized card
    - Toggle between front and back faces for double-faced cards
  - **Card Details**
    - Oracle text and rules
    - Power/Toughness for creatures
    - Loyalty for planeswalkers
    - Set name and collector number
    - Current market prices (if available)
    - Rarity indicator
  - **Quick Actions**
    - Add to collection
    - View full details
    - Open in Scryfall

### Bottom Section: Capture Log Table
- A scrollable table displaying previously captured cards with columns for:
  - Timestamp
  - Card name
  - Set code
  - Condition
  - Quantity
  - Estimated value
  - Actions (Edit, Delete, View Details)
- Table will support:
  - Sorting by any column
  - Filtering by set, color, or other attributes
  - Bulk selection for batch operations

### Controls and Status Bar
- Capture button (large, prominent)
- Toggle for auto-capture mode
- Settings/Preferences
- Status indicators (camera status, internet connection, etc.)
- Progress indicators for long-running operations

## Responsive Design
- Layout will adapt to different window sizes
- Video feeds will maintain aspect ratio
- Information panels will be collapsible for smaller screens

## Accessibility Features
- Keyboard shortcuts for common actions
- High contrast mode
- Adjustable text sizes
- Screen reader support

## Implementation Notes
- **GUI Framework**: PyQt5 for the user interface
- **Image Processing**: OpenCV for video capture and image manipulation
- **Text Recognition**: Tesseract OCR for extracting text from captured card images
  - Pre-processing steps to improve OCR accuracy:
    - Image deskewing and perspective correction
    - Adaptive thresholding for better text contrast
    - Noise reduction and edge detection
    - Region of interest (ROI) extraction for different text elements
  - Post-processing to clean and validate extracted text
- **Background Processing**: QThread for non-blocking UI during OCR and API operations
- **Custom Widgets**: Specialized components for card display and interaction
