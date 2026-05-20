import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

ANSWER_CHOICES = {0: "A", 1: "B", 2: "C", 3: "D"}

# ══════════════════════════════════════════════════════════════════
#  CALIBRATION  (actual image analysis থেকে নেওয়া — পরিবর্তন করবেন না)
# ══════════════════════════════════════════════════════════════════

WARP_W    = 1133
WARP_H    = 1661
PX_MM     = 6.0588
ORIG_MM   = 11.5   # TL marker center থেকে coordinate শুরু

# বাবলের radius (px)
BUBBLE_R  = int(2.8 * PX_MM)   # ≈ 16px

# কলামের X positions (bubble center, px) — A,B,C,D
COL_BUBBLE_X = []
for cx in [18, 61.5, 105, 148.5]:
    row_xs = []
    for idx in range(4):
        bx_mm = cx + 7.5 + 1.5 + idx * 8.0 + 2.5
        row_xs.append(int((bx_mm - ORIG_MM) * PX_MM))
    COL_BUBBLE_X.append(row_xs)

# Row Y centers (px) — row 0-indexed, 25 rows per column
ROW_Y = []
for row in range(25):
    cy_mm = 33.0 + row * 9.5 + 4.75
    ROW_Y.append(int((cy_mm - ORIG_MM) * PX_MM))

# ══════════════════════════════════════════════════════════════════
#  THRESHOLD টিউনিং (আপডেটেড)
#  Adaptive threshold ব্যবহারের কারণে স্কোরের মান সামান্য পরিবর্তন করা হয়েছে।
# ══════════════════════════════════════════════════════════════════
FILL_THRESHOLD = 120   # blank ≈ 30-80, filled ≈ 150-400
DIFF_THRESHOLD = 40    # blank diff ≈ 10-30, filled diff ≈ 80+
INNER_PAD_RATIO = 0.25 # বাবলের ভেতরের কতটুকু দেখব (চারপাশের অক্ষর/border বাদ)


def find_markers(gray):
    """৪টি কালো স্কয়ার মার্কার খুঁজে বের করা (বাঁকা ও ছায়াযুক্ত ছবির জন্য অপ্টিমাইজড)।"""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Adaptive threshold - ছায়া এবং কম আলোতে ভালো কাজ করে
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10
    )
    
    # মার্কারের ভেতরের সাদা নয়েজ ফিল করতে dilate করা হচ্ছে
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = []
    for c in cnts:
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area   = cv2.contourArea(c)
        
        # Area range বাড়ানো হয়েছে যাতে জুম করা বা দূরের ছবিও কাজ করে
        if 150 < area < 25000:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h)
            
            # Convex Hull দিয়ে solidity চেক করা হচ্ছে (বক্সটি কতটা ভরাট)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = area / float(hull_area)
            
            # Aspect ratio ফ্লেক্সিবল করা হয়েছে (0.5 - 2.0) যাতে ছবি বাঁকা হলেও কাজ করে
            if len(approx) >= 4 and 0.5 < aspect < 2.0 and solidity > 0.8:
                cx_m = x + w // 2
                cy_m = y + h // 2
                markers.append((cx_m, cy_m, area))

    if len(markers) < 4:
        return None, f"মাত্র {len(markers)}টি মার্কার পাওয়া গেছে, দরকার ৪টি"

    # সবচেয়ে বড় ৪টি এরিয়া নেওয়া (ব্যাকগ্রাউন্ড নয়েজ বাদ দিতে)
    markers.sort(key=lambda m: m[2], reverse=True)
    centers = np.array([(m[0], m[1]) for m in markers[:4]], dtype="float32")

    # TL, TR, BR, BL সাজানো (Perspective warp এর জন্য অত্যন্ত জরুরি)
    s    = centers.sum(axis=1)
    diff = np.diff(centers, axis=1)
    tl   = centers[np.argmin(s)]
    br   = centers[np.argmax(s)]
    tr   = centers[np.argmin(diff)]
    bl   = centers[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32"), None


def warp_image(gray, marker_pts):
    """Perspective warp করে নির্দিষ্ট আকারে resize।"""
    dst = np.array([
        [0,      0],
        [WARP_W, 0],
        [WARP_W, WARP_H],
        [0,      WARP_H],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(marker_pts, dst)
    return cv2.warpPerspective(gray, M, (WARP_W, WARP_H))


def bubble_score(thresh_img, cy, cx):
    """একটি বাবলের ভরাট স্কোর বের করা (inner crop)।"""
    pad  = int(BUBBLE_R * INNER_PAD_RATIO)
    y1   = max(0, cy - BUBBLE_R + pad)
    y2   = min(thresh_img.shape[0], cy + BUBBLE_R - pad)
    x1   = max(0, cx - BUBBLE_R + pad)
    x2   = min(thresh_img.shape[1], cx + BUBBLE_R - pad)
    cell = thresh_img[y1:y2, x1:x2]
    return cv2.countNonZero(cell) if cell.size > 0 else 0


def scan_answers(warped_gray):
    """Warped grayscale থেকে ১০০টি উত্তর বের করা।"""
    # গ্লোবাল Otsu এর বদলে Adaptive Thresholding (ছায়া/আলোর সমস্যার জন্য)
    blurred_warped = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred_warped, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=45, C=12
    )

    answers = {}
    q_num   = 1

    for col_idx in range(4):
        xs = COL_BUBBLE_X[col_idx]  # [A_px, B_px, C_px, D_px]

        for row_idx in range(25):
            cy     = ROW_Y[row_idx]
            scores = [bubble_score(thresh, cy, xs[opt]) for opt in range(4)]

            max_s  = max(scores)
            min_s  = min(scores)
            diff   = max_s - min_s

            if max_s < FILL_THRESHOLD or diff < DIFF_THRESHOLD:
                answers[str(q_num)] = "skipped"
            else:
                answers[str(q_num)] = ANSWER_CHOICES[scores.index(max_s)]

            q_num += 1

    return answers


def process_omr(image_bytes):
    """মূল pipeline।"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return {"success": False, "error": "Invalid image — decode failed"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: মার্কার খোঁজা
    marker_pts, err = find_markers(gray)
    if marker_pts is None:
        # Fallback: ছবি সোজা হলে সরাসরি resize করে চেষ্টা
        warped = cv2.resize(gray, (WARP_W, WARP_H))
        warn   = err + " — সরাসরি resize দিয়ে চেষ্টা করা হচ্ছে"
    else:
        warped = warp_image(gray, marker_pts)
        warn   = None

    # Step 2: উত্তর স্ক্যান
    answers = scan_answers(warped)

    # Step 3: Summary
    answered = sum(1 for v in answers.values() if v != "skipped")
    skipped  = 100 - answered

    result = {
        "success":  True,
        "summary":  {
            "total":    100,
            "answered": answered,
            "skipped":  skipped,
        },
        "answers":  answers,
    }
    if warn:
        result["warning"] = warn
    return result


# ══════════════════════════════════════════════════════════════════
#  Flask Routes
# ══════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status":    "✅ Academic Recap OMR Engine v3 (Pro Vision)",
        "endpoints": {
            "POST /scan":        "multipart/form-data, field='image'",
            "POST /scan/base64": "JSON body: {\"image\": \"<base64>\"}",
            "GET  /health":      "health check",
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/scan', methods=['POST'])
def scan_file():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No 'image' field"}), 400
    file = request.files['image']
    if not file.filename:
        return jsonify({"success": False, "error": "Empty file"}), 400
    try:
        return jsonify(process_omr(file.read()))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/scan/base64', methods=['POST'])
def scan_base64():
    import base64
    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "JSON must have 'image' (base64 string)"}), 400
    try:
        image_bytes = base64.b64decode(data['image'])
        return jsonify(process_omr(image_bytes))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
