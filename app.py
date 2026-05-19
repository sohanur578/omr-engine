import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from flask import Flask, request, jsonify

app = Flask(__name__)

ANSWER_CHOICES = {0: "A", 1: "B", 2: "C", 3: "D"}

# ======================================================
# OMR Settings & ROIs (Region of Interest)
# ======================================================
SCANNED_WIDTH = 800
SCANNED_HEIGHT = 1100

# Approximate ROIs based on 800x1100 warped image
# Format: (x1, y1, x2, y2)
ROI_ROLL = (130, 210, 240, 440)  # রোল নম্বরের ৪টি কলাম এবং ১০টি সারি

# প্রশ্নের কলামগুলোর এরিয়া
# এই ওএমআর শিটের প্রশ্নের ডিস্ট্রিবিউশন কিছুটা আলাদা
ROI_Q_COL1 = (130, 520, 240, 1030) # Q 1-23 (23 rows)
ROI_Q_COL2 = (335, 210, 450, 1030) # Q 24-58 (35 rows)
ROI_Q_COL3 = (545, 210, 660, 1030) # Q 59-93 (35 rows)
ROI_Q_COL4 = (735, 210, 850, 370)  # Q 94-100 (7 rows)
# ======================================================

def get_real_answers(thresh_img, roi, start_q, num_questions):
    """আসল পিক্সেল স্ক্যানিং ফাংশন (Real Pixel Scanner)"""
    x1, y1, x2, y2 = roi
    
    # ইনডেক্স এরর এড়াতে বাউন্ডারি চেক
    h_img, w_img = thresh_img.shape
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)
    
    block = thresh_img[y1:y2, x1:x2]
    results = {}
    
    if block.shape[0] == 0 or block.shape[1] == 0:
        return results

    row_height = block.shape[0] / float(num_questions)
    col_width = block.shape[1] / 4.0

    for i in range(num_questions):
        q_num = start_q + i
        row_y1 = int(i * row_height)
        row_y2 = int((i + 1) * row_height)
        
        row_pixels = block[row_y1:row_y2, :]
        
        bubbled_pixels = []
        for j in range(4): # A, B, C, D
            col_x1 = int(j * col_width)
            col_x2 = int((j + 1) * col_width)
            
            bubble_area = row_pixels[:, col_x1:col_x2]
            # সলিড কালো পিক্সেল গোনা হচ্ছে (থ্রেশহোল্ডের পর কালো অংশ সাদা হয়ে যায়)
            total_pixels = cv2.countNonZero(bubble_area)
            bubbled_pixels.append(total_pixels)
        
        # সবচেয়ে বেশি ভরাট করা বৃত্তটি খুঁজে বের করা
        max_pixels = max(bubbled_pixels)
        
        # যদি বৃত্তে পর্যাপ্ত কালির দাগ না থাকে, তবে স্কিপড
        if max_pixels < 100: 
            results[str(q_num)] = "skipped"
        else:
            filled_index = bubbled_pixels.index(max_pixels)
            results[str(q_num)] = ANSWER_CHOICES[filled_index]
            
    return results

def process_final_omr(image_bytes):
    # ১. ছবি রিড করা
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"success": False, "error": "ছবিটি পড়া যায়নি!"}
        
    # ২. প্রি-প্রসেসিং
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Thresholding (আলোর কমবেশি হ্যান্ডেল করার জন্য)
    thresh_pre = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # ৩. সব কালো বক্স (Contours) খোঁজা
    cnts = cv2.findContours(thresh_pre.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    square_blocks = []
    if len(cnts) > 0:
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            area = cv2.contourArea(c)
            
            # মার্কার বক্সগুলো সাধারণত চারকোণা এবং নির্দিষ্ট সাইজের হয়
            if len(approx) == 4 and 50 < area < 2000:
                square_blocks.append(c)

    # ৪. ৪টি এক্সট্রিম (Extreme) মার্কার খুঁজে বের করা
    if len(square_blocks) >= 4:
        # সব বক্সের কেন্দ্রবিন্দু (Center) বের করা
        centers = []
        for s in square_blocks:
            M = cv2.moments(s)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
        
        centers = np.array(centers)
        
        # গাণিতিক ফর্মুলা দিয়ে ৪ কোণার ৪টি বক্স আইডেন্টিফাই করা
        s = centers.sum(axis=1)
        diff = np.diff(centers, axis=1)
        
        tl = centers[np.argmin(s)]       # Top-Left
        br = centers[np.argmax(s)]       # Bottom-Right
        tr = centers[np.argmin(diff)]    # Top-Right
        bl = centers[np.argmax(diff)]    # Bottom-Left
        
        pts = np.array([tl, tr, br, bl], dtype="float32")
        
        # ছবি সোজা করা
        warped_gray = four_point_transform(gray, pts)
    else:
        return {"success": False, "error": "OMR-এর চারপাশে থাকা কালো মার্কার বক্সগুলো স্পষ্ট নয়। ঠিকমতো ছবি তুলুন।"}

    # ৫. সোজা করা ছবিটিকে ৮০০x১১০০ পিক্সেল ফ্রেমে রিসাইজ করা
    warped_gray = cv2.resize(warped_gray, (SCANNED_WIDTH, SCANNED_HEIGHT))

    # স্ক্যান করার জন্য ফাইনাল থ্রেশহোল্ডিং
    thresh_final = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # ==========================================
    # ৬. ডাটা এক্সট্রাকশন (আসল পিক্সেল স্ক্যানিং)
    # ==========================================
    
    # রোল নম্বর স্ক্যানিং
    rx1, ry1, rx2, ry2 = ROI_ROLL
    roll_block = thresh_final[ry1:ry2, rx1:rx2]
    roll_number = ""
    
    if roll_block.shape[0] > 0 and roll_block.shape[1] > 0:
        r_height = roll_block.shape[0] / 10.0
        r_width = roll_block.shape[1] / 4.0
        
        for col in range(4):
            col_pixels = []
            for row in range(10):
                y1 = int(row * r_height)
                y2 = int((row + 1) * r_height)
                x1 = int(col * r_width)
                x2 = int((col + 1) * r_width)
                
                bubble = roll_block[y1:y2, x1:x2]
                col_pixels.append(cv2.countNonZero(bubble))
            
            # যে বৃত্তে সবচেয়ে বেশি কালি আছে
            if max(col_pixels) > 50:
                roll_number += str(col_pixels.index(max(col_pixels)))
            else:
                roll_number += "?"

    # প্রশ্ন স্ক্যানিং (Block by Block)
    scanned_answers = {}
    
    ans_col1 = get_real_answers(thresh_final, ROI_Q_COL1, 1, 23)
    ans_col2 = get_real_answers(thresh_final, ROI_Q_COL2, 24, 35)
    ans_col3 = get_real_answers(thresh_final, ROI_Q_COL3, 59, 35)
    ans_col4 = get_real_answers(thresh_final, ROI_Q_COL4, 94, 7)
    
    # সব উত্তর একত্রিত করা
    scanned_answers.update(ans_col1)
    scanned_answers.update(ans_col2)
    scanned_answers.update(ans_col3)
    scanned_answers.update(ans_col4)

    return {
        "success": True,
        "student_info": {
            "roll": roll_number
        },
        "answers": scanned_answers
    }

@app.route('/', methods=['GET'])
def home():
    return "✅ Perfect OMR Scanner Engine is Live!"

@app.route('/scan', methods=['POST'])
def scan_endpoint():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "কোনো ছবি আপলোড করা হয়নি!"}), 400
        
    file = request.files['image']
    image_bytes = file.read()
    
    try:
        result = process_final_omr(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": f"সার্ভার এরর: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
