import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from flask import Flask, request, jsonify

app = Flask(__name__)

ANSWER_CHOICES = {0: "A", 1: "B", 2: "C", 3: "D"}

# ======================================================
# OMR Settings & ROIs 
# ======================================================
SCANNED_WIDTH = 800
SCANNED_HEIGHT = 1100

ROI_ROLL = (130, 210, 240, 440)  

ROI_Q_COL1 = (130, 520, 240, 1030) 
ROI_Q_COL2 = (335, 210, 450, 1030) 
ROI_Q_COL3 = (545, 210, 660, 1030) 
ROI_Q_COL4 = (735, 210, 850, 370)  
# ======================================================

def get_real_answers(thresh_img, roi, start_q, num_questions):
    """Smart Pixel Scanner (Border Ignore Logic)"""
    x1, y1, x2, y2 = roi
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
            
            # 👉 SMART CROP: চারপাশ থেকে ২৫% কেটে ফেলা, যাতে বৃত্তের কালো বর্ডার বাদ যায়
            bh, bw = bubble_area.shape
            if bh > 4 and bw > 4:
                pad_h, pad_w = int(bh * 0.25), int(bw * 0.25)
                inner_bubble = bubble_area[pad_h:bh-pad_h, pad_w:bw-pad_w]
            else:
                inner_bubble = bubble_area
                
            total_pixels = cv2.countNonZero(inner_bubble)
            bubbled_pixels.append(total_pixels)
        
        max_pixels = max(bubbled_pixels)
        min_pixels = min(bubbled_pixels)
        
        # 👉 SMART LOGIC: পার্থক্য কম হলে বা পিক্সেল কম হলে মানে কাগজ খালি
        if max_pixels < 40 or (max_pixels - min_pixels) < 30: 
            results[str(q_num)] = "skipped"
        else:
            filled_index = bubbled_pixels.index(max_pixels)
            results[str(q_num)] = ANSWER_CHOICES[filled_index]
            
    return results

def process_final_omr(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"success": False, "error": "ছবিটি পড়া যায়নি!"}
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_pre = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    cnts = cv2.findContours(thresh_pre.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    square_blocks = []
    if len(cnts) > 0:
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            area = cv2.contourArea(c)
            if len(approx) == 4 and 50 < area < 2000:
                square_blocks.append(c)

    if len(square_blocks) >= 4:
        centers = []
        for s in square_blocks:
            M = cv2.moments(s)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
        
        centers = np.array(centers)
        s = centers.sum(axis=1)
        diff = np.diff(centers, axis=1)
        
        tl = centers[np.argmin(s)]       
        br = centers[np.argmax(s)]       
        tr = centers[np.argmin(diff)]    
        bl = centers[np.argmax(diff)]    
        
        pts = np.array([tl, tr, br, bl], dtype="float32")
        warped_gray = four_point_transform(gray, pts)
    else:
        return {"success": False, "error": "OMR-এর চারপাশে থাকা কালো মার্কার বক্সগুলো স্পষ্ট নয়। ঠিকমতো ছবি তুলুন।"}

    warped_gray = cv2.resize(warped_gray, (SCANNED_WIDTH, SCANNED_HEIGHT))
    thresh_final = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # ==========================================
    # ডাটা এক্সট্রাকশন (আসল পিক্সেল স্ক্যানিং)
    # ==========================================
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
                
                # রোল নম্বরের জন্যও বর্ডার বাদ দেওয়া
                bh, bw = bubble.shape
                if bh > 4 and bw > 4:
                    ph, pw = int(bh * 0.25), int(bw * 0.25)
                    inner_b = bubble[ph:bh-ph, pw:bw-pw]
                else:
                    inner_b = bubble
                    
                col_pixels.append(cv2.countNonZero(inner_b))
            
            max_p = max(col_pixels)
            min_p = min(col_pixels)
            
            if max_p > 20 and (max_p - min_p) > 15:
                roll_number += str(col_pixels.index(max_p))
            else:
                roll_number += "?"

    scanned_answers = {}
    scanned_answers.update(get_real_answers(thresh_final, ROI_Q_COL1, 1, 23))
    scanned_answers.update(get_real_answers(thresh_final, ROI_Q_COL2, 24, 35))
    scanned_answers.update(get_real_answers(thresh_final, ROI_Q_COL3, 59, 35))
    scanned_answers.update(get_real_answers(thresh_final, ROI_Q_COL4, 94, 7))

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
