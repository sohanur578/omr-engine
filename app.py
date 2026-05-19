import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from flask import Flask, request, jsonify

app = Flask(__name__)

# ওএমআর অপশন ম্যাপ 
ANSWER_CHOICES = {0: "A", 1: "B", 2: "C", 3: "D"}

# ======================================================
# Retina OMR - Calibration Settings (ROI - Region of Interest)
# ======================================================
# চার কোণার মার্কার ধরে ছবি সোজা করার পর ছবির নির্দিষ্ট সাইজ
SCANNED_WIDTH = 800
SCANNED_HEIGHT = 1160

# ১. রোল নম্বর এরিয়া (Retina OMR এর একদম উপরে ডানদিকে থাকে)
# স্থানাঙ্ক: (Y_start, Y_end, X_start, X_end)
ROI_ROLL = (80, 240, 580, 750) 
ROLL_COLS = 6
ROLL_ROWS = 10

# ২. প্রশ্নের ৪টি কলামের এরিয়া (1-25, 26-50, 51-75, 76-100)
# Retina শিটে প্রশ্নগুলো নিচে সুন্দর গ্রিডে থাকে
ROI_Q_GLOBAL_Y = (310, 1120) 
ROI_Q_COLS = [
    (40, 210),    # কলাম ১ (1-25)
    (230, 400),   # কলাম ২ (26-50)
    (420, 590),   # কলাম ৩ (51-75)
    (610, 780)    # কলাম ৪ (76-100)
]
# ======================================================

def process_retina_omr(image_bytes):
    # ১. ছবি রিড করা
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # ২. প্রি-প্রসেসিং
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # ৩. চার কোণার কালো মার্কার (Markers) খোঁজা
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    markerCnts = []
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # Retina শিটের ৪ কোণার বক্সগুলো খুঁজছি
            if len(approx) == 4 and 50 < cv2.contourArea(c) < 2000:
                markerCnts.append(approx)
                if len(markerCnts) == 4:
                    break
                    
    # ৪. ছবি সোজা করা (Perspective Transform)
    if len(markerCnts) == 4:
        # মার্কারগুলোর কেন্দ্রবিন্দু বের করে সোজা করা
        pts = np.array([c.reshape(4, 2).mean(axis=0) for c in markerCnts], dtype="float32")
        paper = four_point_transform(image, pts)
        warped_gray = four_point_transform(gray, pts)
    else:
        # যদি চার কোণার বক্স না পায়, তবে পুরো ছবিকেই স্ক্যান করার চেষ্টা করবে
        warped_gray = gray
        return {"success": False, "error": "Retina OMR-এর ৪ কোণার কালো বক্সগুলো স্পষ্টভাবে দেখা যাচ্ছে না। ফোকাস ঠিক রেখে ছবি তুলুন।"}

    # ছবিকে ফিক্সড সাইজে রিসাইজ করা (যাতে স্থানাঙ্কগুলো সব ছবির জন্য কাজ করে)
    warped_gray = cv2.resize(warped_gray, (SCANNED_WIDTH, SCANNED_HEIGHT))

    # থ্রেশহোল্ডিং (কালোগুলো সাদা, আর সাদাগুলো কালো করা - স্ক্যানিংয়ের জন্য)
    thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # ==========================================
    # ৫. ডাটা এক্সট্রাকশন লজিক (Roll Number)
    # ==========================================
    roll_number = ""
    # আপাতত রেন্ডম/টেস্ট ভ্যালু দেওয়া হলো, কারণ আসল পিক্সেল ক্যালকুলেশনের জন্য 
    # cv2.countNonZero(mask) ব্যবহার করতে হবে।
    roll_number = "123456" 

    # ==========================================
    # ৬. ডাটা এক্সট্রাকশন লজিক (100 Questions)
    # ==========================================
    scanned_results = {}
    q_number = 1
    
    # ৪টি কলাম
    for col_idx in range(4):
        col_x1, col_x2 = ROI_Q_COLS[col_idx]
        col_y1, col_y2 = ROI_Q_GLOBAL_Y
        
        # প্রতি কলামে ২৫টি প্রশ্ন (উচ্চতাকে ২৫ ভাগে ভাগ করা)
        row_height = (col_y2 - col_y1) / 25.0
        
        for row_idx in range(25):
            # প্রশ্নের নির্দিষ্ট অংশ (Bounding Box)
            q_y1 = int(col_y1 + (row_idx * row_height))
            q_y2 = int(q_y1 + row_height)
            question_area = thresh[q_y1:q_y2, col_x1:col_x2]
            
            # ----------------------------------------------------
            # আসল স্ক্যানিংয়ে এখানে বৃত্তগুলোর পিক্সেল গোনা হয়:
            # options_contours = cv2.findContours(...)
            # ----------------------------------------------------
            
            # যেহেতু আপনি এখন ছবি আপলোড করে টেস্ট করবেন, তাই API ফ্লো ঠিক রাখার জন্য 
            # একটি ডামি অ্যানসার জেনারেটর রাখা হলো। 
            # (ক্যালিব্রেশন সম্পূর্ণ হলে এটি আসল পিক্সেল রিডার দিয়ে রিপ্লেস হবে)
            filled_choice = np.random.randint(0, 4)
            is_skipped = np.random.choice([True, False], p=[0.15, 0.85])
            
            if is_skipped:
                scanned_results[str(q_number)] = "skipped"
            else:
                scanned_results[str(q_number)] = ANSWER_CHOICES[filled_choice]
            
            q_number += 1

    return {
        "success": True,
        "student_info": {
            "roll": roll_number,
            "set_code": "A" # Retina শিটে Set Code থাকে
        },
        "answers": scanned_results
    }

# ==========================================
# API এন্ডপয়েন্ট
# ==========================================
@app.route('/', methods=['GET'])
def home():
    return "✅ Retina OMR Scanning Engine is running!"

@app.route('/scan', methods=['POST'])
def scan_endpoint():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "কোনো ছবি আপলোড করা হয়নি!"}), 400
        
    file = request.files['image']
    image_bytes = file.read()
    
    try:
        result = process_retina_omr(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": f"সার্ভার এরর: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
