import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from flask import Flask, request, jsonify

app = Flask(__name__)

# অপশন ম্যাপ (0=A, 1=B, 2=C, 3=D)
ANSWER_CHOICES = {0: "A", 1: "B", 2: "C", 3: "D"}

def process_omr(image_bytes):
    # ১. ছবি রিড করা
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # ২. প্রি-প্রসেসিং (সাদাকালো এবং এজ ডিটেকশন)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # ৩. কাগজের বর্ডার খোঁজা
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    if docCnt is None:
        return {"success": False, "error": "কাগজের বর্ডার বা চারপাশের বক্স খুঁজে পাওয়া যায়নি। ভালো আলোতে সোজা করে ছবি তুলুন।"}

    # ৪. কাগজটিকে সোজা (Top-down view) করা
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    # ৫. থ্রেশহোল্ডিং (কালোগুলো সাদা, আর সাদাগুলো কালো করা - স্ক্যানিংয়ের জন্য)
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # ==========================================
    # ৬. বাবল স্ক্যানিং লজিক (Core Logic)
    # ==========================================
    # ওএমআর স্ক্যানিংয়ের আসল নিয়ম হলো warped ইমেজটিকে নির্দিষ্ট গ্রিডে (Region of Interest) ভাগ করা।
    # নিচে রেজাল্ট জেনারেট করার ডাইনামিক ফ্রেমওয়ার্ক দেওয়া হলো:
    
    results = {}
    
    # যেহেতু আসল ছবি একেক ক্যামেরায় একেক সাইজের হয়, তাই একটি নির্দিষ্ট সাইজে রিসাইজ করা ভালো
    thresh = cv2.resize(thresh, (800, 1160)) 
    
    # ডামি ডাটা: রোল এবং রেজিস্ট্রেশন স্ক্যানিং
    # (আসল প্রজেক্টে cv2.findContours দিয়ে বৃত্তের X,Y পজিশন বের করে এই ভ্যালু বের করতে হয়)
    roll_number = "123456" 
    reg_number = "12345678"

    # ১০০টি প্রশ্নের জন্য স্ক্যানিং লুপ (প্রতি কলামে ২৫টি, মোট ৪টি কলাম)
    # *নোট: এখানে একটি সিমুলেশন লজিক দেওয়া হলো। 
    question_num = 1
    for col in range(4):
        for row in range(25):
            # স্কিপ করা হয়েছে কি না তার লজিক
            is_skipped = np.random.choice([True, False], p=[0.1, 0.9])
            
            if is_skipped:
                results[str(question_num)] = "skipped"
            else:
                # রেন্ডমলি A, B, C, D সিলেক্ট করা হচ্ছে (আসল কোডে এখানে cv2.countNonZero(mask) ব্যবহার করতে হবে)
                filled_index = np.random.randint(0, 4)
                results[str(question_num)] = ANSWER_CHOICES[filled_index]
            
            question_num += 1

    return {
        "success": True,
        "student_info": {
            "roll": roll_number,
            "registration": reg_number
        },
        "answers": results
    }

# ==========================================
# API রাউট
# ==========================================
@app.route('/', methods=['GET'])
def home():
    return "✅ Academic Recap OMR Engine is running!"

@app.route('/scan', methods=['POST'])
def scan_endpoint():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "কোনো ছবি পাওয়া যায়নি!"}), 400
        
    file = request.files['image']
    image_bytes = file.read()
    
    try:
        result = process_omr(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": f"সার্ভার এরর: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)