import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
from PIL import Image

def process_omr_image(image_array, choices_per_question=4):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    doc_cnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break

    if doc_cnt is not None:
        warped = four_point_transform(gray, doc_cnt.reshape(4, 2))
    else:
        warped = gray

    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    question_cnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 10 and h >= 10 and 0.5 <= ar <= 1.5:
            question_cnts.append(c)

    if len(question_cnts) != 720:
        return None, f"Found {len(question_cnts)} bubbles instead of exactly 720. Please ensure the image is perfectly flat, well-lit, and none of the bubbles are cut off."

    question_cnts = contours.sort_contours(question_cnts, method="left-to-right")[0]

    col1 = question_cnts[0:180]
    col2 = question_cnts[180:360]
    col3 = question_cnts[360:540]
    col4 = question_cnts[540:720]
    
    columns = [col1, col2, col3, col4]
    extracted_answers = []

    for col in columns:
        col = contours.sort_contours(col, method="top-to-bottom")[0]
        
        for (q, i) in enumerate(np.arange(0, len(col), choices_per_question)):
            cnts_row = contours.sort_contours(col[i:i + choices_per_question])[0]
            bubbled = None

            for (j, c) in enumerate(cnts_row):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)

                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)

            # If the darkest bubble doesn't have enough filled pixels, mark as unattempted (-1)
            # 35 pixels is a safe low threshold to ignore smudges
            if bubbled[0] > 35:
                extracted_answers.append(bubbled[1])
            else:
                extracted_answers.append(-1)

    return extracted_answers, "Success"

# --- USER INTERFACE ---
st.title("📝 Instant OMR Checker")
st.write("Upload your Answer Key, then upload the Student Sheet to get a score.")

key_file = st.file_uploader("1. Upload Answer Key Image", type=["jpg", "jpeg", "png"])
student_file = st.file_uploader("2. Upload Student OMR Image", type=["jpg", "jpeg", "png"])

if key_file and student_file:
    if st.button("Calculate Score"):
        with st.spinner("Analyzing sheets..."):
            key_image = np.array(Image.open(key_file).convert('RGB'))
            key_image = cv2.cvtColor(key_image, cv2.COLOR_RGB2BGR)
            
            student_image = np.array(Image.open(student_file).convert('RGB'))
            student_image = cv2.cvtColor(student_image, cv2.COLOR_RGB2BGR)

            key_answers, key_msg = process_omr_image(key_image)
            student_answers, student_msg = process_omr_image(student_image)

            if key_answers is None:
                st.error(f"Answer Key Error: {key_msg}")
            elif student_answers is None:
                st.error(f"Student Sheet Error: {student_msg}")
            else:
                score = 0
                correct = 0
                unattempted = 0
                incorrect = []
                
                for i in range(len(key_answers)):
                    if student_answers[i] == -1:
                        unattempted += 1
                    elif key_answers[i] == student_answers[i]:
                        score += 4
                        correct += 1
                    else:
                        score -= 1
                        incorrect.append(i + 1)
                
                max_score = len(key_answers) * 4
                
                st.success(f"### Final Score: {score} / {max_score}")
                
                # Show a breakdown of the performance
                col1, col2, col3 = st.columns(3)
                col1.metric("Correct (+4)", correct)
                col2.metric("Incorrect (-1)", len(incorrect))
                col3.metric("Unattempted (0)", unattempted)
                
                if incorrect:
                    st.warning(f"**Questions to log in your mistake diary:** {', '.join(map(str, incorrect))}")
                else:
                    st.balloons()
                    st.write("Perfect score!")
