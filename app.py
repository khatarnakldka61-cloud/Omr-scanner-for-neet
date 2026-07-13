import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
from PIL import Image

def process_omr_image(image_array, choices_per_question=5):
    # Convert image to grayscale and find edges
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Find the OMR document outline
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

    if doc_cnt is None:
        return None, "Could not detect the edges of the paper. Please try a clearer photo."

    # Warp image to a top-down view
    warped = four_point_transform(gray, doc_cnt.reshape(4, 2))
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find bubbles
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    question_cnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
            question_cnts.append(c)

    if not question_cnts:
        return None, "Could not detect bubbles. Ensure the image is well-lit."

    question_cnts = contours.sort_contours(question_cnts, method="top-to-bottom")[0]
    extracted_answers = []

    # Read marked answers
    for (q, i) in enumerate(np.arange(0, len(question_cnts), choices_per_question)):
        cnts_row = contours.sort_contours(question_cnts[i:i + choices_per_question])[0]
        bubbled = None

        for (j, c) in enumerate(cnts_row):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        extracted_answers.append(bubbled[1])

    return extracted_answers, "Success"

# --- USER INTERFACE ---
st.title("📝 Instant OMR Checker")
st.write("Upload your Answer Key, then upload the Student Sheet to get a score.")

key_file = st.file_uploader("1. Upload Answer Key Image", type=["jpg", "jpeg", "png"])
student_file = st.file_uploader("2. Upload Student OMR Image", type=["jpg", "jpeg", "png"])

if key_file and student_file:
    if st.button("Calculate Score"):
        with st.spinner("Analyzing sheets..."):
            # Convert uploaded files to OpenCV format
            key_image = np.array(Image.open(key_file).convert('RGB'))
            key_image = cv2.cvtColor(key_image, cv2.COLOR_RGB2BGR)
            
            student_image = np.array(Image.open(student_file).convert('RGB'))
            student_image = cv2.cvtColor(student_image, cv2.COLOR_RGB2BGR)

            # Process both images
            key_answers, key_msg = process_omr_image(key_image)
            student_answers, student_msg = process_omr_image(student_image)

            if key_answers is None:
                st.error(f"Answer Key Error: {key_msg}")
            elif student_answers is None:
                st.error(f"Student Sheet Error: {student_msg}")
            elif len(key_answers) != len(student_answers):
                st.error("Error: The number of detected bubbles doesn't match between the two images.")
            else:
                # Calculate Score
                score = 0
                incorrect = []
                for i in range(len(key_answers)):
                    if key_answers[i] == student_answers[i]:
                        score += 1
                    else:
                        incorrect.append(i + 1)
                
                max_score = len(key_answers)
                
                st.success(f"### Final Score: {score} / {max_score}")
                st.write(f"**Accuracy:** {(score/max_score)*100:.2f}%")
                
                if incorrect:
                    st.warning(f"**Questions to review in your mistake diary:** {', '.join(map(str, incorrect))}")
                else:
                    st.balloons()
                    st.write("Perfect score!")
