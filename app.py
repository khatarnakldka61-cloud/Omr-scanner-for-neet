"""
NEET OMR Checker — Production-grade Streamlit app
==================================================

Dependencies: streamlit, opencv-python-headless, numpy, scipy, imutils, pillow, pandas
"""

import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import imutils
from scipy.cluster.vq import kmeans2
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
TOTAL_QUESTIONS = 180
OPTIONS_PER_Q = 4
NUM_COLUMNS = 4
ROWS_PER_COLUMN = TOTAL_QUESTIONS // NUM_COLUMNS  # 45

MARKS_CORRECT = 4
MARKS_INCORRECT = -1
MARKS_UNATTEMPTED = 0

# Low strictness to handle different lighting and pen types
FILL_THRESHOLD = 0.15      
AMBIGUOUS_MARGIN = 0.06    

OPTION_LABELS = ["1", "2", "3", "4"]

# ----------------------------------------------------------------------------
# BASIC IMAGE UTILITIES
# ----------------------------------------------------------------------------
def load_image_from_upload(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    # Only used if a full page is detected
    out_w, out_h = 1400, 1980 
    rect = order_points(pts)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    return warped

# ----------------------------------------------------------------------------
# DOCUMENT DETECTION / DESKEW 
# ----------------------------------------------------------------------------
def try_find_document_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=2)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    img_area = image.shape[0] * image.shape[1]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 0.35 * img_area:
            return approx.reshape(4, 2)
    return None

def deskew_via_rotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:
        return image  

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5 or abs(angle) > 15:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated

def get_normalized_sheet(image):
    corners = try_find_document_corners(image)
    if corners is not None:
        try:
            return four_point_transform(image, corners.astype("float32")), "perspective"
        except Exception:
            pass

    # FIXED: Just rotate it if it's crooked, but DO NOT forcefully resize it. 
    # This preserves perfectly cropped screenshots!
    deskewed = deskew_via_rotation(image)
    return deskewed, "deskew-fallback"

# ----------------------------------------------------------------------------
# PREPROCESSING FOR BUBBLE / INK DETECTION
# ----------------------------------------------------------------------------
def build_binary_ink_image(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=35,
        C=10,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return binary

# ----------------------------------------------------------------------------
# BUBBLE DETECTION -> GRID BOUNDARY ESTIMATION
# ----------------------------------------------------------------------------
def find_bubble_like_contours(binary):
    cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    candidates = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Lowered size limit to 8px so it doesn't ignore bubbles in cropped images
        if w < 8 or h < 8:
            continue
        ar = w / float(h)
        if not (0.6 <= ar <= 1.4):
            continue
        candidates.append((x + w / 2.0, y + h / 2.0, w, h))
    return candidates

def robust_range(values, low_pct=3, high_pct=97):
    lo = np.percentile(values, low_pct)
    hi = np.percentile(values, high_pct)
    return lo, hi

def split_into_columns(candidates, warp_w, k=NUM_COLUMNS):
    if len(candidates) < k * 5:
        edges = np.linspace(0, warp_w, k + 1)
        return edges

    xs = sorted(c[0] for c in candidates)
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    gaps.sort(reverse=True)
    cut_positions = sorted(idx for _, idx in gaps[: k - 1])

    edges = [0.0]
    for idx in cut_positions:
        edges.append((xs[idx] + xs[idx + 1]) / 2.0)
    edges.append(float(warp_w))
    edges = sorted(set(edges))

    if len(edges) != k + 1:
        edges = np.linspace(0, warp_w, k + 1)
    return edges

def build_expected_grid(candidates, warp_w, warp_h):
    col_edges = split_into_columns(candidates, warp_w)

    radii = [max(w, h) / 2.0 for (_, _, w, h) in candidates] if candidates else []
    # Dynamic radius sizing for smaller cropped images
    avg_radius = float(np.median(radii)) if radii else (warp_w * 0.015)

    grid = np.zeros((NUM_COLUMNS, ROWS_PER_COLUMN, OPTIONS_PER_Q, 2), dtype="float32")

    for col in range(NUM_COLUMNS):
        x_lo, x_hi = col_edges[col], col_edges[col + 1]
        col_pts = [c for c in candidates if x_lo <= c[0] < x_hi]

        if len(col_pts) >= 10:
            y_lo, y_hi = robust_range([c[1] for c in col_pts])
        else:
            # Tighter margins to accommodate tightly cropped images
            y_lo, y_hi = warp_h * 0.02, warp_h * 0.98

        if len(col_pts) >= 10:
            x_lo_opt, x_hi_opt = robust_range([c[0] for c in col_pts])
        else:
            col_width = x_hi - x_lo
            x_lo_opt, x_hi_opt = x_lo + col_width * 0.15, x_hi - col_width * 0.05

        row_centers = np.linspace(y_lo, y_hi, ROWS_PER_COLUMN)
        opt_centers = np.linspace(x_lo_opt, x_hi_opt, OPTIONS_PER_Q)

        for r in range(ROWS_PER_COLUMN):
            for o in range(OPTIONS_PER_Q):
                grid[col, r, o] = (opt_centers[o], row_centers[r])

    return grid, avg_radius

# ----------------------------------------------------------------------------
# FILL CLASSIFICATION
# ----------------------------------------------------------------------------
def sample_fill_ratio(binary_ink, cx, cy, radius):
    h, w = binary_ink.shape
    cx, cy = int(round(cx)), int(round(cy))
    r = max(int(round(radius * 0.85)), 4)  

    x0, x1 = max(cx - r, 0), min(cx + r, w)
    y0, y1 = max(cy - r, 0), min(cy + r, h)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    roi = binary_ink[y0:y1, x0:x1]
    mask = np.zeros(roi.shape, dtype="uint8")
    cv2.circle(mask, (roi.shape[1] // 2, roi.shape[0] // 2), r, 255, -1)

    ink_pixels = cv2.countNonZero(cv2.bitwise_and(roi, roi, mask=mask))
    total_pixels = cv2.countNonZero(mask)
    if total_pixels == 0:
        return 0.0
    return ink_pixels / float(total_pixels)

def classify_all_answers(binary_ink, grid, radius):
    results = []
    q_num = 1
    for col in range(NUM_COLUMNS):
        for r in range(ROWS_PER_COLUMN):
            ratios = []
            for o in range(OPTIONS_PER_Q):
                cx, cy = grid[col, r, o]
                ratios.append(sample_fill_ratio(binary_ink, cx, cy, radius))

            ratios_arr = np.array(ratios)
            best_idx = int(np.argmax(ratios_arr))
            best_val = ratios_arr[best_idx]

            sorted_vals = np.sort(ratios_arr)[::-1]
            is_multi = (
                best_val >= FILL_THRESHOLD
                and (sorted_vals[0] - sorted_vals[1]) < AMBIGUOUS_MARGIN
            )

            if best_val < FILL_THRESHOLD:
                status = "unattempted"
                answer = None
            elif is_multi:
                status = "multi-mark"
                answer = None
            else:
                status = "answered"
                answer = OPTION_LABELS[best_idx]

            results.append(
                {
                    "question": q_num,
                    "answer": answer,
                    "status": status,
                    "ratios": [round(float(x), 3) for x in ratios],
                }
            )
            q_num += 1
    return results

# ----------------------------------------------------------------------------
# FULL PIPELINE
# ----------------------------------------------------------------------------
def process_sheet_image(image):
    warped, method = get_normalized_sheet(image)
    # Dynamically extract actual width and height of the user's specific image
    img_h, img_w = warped.shape[:2]
    
    binary_ink = build_binary_ink_image(warped)
    candidates = find_bubble_like_contours(binary_ink)
    
    grid, radius = build_expected_grid(candidates, img_w, img_h)
    results = classify_all_answers(binary_ink, grid, radius)

    debug_info = {
        "normalization_method": method,
        "bubble_candidates_found": len(candidates),
        "estimated_radius": round(radius, 1),
        "warped_image": warped,
        "binary_image": binary_ink,
        "grid": grid,
    }
    return results, debug_info

# ----------------------------------------------------------------------------
# SCORING
# ----------------------------------------------------------------------------
def score_sheet(student_results, key_results):
    rows = []
    correct = incorrect = unattempted = multi = 0
    total_score = 0

    for s, k in zip(student_results, key_results):
        q = s["question"]
        student_ans = s["answer"]
        key_ans = k["answer"]

        if s["status"] == "unattempted":
            outcome = "Unattempted"
            marks = MARKS_UNATTEMPTED
            unattempted += 1
        elif s["status"] == "multi-mark":
            outcome = "Multiple Marks"
            marks = MARKS_INCORRECT
            multi += 1
        elif key_ans is None:
            outcome = "Key unreadable — skipped"
            marks = 0
        elif student_ans == key_ans:
            outcome = "Correct"
            marks = MARKS_CORRECT
            correct += 1
        else:
            outcome = "Incorrect"
            marks = MARKS_INCORRECT
            incorrect += 1

        total_score += marks
        rows.append(
            {
                "Question": q,
                "Key Answer": key_ans if key_ans else "-",
                "Student Answer": student_ans if student_ans else "-",
                "Result": outcome,
                "Marks": marks,
            }
        )

    summary = {
        "total_score": total_score,
        "correct": correct,
        "incorrect": incorrect,
        "unattempted": unattempted,
        "multi_mark": multi,
        "max_possible": TOTAL_QUESTIONS * MARKS_CORRECT,
    }
    return rows, summary

# ----------------------------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------------------------
def render_upload_and_process(label, key_prefix):
    st.subheader(label)
    col1, col2 = st.columns(2)
    with col1:
        camera_file = st.camera_input("Take a photo", key=f"{key_prefix}_camera")
    with col2:
        uploaded_file = st.file_uploader(
            "...or upload an image", type=["jpg", "jpeg", "png"], key=f"{key_prefix}_upload"
        )

    source = camera_file or uploaded_file
    if source is None:
        return None, None

    image = load_image_from_upload(source)
    with st.spinner(f"Analyzing {label.lower()}..."):
        results, debug_info = process_sheet_image(image)

    with st.expander(f"Detection details — {label}"):
        st.write(
            f"Normalization method: **{debug_info['normalization_method']}** | "
            f"Bubble candidates found: **{debug_info['bubble_candidates_found']}** | "
            f"Estimated bubble radius: **{debug_info['estimated_radius']}px**"
        )
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.image(
                cv2.cvtColor(debug_info["warped_image"], cv2.COLOR_BGR2RGB),
                caption="Normalized sheet",
                use_container_width=True,
            )
        with dcol2:
            st.image(debug_info["binary_image"], caption="Ink detection mask", use_container_width=True)

    return results, debug_info

def editable_answer_table(results, key_prefix):
    df = pd.DataFrame(
        [{"Question": r["question"], "Answer": r["answer"] if r["answer"] else ""} for r in results]
    )
    edited = st.data_editor(
        df,
        key=f"{key_prefix}_editor",
        num_rows="fixed",
        use_container_width=True,
        height=300,
        column_config={
            "Answer": st.column_config.SelectboxColumn(
                "Answer", options=["", "1", "2", "3", "4"], required=False
            )
        },
    )
    corrected = []
    for i, r in enumerate(results):
        ans = edited.iloc[i]["Answer"]
        ans = ans if ans in OPTION_LABELS else None
        status = "answered" if ans else "unattempted"
        corrected.append({**r, "answer": ans, "status": status})
    return corrected

def main():
    st.set_page_config(page_title="NEET OMR Checker", page_icon="✅", layout="wide")
    st.title("✅ NEET OMR Checker")
    st.caption(
        f"Photograph the answer key and a student's OMR sheet (180 questions, "
        f"4 columns × 45 rows, 4 options each). The app scores it automatically: "
        f"+{MARKS_CORRECT} correct, {MARKS_INCORRECT} incorrect, 0 unattempted."
    )

    st.divider()
    key_results, _ = render_upload_and_process("1. Answer Key", "key")

    if key_results is not None:
        st.markdown("**Review / correct the detected answer key if needed:**")
        key_results = editable_answer_table(key_results, "key")

    st.divider()
    student_results, _ = render_upload_and_process("2. Student OMR Sheet", "student")

    if student_results is not None:
        st.markdown("**Review / correct the detected student answers if needed:**")
        student_results = editable_answer_table(student_results, "student")

    st.divider()

    if key_results is not None and student_results is not None:
        if st.button("📊 Calculate Score", type="primary"):
            rows, summary = score_sheet(student_results, key_results)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Score", f"{summary['total_score']} / {summary['max_possible']}")
            m2.metric("Correct", summary["correct"])
            m3.metric("Incorrect", summary["incorrect"])
            m4.metric("Unattempted", summary["unattempted"])
            m5.metric("Multi-marked", summary["multi_mark"])

            st.subheader("Question-by-question breakdown")
            
            result_df = pd.DataFrame(rows)

            def highlight(row):
                color = {
                    "Correct": "background-color: #d4edda",
                    "Incorrect": "background-color: #f8d7da",
                    "Unattempted": "background-color: #fff3cd",
                }.get(row["Result"], "")
                return [color] * len(row)

            st.dataframe(result_df.style.apply(highlight, axis=1), use_container_width=True, height=500)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download full report (CSV)", csv, "neet_omr_report.csv", "text/csv"
            )
    else:
        st.info("Upload or capture both the answer key and the student sheet to calculate a score.")

if __name__ == "__main__":
    main()
