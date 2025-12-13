# main.py
import io
import base64
import os
from typing import Optional

import requests
import numpy as np
import cv2
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageOps

from shopify_fetch import get_clothes

ROBOFLOW_API_KEY = "fb8FDC2lnqTjyHhWeQF2"
ROBOFLOW_MODEL_URL = "https://serverless.roboflow.com/gender-classification-wadex/1"  # replace with actual URL


app = FastAPI()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Default cloth path (local transparent PNG)
CLOTHES_PATH = "clothes/tshirt1.png"


def read_imagefile(uploaded_file: UploadFile) -> np.ndarray:
    image_bytes = uploaded_file.file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img  # BGR


def download_cloth_to_png(url: str, out_path: str) -> str:
    """
    Download cloth image from Shopify URL and save as PNG with alpha preserved.
    Uses PIL instead of cv2 to avoid losing transparency.
    """
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    img_bytes = io.BytesIO(resp.content)

    cloth = Image.open(img_bytes).convert("RGBA")
    cloth.save(out_path, format="PNG")
    return out_path

def detect_gender_with_roboflow(img_bgr: np.ndarray) -> Optional[str]:
    """
    Send the uploaded image to Roboflow and return 'men' or 'women'
    based on the model's prediction. Returns None on failure.
    """
    try:
        # Encode image as JPEG in memory
        success, encoded_img = cv2.imencode(".jpg", img_bgr)
        if not success:
            print("Roboflow: could not encode image")
            return None

        files = {
            "file": ("image.jpg", encoded_img.tobytes(), "image/jpeg")
        }
        params = {
            "api_key": ROBOFLOW_API_KEY
        }

        resp = requests.post(ROBOFLOW_MODEL_URL, files=files, params=params, timeout=15)
        data = resp.json()
        print("Roboflow response:", data)

        # Adapt this parsing to Roboflow's exact JSON format
        # Example: for a classification model with 'predictions'
        if "predictions" in data and data["predictions"]:
            top_pred = data["predictions"][0]
            class_name = top_pred.get("class", "").lower()

            if "male" in class_name or "men" in class_name:
                return "men"
            if "female" in class_name or "woman" in class_name:
                return "women"

        return None
    except Exception as e:
        print("Roboflow error:", e)
        return None



def overlay_cloth_on_image(bg_bgr: np.ndarray, cloth_png_path: str, gender: str = "men") -> np.ndarray:

    # Convert background to RGBA
    img_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")

    results = pose.process(img_rgb)
    if not results.pose_landmarks:
        return bg_bgr

    h, w = img_rgb.shape[:2]
    lm = results.pose_landmarks.landmark

    # Key points
    Ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    Rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    Lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
    Rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]

    Lsx, Lsy = int(Ls.x * w), int(Ls.y * h)
    Rsx, Rsy = int(Rs.x * w), int(Rs.y * h)
    Lhx, Lhy = int(Lh.x * w), int(Lh.y * h)
    Rhx, Rhy = int(Rh.x * w), int(Rh.y * h)

       # Distances
    shoulder_width = float(np.hypot(Rsx - Lsx, Rsy - Lsy))
    hip_y = int((Lhy + Rhy) / 2)
    shoulder_y = int((Lsy + Rsy) / 2)
    torso_height = hip_y - shoulder_y

    # Target shirt/dress size
    if gender == "women":
        width_factor = 2.0     # similar width
        height_factor = 2.2    # extend down toward knees
    else:  # men
        width_factor = 2.0
        height_factor = 1.2    # just torso length

    tshirt_width = int(shoulder_width * width_factor)
    tshirt_height = int(torso_height * height_factor)


    # Load cloth WITH alpha
    cloth = Image.open(cloth_png_path).convert("RGBA")
    cloth = ImageOps.crop(cloth, border=2)

    # Resize while keeping aspect ratio, based on width
    orig_w, orig_h = cloth.size
    scale = tshirt_width / orig_w
    new_h = int(orig_h * scale)
    cloth_resized = cloth.resize((tshirt_width, new_h), Image.Resampling.LANCZOS)

    # If too tall, crop bottom
    if new_h > tshirt_height:
        cloth_resized = cloth_resized.crop((0, 0, tshirt_width, tshirt_height))

    # Position: top of shirt just above shoulders
    center_x = int((Lsx + Rsx) / 2)
    paste_x = int(center_x - tshirt_width / 2)
    paste_y = int(shoulder_y - tshirt_height * 0.15)  # small negative to cover collar

    base = pil_img.copy()
    base.paste(cloth_resized, (paste_x, paste_y), mask=cloth_resized)

    result_bgr = cv2.cvtColor(np.array(base), cv2.COLOR_RGBA2BGR)
    return result_bgr



@app.post("/tryon")
async def tryon_api(
    file: UploadFile = File(...),
    cloth_url: Optional[str] = Form(None),
    gender: Optional[str] = Form("men"),
):
    # 1) Read user image
    try:
        img = read_imagefile(file)
    except Exception as e:
        return JSONResponse(
            {"error": "Could not read image", "details": str(e)},
            status_code=400,
        )

    # 2) Require cloth_url (user selects cloth after classify)
    if not cloth_url:
        return JSONResponse(
            {"error": "No cloth selected yet"},
            status_code=400,
        )

    # 3) Download cloth
    try:
        tmp_path = "clothes/_tmp_from_shopify.png"
        cloth_path = download_cloth_to_png(cloth_url, tmp_path)
    except Exception as e:
        return JSONResponse(
            {"error": "Could not load cloth image", "details": str(e)},
            status_code=400,
        )

    # 4) Overlay
    try:
        out = overlay_cloth_on_image(img, cloth_path, gender)

    except Exception as e:
        return JSONResponse(
            {"error": "Processing failed", "details": str(e)},
            status_code=500,
        )

    _, im_buf_arr = cv2.imencode(".png", out)
    byte_im = im_buf_arr.tobytes()
    b64 = base64.b64encode(byte_im).decode("utf-8")
    return {"image_base64": b64}




@app.get("/clothes-list")
async def clothes_list():
    files = os.listdir("clothes")
    pngs = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    return JSONResponse(pngs)


@app.get("/clothes/{filename}")
async def clothes_file(filename: str):
    return FileResponse(f"clothes/{filename}")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.get("/clothes")
async def clothes_api(gender: str = Query("men")):
    # gender will be "men" or "women"
    return get_clothes(gender)


from fastapi import FastAPI, File, UploadFile, Form, Query

@app.post("/classify-gender")
async def classify_gender(file: UploadFile = File(...)):
    # Read user image
    try:
        img = read_imagefile(file)
    except Exception as e:
        return JSONResponse(
            {"error": "Could not read image", "details": str(e)},
            status_code=400,
        )

    # Call Roboflow
    auto_gender = detect_gender_with_roboflow(img)
    if auto_gender not in ["men", "women"]:
        # default / unknown
        auto_gender = "men"

    print("Classified gender:", auto_gender)
    return {"gender": auto_gender}
