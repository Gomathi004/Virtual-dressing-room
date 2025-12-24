# main.py
import io
import base64
import os
from typing import Optional, List, Dict

import requests
import numpy as np
import cv2
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageOps

from shopify_fetch import get_clothes  # get_clothes(gender, kind) -> list[str]

from db import Selection, init_db, get_session
from typing import List
from sqlmodel import select


ROBOFLOW_API_KEY = "fb8FDC2lnqTjyHhWeQF2"
ROBOFLOW_MODEL_URL = "https://serverless.roboflow.com/gender-classification-wadex/1"

app = FastAPI()
init_db()


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

CLOTHES_PATH = "clothes/tshirt1.png"


def read_imagefile(uploaded_file: UploadFile) -> np.ndarray:
    image_bytes = uploaded_file.file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img  # BGR


def download_cloth_to_png(url: str, out_path: str) -> str:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    img_bytes = io.BytesIO(resp.content)

    cloth = Image.open(img_bytes).convert("RGBA")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cloth.save(out_path, format="PNG")
    return out_path


def detect_gender_with_roboflow(img_bgr: np.ndarray) -> Optional[str]:
    try:
        success, encoded_img = cv2.imencode(".jpg", img_bgr)
        if not success:
            return None

        files = {"file": ("image.jpg", encoded_img.tobytes(), "image/jpeg")}
        params = {"api_key": ROBOFLOW_API_KEY}

        resp = requests.post(ROBOFLOW_MODEL_URL, files=files, params=params, timeout=15)
        data = resp.json()
        print("Roboflow response:", data)

        if "predictions" in data and data["predictions"]:
            top_pred = data["predictions"][0]
            class_name = top_pred.get("class", "").lower()
            conf = float(top_pred.get("confidence", 0.0))

            if "male" in class_name or "men" in class_name:
                return "men"
            if "female" in class_name or "woman" in class_name:
                return "women"

        return None
    except Exception as e:
        print("Roboflow error:", e)
        return None



def overlay_top_on_image(
    pil_img: Image.Image,
    img_rgb: np.ndarray,
    cloth_png_path: str,
    gender: str = "men",
) -> Image.Image:
    h, w = img_rgb.shape[:2]
    results = pose.process(img_rgb)
    if not results.pose_landmarks:
        return pil_img

    lm = results.pose_landmarks.landmark
    Ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    Rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    Lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
    Rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]

    Lsx, Lsy = int(Ls.x * w), int(Ls.y * h)
    Rsx, Rsy = int(Rs.x * w), int(Rs.y * h)
    Lhy, Rhy = int(Lh.y * h), int(Rh.y * h)

    shoulder_width = float(np.hypot(Rsx - Lsx, Rsy - Lsy))
    hip_y = int((Lhy + Rhy) / 2)
    shoulder_y = int((Lsy + Rsy) / 2)
    torso_height = hip_y - shoulder_y

    if gender == "women":
        width_factor = 2.0
        height_factor = 2.2
    else:
        width_factor = 2.0
        height_factor = 1.2

    tshirt_width = int(shoulder_width * width_factor)
    tshirt_height = int(torso_height * height_factor)

    cloth = Image.open(cloth_png_path).convert("RGBA")
    cloth = ImageOps.crop(cloth, border=2)

    orig_w, orig_h = cloth.size
    scale = tshirt_width / orig_w
    new_h = int(orig_h * scale)
    cloth_resized = cloth.resize((tshirt_width, new_h), Image.Resampling.LANCZOS)

    if new_h > tshirt_height:
        cloth_resized = cloth_resized.crop((0, 0, tshirt_width, tshirt_height))

    center_x = int((Lsx + Rsx) / 2)
    paste_x = int(center_x - tshirt_width / 2)
    paste_y = int(shoulder_y - tshirt_height * 0.15)

    base = pil_img.copy()
    base.paste(cloth_resized, (paste_x, paste_y), mask=cloth_resized)
    return base


def overlay_bottom_on_image(
    pil_img: Image.Image,
    img_rgb: np.ndarray,
    cloth_png_path: str,
    gender: str = "men",
) -> Image.Image:
    h, w = img_rgb.shape[:2]
    results = pose.process(img_rgb)
    if not results.pose_landmarks:
        return pil_img

    lm = results.pose_landmarks.landmark
    Lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
    Rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
    Lk = lm[mp_pose.PoseLandmark.LEFT_KNEE]
    Rk = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
    La = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
    Ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

    Lhx, Lhy = int(Lh.x * w), int(Lh.y * h)
    Rhx, Rhy = int(Rh.x * w), int(Rh.y * h)
    Lky, Rky = int(Lk.y * h), int(Rk.y * h)
    Lay, Ray = int(La.y * h), int(Ra.y * h)

    hip_y = int((Lhy + Rhy) / 2)
    knee_y = int((Lky + Rky) / 2)
    ankle_y = int((Lay + Ray) / 2)

    knee_width = float(abs(Rhx - Lhx)) * 1.1
    if gender == "men":
        width_factor = 4.2
    else:
        width_factor = 3.2

    pant_width_target = int(knee_width * width_factor)

    waistband_y = max(0, int(hip_y - 0.08 * h))
    hem_y = min(h, int(ankle_y + 0.03 * h))
    pant_height = max(20, hem_y - waistband_y)

    cloth = Image.open(cloth_png_path).convert("RGBA")
    cloth = ImageOps.crop(cloth, border=2)

    orig_w, orig_h = cloth.size

    scale_h = pant_height / orig_h
    new_w = int(orig_w * scale_h)
    new_h = pant_height
    pant_resized = cloth.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if new_w > pant_width_target:
        new_w = pant_width_target
        pant_resized = pant_resized.resize((new_w, pant_height), Image.Resampling.LANCZOS)

    center_x = int((Lhx + Rhx) / 2)
    paste_x = int(center_x - new_w / 2)
    paste_y = waistband_y

    base = pil_img.copy()
    base.paste(pant_resized, (paste_x, paste_y), mask=pant_resized)
    return base


def overlay_outfit_on_image(
    bg_bgr: np.ndarray,
    top_png_path: Optional[str],
    bottom_png_path: Optional[str],
    gender: str = "men",
) -> np.ndarray:
    img_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")

    if bottom_png_path:
        pil_img = overlay_bottom_on_image(pil_img, img_rgb, bottom_png_path, gender)
    if top_png_path:
        pil_img = overlay_top_on_image(pil_img, img_rgb, top_png_path, gender)

    result_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
    return result_bgr

def has_person_pose(img_bgr: np.ndarray) -> bool:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    # just check if MediaPipe sees *any* pose at all
    return results.pose_landmarks is not None



@app.post("/tryon")
async def tryon_api(
    file: UploadFile = File(...),
    top_url: Optional[str] = Form(None),
    bottom_url: Optional[str] = Form(None),
    gender: Optional[str] = Form("men"),
):
    try:
        img = read_imagefile(file)
    except Exception as e:
        return JSONResponse(
            {"error": "Could not read image", "details": str(e)},
            status_code=400,
        )

    if not top_url and not bottom_url:
        return JSONResponse(
            {"error": "No cloth selected yet (top or bottom required)"},
            status_code=400,
        )

    top_path = bottom_path = None
    try:
        if top_url:
            top_path = download_cloth_to_png(top_url, "clothes/_tmp_top.png")
        if bottom_url:
            bottom_path = download_cloth_to_png(bottom_url, "clothes/_tmp_bottom.png")
    except Exception as e:
        return JSONResponse(
            {"error": "Could not load cloth image", "details": str(e)},
            status_code=400,
        )

    try:
        out = overlay_outfit_on_image(img, top_path, bottom_path, gender)
    except Exception as e:
        return JSONResponse(
            {"error": "Processing failed", "details": str(e)},
            status_code=500,
        )

    # <<< INSERT DB LOGGING HERE >>>
    from db import get_session, Selection  # or keep these imports at the top of file
    with get_session() as session:
        sel = Selection(
            gender=gender,
            top_url=top_url,
            bottom_url=bottom_url,
        )
        session.add(sel)
        session.commit()
    # <<< END INSERT >>>

    _, im_buf_arr = cv2.imencode(".png", out)
    byte_im = im_buf_arr.tobytes()
    b64 = base64.b64encode(byte_im).decode("utf-8")
    return {"image_base64": b64}


@app.get("/clothes")
async def clothes_api(gender: str = Query("men")) -> List[Dict[str, str]]:
    """
    Return both tops and bottoms for this gender.
    Each item: {"url": "...", "type": "top" or "bottom"}
    """
    tops = get_clothes(gender, "top")
    bottoms = get_clothes(gender, "bottom")

    items: List[Dict[str, str]] = []
    for url in tops:
        items.append({"url": url, "type": "top"})
    for url in bottoms:
        items.append({"url": url, "type": "bottom"})
    return items


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



@app.post("/classify-gender")
async def classify_gender(file: UploadFile = File(...)):
    try:
        img = read_imagefile(file)
    except Exception as e:
        return JSONResponse(
            {"error": "Could not read image", "details": str(e)},
            status_code=400,
        )

    # 1) quick person check
    if not has_person_pose(img):
        return JSONResponse(
            {"error": "no_person", "message": "No person detected in image."},
            status_code=400,
        )
    
    # 2) Call Roboflow
    auto_gender = detect_gender_with_roboflow(img)

    # If Roboflow is unsure, default to 'men' (or 'women' if you prefer)
    if auto_gender not in ["men", "women"]:
        auto_gender = "men"

    return {"gender": auto_gender}

@app.get("/selections", response_model=List[Selection])
def list_selections():
    with get_session() as session:
        result = session.exec(select(Selection).order_by(Selection.created_at.desc()))
        return result.all()
