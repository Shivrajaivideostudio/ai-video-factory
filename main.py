import os
import random
import json
import shutil
import tempfile
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from pathlib import Path

from jose import jwt, JWTError
from google import genai
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status, BackgroundTasks, Response
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from slugify import slugify
import edge_tts
import ffmpeg
import aiofiles

# --- Initial Setup ---

# Load environment variables from .env file
load_dotenv()

# Configure Logging for Execution Validation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AIVideoFactory")

app = FastAPI(title="AI Video Factory")

# --- Configuration & Global State ---

# Security
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY is not set in the environment. Please add it to your .env file.")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# Owner/Admin Configuration
OWNER_EMAILS = [e.strip() for e in os.getenv("OWNER_EMAILS", "").split(",") if e.strip()]
OWNER_MOBILES = [m.strip() for m in os.getenv("OWNER_MOBILES", "").split(",") if m.strip()]

# AI Clients Initialization (Corrected)
try:
    # Correct way to initialize 'google-genai' library
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Other clients remain the same
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("AI clients initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize AI clients: {e}")
    raise # Stop the app if keys are missing or invalid

# Directories
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# --- Mock Database & In-Memory State ---
user_db = {}
otp_db = {}
generation_status = {}

# Initialize Owners from .env
for email in OWNER_EMAILS:
    user_db[email] = {"status": "approved", "is_owner": True, "failed_otp": 0, "lockout_until": None}
for mob in OWNER_MOBILES:
    user_db[mob] = {"status": "approved", "is_owner": True, "failed_otp": 0, "lockout_until": None}

logger.info(f"Initialized with {len(user_db)} owner accounts.")


# --- Security & Authentication ---

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return None
    
    try:
        payload = jwt.decode(token.split(" ")[1], SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None or user_id not in user_db:
            return None
        
        user = user_db[user_id]
        if user["status"] != "approved":
            return None
            
        return {"id": user_id, **user}
    except JWTError:
        return None

async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"}
        )
    return current_user


# --- AI Engine Modules ---

class CreativeEngine:
    """Role: Gemini - The Scriptwriter (Corrected)"""
    @staticmethod
    async def generate_script(topic: str, language: str, style: str):
        logger.info(f"Generating script for topic: {topic}")
        prompt = f"""
        You are a world-class viral video scriptwriter.
        Generate a complete video production package for a YouTube Short on the topic: '{topic}'.
        The video should be in {language} and have a {style} style.
        Your response MUST be a single, valid JSON object. Do not include any text before or after the JSON.
        The JSON structure must be exactly as follows:
        {{
            "title": "A short, catchy, SEO-optimized title (under 70 characters).",
            "description": "A full YouTube description including a summary, relevant keywords, and 3-5 relevant hashtags.",
            "tags": ["list", "of", "5", "to", "10", "relevant", "tags"],
            "thumbnail_prompt": "A detailed, vivid prompt for an AI image generator to create a clickable thumbnail for this video.",
            "script": [
                {{"scene": 1, "narration": "The first sentence of the script. This should be a strong hook.", "visual_idea": "A brief description of the visual content for this scene."}},
                {{"scene": 2, "narration": "The second part of the script.", "visual_idea": "Description of the visuals for the second scene."}}
            ]
        }}
        """
        try:
            # Correct way to call the synchronous 'google-genai' library in an async function
            response_text = await asyncio.to_thread(
                gemini_client.generate_text,
                prompt=prompt,
                model="models/gemini-1.5-pro-latest" # Using full model name
            )
            json_text = response_text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(json_text)
        except Exception as e:
            logger.error(f"Error in Gemini script generation: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate script from AI.")


class NarrationEngine:
    """Role: Edge-TTS - The Voice"""
    @staticmethod
async def generate_audio(text: str, output_path: Path, voice: str = "en-US-JasonNeural"):
        logger.info(f"Generating audio for text, saving to {output_path}")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        if not output_path.exists():
            raise Exception("Failed to save TTS audio file.")


class VideoEngine:
    """Role: FFMPEG - The Director"""
    @staticmethod
    def compile_video(audio_path: Path, image_path: Path, output_path: Path, duration: float):
        logger.info(f"Compiling video: {output_path}")
        try:
            (
                ffmpeg
                .input(image_path, loop=1, t=duration, framerate=30)
                .input(audio_path)
                .output(str(output_path), vcodec='libx264', acodec='aac', pix_fmt='yuv420p', shortest=None)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"Video compiled successfully: {output_path}")
        except ffmpeg.Error as e:
            logger.error(f"FFMPEG Error: {e.stderr.decode()}")
            raise


# --- Background Task for Video Generation ---

async def background_video_generation(task_id: str, topic: str, language: str, style: str):
    generation_status[task_id] = {"status": "processing", "progress": 10, "message": "Generating script..."}
    
    base_filename = slugify(f"{topic}-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        script_data = await CreativeEngine.generate_script(topic, language, style)
        full_narration = " ".join([scene["narration"] for scene in script_data["script"]])
        
        generation_status[task_id].update({"progress": 30, "message": "Generating audio narration..."})

        audio_path = temp_dir / f"{base_filename}.mp3"
        await NarrationEngine.generate_audio(full_narration, audio_path)
        
        probe = ffmpeg.probe(str(audio_path))
        audio_duration = float(probe['format']['duration'])

        generation_status[task_id].update({"progress": 60, "message": "Generating visuals (using placeholder)..."})

        placeholder_image = BASE_DIR / "static/placeholder.png"
        if not placeholder_image.exists():
             raise FileNotFoundError("Please create a 'static/placeholder.png' image file.")
        visual_path = temp_dir / "background.png"
        shutil.copy(placeholder_image, visual_path)

        generation_status[task_id].update({"progress": 80, "message": "Compiling final video..."})
        
        final_video_path = OUTPUT_DIR / f"{base_filename}.mp4"
        VideoEngine.compile_video(audio_path, visual_path, final_video_path, audio_duration)

        final_url = f"/static/outputs/{base_filename}.mp4"
        generation_status[task_id] = {"status": "completed", "progress": 100, "url": final_url, "title": script_data.get("title")}

    except Exception as e:
        logger.error(f"Error in background task {task_id}: {e}")
        generation_status[task_id] = {"status": "failed", "progress": 100, "message": str(e)}
    finally:
        shutil.rmtree(temp_dir)


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if await get_current_user(request):
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    
    whatsapp_link = f"https://wa.me/{OWNER_MOBILES[0]}?text=I'd%20like%20to%20request%20access%20to%20the%20AI%20Video%20Factory." if OWNER_MOBILES else "#"
    return templates.TemplateResponse("landing.html", {"request": request, "whatsapp_link": whatsapp_link})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/api/send-otp")
async def send_otp(identifier: str = Form(...)):
    if identifier not in user_db:
        raise HTTPException(status_code=404, detail="Identifier not found or not registered.")
    
    user = user_db[identifier]
    if user.get("lockout_until") and user["lockout_until"] > datetime.now(timezone.utc):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Please try again later.")

    otp = str(random.randint(100000, 999999))
    otp_db[identifier] = {"otp": otp, "expires": datetime.now(timezone.utc) + timedelta(minutes=5)}
    
    logger.info(f"Generated OTP for {identifier}: {otp}")
    
    return {"message": f"OTP sent to {identifier}. For testing, the OTP is {otp}."}

@app.post("/api/verify-otp")
async def verify_otp(response: Response, identifier: str = Form(...), otp: str = Form(...)):
    if identifier not in otp_db or identifier not in user_db:
        raise HTTPException(status_code=400, detail="Invalid request.")

    user = user_db[identifier]
    if user.get("lockout_until") and user["lockout_until"] > datetime.now(timezone.utc):
        raise HTTPException(status_code=429, detail="Account locked. Try again later.")

    otp_info = otp_db[identifier]
    if otp_info["expires"] < datetime.now(timezone.utc) or otp_info["otp"] != otp:
        user["failed_otp"] = user.get("failed_otp", 0) + 1
        if user["failed_otp"] >= 5:
            user["lockout_until"] = datetime.now(timezone.utc) + timedelta(minutes=15)
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")

    user["failed_otp"] = 0
    user["lockout_until"] = None
    del otp_db[identifier]
    
    access_token = create_access_token(data={"sub": identifier})
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True, samesite="lax")
    return {"message": "Login successful."}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: dict = Depends(get_current_active_user)):
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})

@app.get("/create-video", response_class=HTMLResponse)
async def create_video_page(request: Request, user: dict = Depends(get_current_active_user)):
    return templates.TemplateResponse("create-video.html", {"request": request, "user": user})

@app.post("/api/generate-video")
async def generate_video_endpoint(
    background_tasks: BackgroundTasks,
    topic: str = Form(...),
    language: str = Form("en-US"),
    style: str = Form("informative"),
    user: dict = Depends(get_current_active_user)
):
    task_id = f"{user['id']}-{slugify(topic)}-{int(datetime.now().timestamp())}"
    generation_status[task_id] = {"status": "starting", "progress": 0, "message": "Task initiated."}
    
    background_tasks.add_task(background_video_generation, task_id, topic, language, style)
    
    return {"message": "Video generation started in the background.", "task_id": task_id}

@app.get("/api/generation-status/{task_id}")
async def get_status(task_id: str, user: dict = Depends(get_current_active_user)):
    if task_id not in generation_status:
        raise HTTPException(status_code=404, detail="Task not found.")
    if not task_id.startswith(user['id']):
        raise HTTPException(status_code=403, detail="Forbidden.")
    return JSONResponse(content=generation_status[task_id])

@app.get("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return RedirectResponse(url="/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
