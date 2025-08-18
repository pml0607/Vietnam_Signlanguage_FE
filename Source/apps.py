from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import json
import asyncio
import uuid, os

app = FastAPI(title="Sign Language Recognition API", version="1.0.0")

VIDEO_DIR = Path("/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/data/rgb")
RESULT_DIR = Path("/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/results")

# Fixed: VIDEO_DIR instead of undefined UPLOAD_DIR
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Thay bằng domain app React Native nếu muốn an toàn hơn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Sign Language Recognition API", "status": "running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "sign-language-api"}

@app.get("/debug/endpoints")
async def list_endpoints():
    """Debug endpoint to list all available routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else ["WebSocket"],
                "name": getattr(route, 'name', 'Unknown')
            })
    return {"endpoints": routes}

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    try:
        print(f"[UPLOAD] Received file: {file.filename}, content_type: {file.content_type}")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            print(f"[UPLOAD] Invalid content type: {file.content_type}")
            return JSONResponse(
                status_code=400,
                content={"error": "File must be a video", "received_type": file.content_type}
            )
        
        # Generate unique ID
        video_id = str(uuid.uuid4())
        
        # Save as .avi (as expected by landmark watcher)
        save_path = VIDEO_DIR / f"{video_id}.avi"

        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        print(f"[UPLOAD] Saved video to {save_path}")
        print(f"[UPLOAD] File size: {len(content)} bytes")
        print(f"[UPLOAD] Video ID: {video_id}")
        
        return {"status": "received", "video_id": video_id, "file_size": len(content)}
        
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Upload failed: {str(e)}"}
        )

@app.websocket("/ws/{video_id}")
async def websocket_endpoint(websocket: WebSocket, video_id: str):
    await websocket.accept()
    result_path = RESULT_DIR / f"{video_id}.json"
    
    print(f"[WS] WebSocket connected for video_id: {video_id}")
    print(f"[WS] Looking for result at: {result_path}")
    
    try:
        # Wait for result file with timeout
        for i in range(120):  # Wait max 120 seconds (2 minutes)
            if result_path.exists():
                try:
                    with result_path.open() as f:
                        result = json.load(f)
                    
                    print(f"[WS] Found result for {video_id}: {result}")
                    await websocket.send_json({
                        "status": "done",
                        "video_id": video_id,
                        "result": result
                    })
                    print(f"[WS] Result sent for {video_id}")
                    break
                    
                except json.JSONDecodeError as e:
                    print(f"[WS] Invalid JSON in result file: {e}")
                    await websocket.send_json({
                        "status": "error", 
                        "video_id": video_id,
                        "error": "Invalid result format"
                    })
                    break
                    
            else:
                # Send processing status every 5 seconds
                if i % 5 == 0:
                    await websocket.send_json({
                        "status": "processing",
                        "video_id": video_id,
                        "progress": min(90, i * 2)  # Fake progress
                    })
                    print(f"[WS] Processing update {i}/120 for {video_id}")
                
                await asyncio.sleep(1)
        else:
            # Timeout reached
            await websocket.send_json({
                "status": "timeout",
                "video_id": video_id
            })
            print(f"[WS] Timeout for {video_id}")
            
    except WebSocketDisconnect:
        print(f"[WS] Client disconnected for {video_id}")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await websocket.send_json({
                "status": "error",
                "video_id": video_id, 
                "error": str(e)
            })
        except:
            pass

# Test WebSocket endpoint for debugging
@app.get("/test-ws/{video_id}")
async def test_websocket_endpoint(video_id: str):
    """Test endpoint to verify WebSocket route is registered"""
    return {
        "message": f"WebSocket endpoint exists for video_id: {video_id}",
        "websocket_url": f"ws://your-domain/ws/{video_id}",
        "video_id": video_id
    }

if __name__ == "__main__":
    import uvicorn
    print("[STARTUP] Starting Sign Language Recognition API...")
    print(f"[STARTUP] Video directory: {VIDEO_DIR}")
    print(f"[STARTUP] Result directory: {RESULT_DIR}")
    print("[STARTUP] Available endpoints:")
    print("  GET  /")
    print("  GET  /health") 
    print("  GET  /debug/endpoints")
    print("  POST /upload/")
    print("  WS   /ws/{video_id}")
    print("  GET  /test-ws/{video_id}")
    uvicorn.run(app, host="0.0.0.0", port=8000)