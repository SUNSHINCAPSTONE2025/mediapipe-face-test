import os, time, math
from typing import Optional, Dict, List, Tuple
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import mediapipe as mp

SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
# 정적 파일(스냅샷/분석영상) 서빙
app.mount("/static", StaticFiles(directory=SNAP_DIR), name="static")

mp_face_mesh = mp.solutions.face_mesh
# 정적 이미지(스냅샷)용
FACE_MESH_IMG = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
# 동영상 프레임용
FACE_MESH_VIDEO = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 랜드마크 인덱스
LEFT_EYE  = [33,160,158,133,153,144]
RIGHT_EYE = [263,387,385,362,380,373]
LEFT_EYE_CORNERS  = (33, 133)
RIGHT_EYE_CORNERS = (263, 362)
LEFT_EYE_LIDS     = (159, 145)   # 위/아래 대표
RIGHT_EYE_LIDS    = (386, 374)

# iris(홍채) 대표 인덱스
LEFT_IRIS_IDXS  = [474, 475, 476, 477]
RIGHT_IRIS_IDXS = [469, 470, 471, 472]

MOUTH_LEFT_CORNER  = 61
MOUTH_RIGHT_CORNER = 291
MOUTH_UPPER_INNER  = 13
MOUTH_LOWER_INNER  = 14
NOSE_TIP = 1

# 임계치/파라미터
MOUTH_DELTA = 0.02
GAZE_OFF_ABS = 0.12        # head-only
EYE_OFF_ABS  = 0.35        # eye-only (정규화 오프셋)
BLINK_RATIO = 0.75
BLINK_MIN_DUR = 0.08
DEFAULT_BLINK_LIMIT = 30
EMA_ALPHA = 0.25

# 세션 상태 (yaw, pitch, ear, mouth, eye_hor, eye_ver) 수집
CALIB_BUFFER: Dict[str, List[Tuple[float,float,float,float,float,float]]] = {}
SESSION_BASELINE: Dict[str, Dict[str, float]] = {}


# 서프라이즈 스냅샷 mouth 값들 저장
SURPRISE_BUFFER: Dict[str, List[Dict]] = {}   # {session_id: [{"path": str, "mouth": float, "ts": str}, ...]}

# 유틸
def norm_pt(lm, w, h):
    return np.array([lm.x * w, lm.y * h, lm.z], dtype=np.float32)

def head_pose_proxy(landmarks, w, h):
    nose = norm_pt(landmarks[NOSE_TIP], w, h)
    cx, cy = w/2.0, h/2.0
    return float((nose[0]-cx)/w), float((nose[1]-cy)/h)

def ear_from_landmarks(landmarks, w, h, idxs):
    pts = [norm_pt(landmarks[i], w, h) for i in idxs]
    p1, p2, p3, p4, p5, p6 = pts
    dv1 = np.linalg.norm(p2[:2] - p6[:2])
    dv2 = np.linalg.norm(p3[:2] - p5[:2])
    dh  = np.linalg.norm(p1[:2] - p4[:2])
    if dh == 0: return 0.0
    return float((dv1 + dv2) / (2.0*dh))

def mouth_corners_relative(landmarks, w, h):
    left  = norm_pt(landmarks[MOUTH_LEFT_CORNER],  w, h)
    right = norm_pt(landmarks[MOUTH_RIGHT_CORNER], w, h)
    up_in = norm_pt(landmarks[MOUTH_UPPER_INNER],  w, h)
    lo_in = norm_pt(landmarks[MOUTH_LOWER_INNER],  w, h)
    center = (up_in + lo_in) / 2.0
    rel_left  = (left[1]  - center[1]) / h
    rel_right = (right[1] - center[1]) / h
    return float((rel_left + rel_right) / 2.0)

def iris_centers_from_landmarks(lms, w, h):
    L = [norm_pt(lms[i], w, h) for i in LEFT_IRIS_IDXS  if i < len(lms)]
    R = [norm_pt(lms[i], w, h) for i in RIGHT_IRIS_IDXS if i < len(lms)]
    lc = np.mean(np.array(L), axis=0) if len(L) >= 3 else None
    rc = np.mean(np.array(R), axis=0) if len(R) >= 3 else None
    return lc, rc

def eye_local_axes(lms, w, h, corners, lids):
    c_out = norm_pt(lms[corners[0]], w, h)
    c_in  = norm_pt(lms[corners[1]], w, h)
    e_center = (c_out + c_in) / 2.0
    ex = c_in[:2] - c_out[:2]    # 수평 축
    exn = ex / (np.linalg.norm(ex) + 1e-6)
    lid_up  = norm_pt(lms[lids[0]], w, h)
    lid_dn  = norm_pt(lms[lids[1]], w, h)
    ey = lid_dn[:2] - lid_up[:2] # 수직 축
    eyn = ey / (np.linalg.norm(ey) + 1e-6)
    w_eye = np.linalg.norm(ex)   # 가로 크기
    h_eye = np.linalg.norm(ey)   # 세로 크기
    return e_center, exn, eyn, w_eye, h_eye

def eye_gaze_offset(lms, w, h, iris_center, corners, lids):
    if iris_center is None:
        return None
    e_center, ex, ey, w_eye, h_eye = eye_local_axes(lms, w, h, corners, lids)
    d = iris_center[:2] - e_center[:2]
    hor = float(np.dot(d, ex) / (w_eye + 1e-6))
    ver = float(np.dot(d, ey) / (h_eye + 1e-6))
    return hor, ver

def ema_update(prev, new, alpha=EMA_ALPHA):
    if prev is None: return new
    return (1 - alpha) * prev + alpha * new

def bbox_from_indices(lms, w, h, idxs, pad_ratio=0.15):
    pts = []
    for i in idxs:
        if i < len(lms):
            p = norm_pt(lms[i], w, h)
            pts.append(p[:2])
    if not pts:
        return None
    pts = np.stack(pts, axis=0)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    pw = (x2 - x1) * pad_ratio
    ph = (y2 - y1) * pad_ratio
    x1 = max(0, int(x1 - pw)); y1 = max(0, int(y1 - ph))
    x2 = int(min(w-1, x2 + pw)); y2 = int(min(h-1, y2 + ph))
    return (x1, y1, x2, y2)

# JSON 안전화 유틸
def _safe_num(x):
    try:
        x = float(x)
    except Exception:
        return None
    return x if math.isfinite(x) else None

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (int, float, np.floating)):
        return _safe_num(obj)
    return obj

# 시선 지표 계산 (MAE/SD/BCEA/S2S)
def compute_fixation_metrics(xs: List[float], ys: List[float], target=(0.0, 0.0)):
    if not xs or not ys or len(xs) != len(ys):
        return {"MAE": None, "SDx": None, "SDy": None, "rho": None, "BCEA": None, "S2S": None}
    X = np.array(xs, dtype=np.float32)
    Y = np.array(ys, dtype=np.float32)
    tx, ty = target
    MAE = float(np.mean(np.sqrt((X - tx)**2 + (Y - ty)**2)))
    SDx = float(np.std(X, ddof=0))
    SDy = float(np.std(Y, ddof=0))
    if len(X) > 1:
        rho = float(np.corrcoef(X, Y)[0, 1])
        if np.isnan(rho): rho = 0.0
    else:
        rho = 0.0
    k = 1.14
    base = 1.0 - rho**2
    base = max(0.0, base)
    BCEA = float(2 * k * SDx * SDy * math.sqrt(base))
    if len(X) > 1:
        diffs = np.sqrt(np.diff(X)**2 + np.diff(Y)**2)
        S2S = float(np.sqrt(np.mean(diffs**2)))
    else:
        S2S = None
    out = {"MAE": MAE, "SDx": SDx, "SDy": SDy, "rho": rho, "BCEA": BCEA, "S2S": S2S}
    return {k: _safe_num(v) for k, v in out.items()}

def _fmt4(x): return f"{x:.4f}" if (x is not None) else "n/a"
def _fmt6(x): return f"{x:.6f}" if (x is not None) else "n/a"

# 피처 추출 + (오버레이용) 포인트
def features_from_bgr(bgr, mesh):
    """
    반환:
      feats: (yaw, pitch, ear, mouth, eye_h, eye_v)  or None
      draw:  {...}  (오버레이용)
      frame: 처리/리사이즈된 프레임(BGR)
    """
    if bgr is None: return None, None
    h0, w0 = bgr.shape[:2]
    if w0 > 640:
        scale = 640.0 / w0
        bgr = cv2.resize(bgr, (640, int(h0*scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None, None
    lms = res.multi_face_landmarks[0].landmark
    h, w = bgr.shape[:2]

    yaw, pitch = head_pose_proxy(lms, w, h)
    ear = (ear_from_landmarks(lms, w, h, LEFT_EYE) + ear_from_landmarks(lms, w, h, RIGHT_EYE)) / 2.0
    mouth = mouth_corners_relative(lms, w, h)

    lc, rc = iris_centers_from_landmarks(lms, w, h)
    l_off = eye_gaze_offset(lms, w, h, lc, LEFT_EYE_CORNERS, LEFT_EYE_LIDS)  if lc is not None else None
    r_off = eye_gaze_offset(lms, w, h, rc, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS) if rc is not None else None
    eye_h = None
    eye_v = None
    if (l_off is not None) or (r_off is not None):
        xs = [x[0] for x in [l_off, r_off] if x is not None]
        ys = [x[1] for x in [l_off, r_off] if x is not None]
        if xs: eye_h = float(np.mean(xs))
        if ys: eye_v = float(np.mean(ys))

    draw = {}
    draw['nose'] = tuple(norm_pt(lms[NOSE_TIP], w, h)[:2].astype(int))

    ml = norm_pt(lms[MOUTH_LEFT_CORNER],  w, h)
    mr = norm_pt(lms[MOUTH_RIGHT_CORNER], w, h)
    mu = norm_pt(lms[MOUTH_UPPER_INNER],  w, h)
    md = norm_pt(lms[MOUTH_LOWER_INNER],  w, h)
    draw['mouth_pts'] = [tuple(ml[:2].astype(int)), tuple(mr[:2].astype(int)), tuple(mu[:2].astype(int)), tuple(md[:2].astype(int))]

    LEFT_EYE_ALL = list(set(LEFT_EYE + [LEFT_EYE_CORNERS[0], LEFT_EYE_CORNERS[1], LEFT_EYE_LIDS[0], LEFT_EYE_LIDS[1]]))
    RIGHT_EYE_ALL = list(set(RIGHT_EYE + [RIGHT_EYE_CORNERS[0], RIGHT_EYE_CORNERS[1], RIGHT_EYE_LIDS[0], RIGHT_EYE_LIDS[1]]))

    l_eye_box  = bbox_from_indices(lms, w, h, LEFT_EYE_ALL, pad_ratio=0.20)
    r_eye_box  = bbox_from_indices(lms, w, h, RIGHT_EYE_ALL, pad_ratio=0.20)
    l_iris_box = bbox_from_indices(lms, w, h, LEFT_IRIS_IDXS, pad_ratio=0.10)
    r_iris_box = bbox_from_indices(lms, w, h, RIGHT_IRIS_IDXS, pad_ratio=0.10)

    draw['l_eye_box']  = l_eye_box
    draw['r_eye_box']  = r_eye_box
    draw['l_iris_box'] = l_iris_box
    draw['r_iris_box'] = r_iris_box

    return (yaw, pitch, ear, mouth, eye_h, eye_v), draw, bgr

# 스냅샷
@app.post("/upload_snapshot")
async def upload_snapshot(
    image: UploadFile = File(...),
    session_id: str = Form(...),
    phase: str = Form("surprise")
):
    try:
        data = await image.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr0 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr0 is None:
            return JSONResponse(status_code=400, content={"detail":"이미지 디코드 실패"})

        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"{phase}_{session_id}_{ts}.jpg"
        save_path = os.path.join(SNAP_DIR, fname)
        cv2.imwrite(save_path, bgr0)

        feats, _, _ = features_from_bgr(bgr0, FACE_MESH_IMG)
        if feats is None:
            return {"saved_path": save_path.replace("\\","/"),
                    "phase": phase, "message": "얼굴 인식 실패"}

        yaw, pitch, ear, mouth, eye_h, eye_v = feats

        if phase == "calib":
            CALIB_BUFFER.setdefault(session_id, []).append((yaw, pitch, ear, mouth, eye_h or 0.0, eye_v or 0.0))
            return {"saved_path": save_path.replace("\\","/"),
                    "phase": phase, "message": f"캘리브레이션 수집: {len(CALIB_BUFFER[session_id])}프레임"}

        if phase == "surprise":
            base = SESSION_BASELINE.get(session_id)
            # 기준 있든 없든 mouth를 버퍼에 누적 저장
            SURPRISE_BUFFER.setdefault(session_id, []).append({
                "path": save_path.replace("\\","/"),
                "mouth": float(mouth),
                "ts": ts
            })

            if not base:
                msg = f"캘리브레이션 없음 (mouth={mouth:.3f})"
            else:
                delta = mouth - base["mouth"]
                msg = f"{'하강' if delta > MOUTH_DELTA else '상승/중립'} (Δ={delta:+.3f}, 기준={base['mouth']:.3f}, 현재={mouth:.3f})"

            return {
                "saved_path": save_path.replace("\\","/"),
                "phase": phase,
                "message": msg,
                "surprise_count": len(SURPRISE_BUFFER.get(session_id, []))
            }


        return JSONResponse(status_code=400, content={"detail": f"알 수 없는 phase: {phase}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# 캘리브레이션 완료
@app.post("/finalize_calibration")
async def finalize_calibration(session_id: str = Form(...)):
    buf = CALIB_BUFFER.get(session_id, [])
    if not buf:
        return JSONResponse(status_code=400, content={"detail":"캘리브레이션 버퍼 비어있음"})
    arr = np.array(buf, dtype=np.float32)  # [N, 6]
    yaw_b, pitch_b, ear_b, mouth_b, eye_h_b, eye_v_b = np.median(arr, axis=0)
    baseline = {
        "yaw": float(yaw_b), "pitch": float(pitch_b),
        "ear": float(ear_b), "mouth": float(mouth_b),
        "eye_h": float(eye_h_b), "eye_v": float(eye_v_b),
    }
    SESSION_BASELINE[session_id] = baseline
    CALIB_BUFFER[session_id] = []
    return {"baseline": baseline, "frames_used": int(arr.shape[0])}

# 동영상 분석
@app.post("/analyze_video")
async def analyze_video(
    video: UploadFile = File(...),
    session_id: str = Form(...),
    blink_limit_per_min: int = Form(DEFAULT_BLINK_LIMIT),
    baseline_seconds: float = Form(2.0),
    frame_stride: int = Form(5),     # 속도 향상
    draw_overlay: int = Form(1)      # 1이면 오버레이 영상 생성
):
    try:
        ts = time.strftime("%Y%m%d-%H%M%S")
        raw_name = f"video_{session_id}_{ts}.mp4"
        vpath = os.path.join(SNAP_DIR, raw_name)
        with open(vpath, "wb") as f:
            f.write(await video.read())

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"detail":"영상을 열 수 없습니다."})

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0 or np.isnan(fps): fps = 30.0

        base = SESSION_BASELINE.get(session_id)
        baseline_from_video = False
        vbuf: List[Tuple[float,float,float,float,float,float]] = []

        stats = {
            "frames": 0,
            "frames_head_ok": 0,
            "frames_eye_ok": 0,
            "frames_eye_valid": 0,     # 눈 신호 유효 프레임
            "frames_head_eye_ok": 0,
            "blinks_count": 0,
        }
        accum = {"ear": [], "mouth": [], "eye_h": [], "eye_v": [], "yaw": [], "pitch": []}

        ema_vals = {"yaw": None, "pitch": None, "ear": None, "mouth": None, "eye_h": None, "eye_v": None}
        blink_in_progress = False
        last_blink_t = 0.0

        # 오버레이 영상 출력 준비
        writer = None
        annotated_url = None

        idx = -1
        out_w = None
        out_h = None

        # 지표 수집 버퍼
        gaze_head_x, gaze_head_y = [], []       # dyaw, dpitch (HEAD 기준)
        gaze_both_x, gaze_both_y = [], []       # eye_h_corr, eye_v_corr (HEAD∧EYE 안정 프레임)

        while True:
            ok, frame0 = cap.read()
            if not ok: break
            idx += 1
            if frame_stride > 1 and (idx % frame_stride) != 0:
                continue
            t = (idx / fps)

            feats, draw, frame = features_from_bgr(frame0, FACE_MESH_VIDEO)
            if feats is None:
                continue

            if out_w is None:
                out_h, out_w = frame.shape[:2]
                if draw_overlay:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    annotated_name = f"annotated_{session_id}_{ts}.mp4"
                    annotated_path = os.path.join(SNAP_DIR, annotated_name)
                    writer = cv2.VideoWriter(annotated_path, fourcc, max(1.0, fps/frame_stride), (out_w, out_h))
                    annotated_url = f"/static/{annotated_name}"

            yaw, pitch, ear, mouth, eye_h, eye_v = feats
            stats["frames"] += 1

            if (base is None) and (t <= baseline_seconds):
                vbuf.append((yaw, pitch, ear, mouth, eye_h or 0.0, eye_v or 0.0))

            # EMA
            ema_vals["yaw"]   = ema_update(ema_vals["yaw"],   yaw)
            ema_vals["pitch"] = ema_update(ema_vals["pitch"], pitch)
            ema_vals["ear"]   = ema_update(ema_vals["ear"],   ear)
            ema_vals["mouth"] = ema_update(ema_vals["mouth"], mouth)
            if eye_h is not None:
                ema_vals["eye_h"] = ema_update(ema_vals["eye_h"], eye_h)
            if eye_v is not None:
                ema_vals["eye_v"] = ema_update(ema_vals["eye_v"], eye_v)

            # baseline 결정
            if base is None and t > baseline_seconds:
                if len(vbuf) > 0:
                    arr = np.array(vbuf, dtype=np.float32)
                    yaw_b, pitch_b, ear_b, mouth_b, eye_h_b, eye_v_b = np.median(arr, axis=0)
                    base = {"yaw": float(yaw_b), "pitch": float(pitch_b), "ear": float(ear_b), "mouth": float(mouth_b),
                            "eye_h": float(eye_h_b), "eye_v": float(eye_v_b)}
                else:
                    base = {"yaw": yaw, "pitch": pitch, "ear": max(1e-6, ear), "mouth": mouth,
                            "eye_h": eye_h or 0.0, "eye_v": eye_v or 0.0}
                baseline_from_video = True
            if base is None:
                base = {"yaw": yaw, "pitch": pitch, "ear": max(1e-6, ear), "mouth": mouth,
                        "eye_h": eye_h or 0.0, "eye_v": eye_v or 0.0}
                baseline_from_video = True

            # 판정(Head)
            dyaw   = (ema_vals["yaw"]   - base["yaw"])   if ema_vals["yaw"]   is not None else 0.0
            dpitch = (ema_vals["pitch"] - base["pitch"]) if ema_vals["pitch"] is not None else 0.0
            HEAD_OK = (abs(dyaw) <= GAZE_OFF_ABS) and (abs(dpitch) <= GAZE_OFF_ABS)
            if HEAD_OK: stats["frames_head_ok"] += 1

            # 판정(Eye)
            EYE_OK = False
            eye_h_corr, eye_v_corr = None, None
            if (ema_vals["eye_h"] is not None) and (ema_vals["eye_v"] is not None):
                stats["frames_eye_valid"] += 1   # 유효 눈 프레임
                eye_h_corr = ema_vals["eye_h"] - base.get("eye_h", 0.0)
                eye_v_corr = ema_vals["eye_v"] - base.get("eye_v", 0.0)
                EYE_OK = (abs(eye_h_corr) <= EYE_OFF_ABS) and (abs(eye_v_corr) <= EYE_OFF_ABS)
                if EYE_OK: stats["frames_eye_ok"] += 1

            # 결합 판정
            HEAD_EYE_OK = HEAD_OK and EYE_OK
            if HEAD_EYE_OK: stats["frames_head_eye_ok"] += 1

            # 깜빡임
            eye_closed = (ema_vals["ear"] is not None) and (ema_vals["ear"] < base["ear"] * BLINK_RATIO)
            if eye_closed and not blink_in_progress:
                blink_in_progress = True
                last_blink_t = t
            elif (not eye_closed) and blink_in_progress:
                if (t - last_blink_t) >= BLINK_MIN_DUR:
                    stats["blinks_count"] += 1
                blink_in_progress = False

            # 누적값 저장
            accum["yaw"].append(yaw)
            accum["pitch"].append(pitch)
            accum["ear"].append(ear)
            accum["mouth"].append(mouth)
            if eye_h is not None: accum["eye_h"].append(eye_h)
            if eye_v is not None: accum["eye_v"].append(eye_v)

            # 지표 좌표 수집
            gaze_head_x.append(float(dyaw))
            gaze_head_y.append(float(dpitch))
            if HEAD_EYE_OK and (eye_h_corr is not None) and (eye_v_corr is not None):
                gaze_both_x.append(float(eye_h_corr))
                gaze_both_y.append(float(eye_v_corr))

            # ----- 오버레이 -----
            if draw_overlay and writer is not None:
                vis = frame.copy()

                if "nose" in draw:
                    cv2.circle(vis, draw["nose"], 3, (0,255,255), -1)

                for p in draw.get("mouth_pts", []):
                    cv2.circle(vis, p, 2, (0,200,255), -1)

                YELLOW = (0, 255, 255)
                PURPLE = (179, 179, 0)
                TH = 1
                if draw.get("l_eye_box"):
                    x1,y1,x2,y2 = draw["l_eye_box"]
                    cv2.rectangle(vis, (x1,y1), (x2,y2), YELLOW, TH)
                if draw.get("r_eye_box"):
                    x1,y1,x2,y2 = draw["r_eye_box"]
                    cv2.rectangle(vis, (x1,y1), (x2,y2), YELLOW, TH)
                if draw.get("l_iris_box"):
                    x1,y1,x2,y2 = draw["l_iris_box"]
                    cv2.rectangle(vis, (x1,y1), (x2,y2), PURPLE, TH)
                if draw.get("r_iris_box"):
                    x1,y1,x2,y2 = draw["r_iris_box"]
                    cv2.rectangle(vis, (x1,y1), (x2,y2), PURPLE, TH)

                hud = [
                    f"yaw={yaw:+.3f} pitch={pitch:+.3f}",
                    f"EAR={ear:.3f}  mouth={mouth:+.3f}",
                    f"eye_h={eye_h if eye_h is not None else float('nan'):+.3f}  eye_v={eye_v if eye_v is not None else float('nan'):+.3f}",
                    f"HEAD_OK={int(HEAD_OK)} EYE_OK={int(EYE_OK)} BOTH={int(HEAD_EYE_OK)}",
                    f"blink_count={stats['blinks_count']}"
                ]
                y0 = 18
                for ln in hud:
                    cv2.putText(vis, ln, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,50), 1, cv2.LINE_AA)
                    y0 += 18

                writer.write(vis)

        # 루프 종료
        cap.release()
        if writer is not None:
            writer.release()

        # 결과 집계
        duration_sec = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / max(1e-6, fps)
        dur = max(1e-6, duration_sec)
        head_gaze_rate = (stats["frames_head_ok"] / max(1, stats["frames"])) * 100.0
        head_eye_gaze_rate = (stats["frames_head_eye_ok"] / max(1, stats["frames"])) * 100.0

        # 눈만(EYE_ONLY) 주시율 — 유효 눈 프레임만 분모
        eye_valid = max(1, stats["frames_eye_valid"])
        eye_only_gaze_rate = (stats["frames_eye_ok"] / eye_valid) * 100.0

        blinks_per_min = stats["blinks_count"] / (dur / 60.0)

        # 지표 계산
        metrics_head = compute_fixation_metrics(gaze_head_x, gaze_head_y, target=(0.0, 0.0))
        metrics_both = compute_fixation_metrics(gaze_both_x, gaze_both_y, target=(0.0, 0.0))

        base_src = "세션 캘리브레이션 기준" if not baseline_from_video else "영상 초반 기준(임시)"
        gaze_msg_head = "(머리) 정면 주시 양호" if head_gaze_rate >= 80 else "(머리) 주시율을 더 높여보세요(80%+)"
        gaze_msg_both = "(머리∧눈) 정면 주시 양호" if head_eye_gaze_rate >= 70 else "(머리∧눈) 주시율 개선(70%+ 권장)"
        gaze_msg_eye = "(눈) 정면 주시 양호" if eye_only_gaze_rate >= 75 else "(눈) 주시율 개선(75%+ 권장)"

        metrics_summary = (
            f"Head-only MAE={_fmt4(metrics_head['MAE'])}, BCEA={_fmt6(metrics_head['BCEA'])} | "
            f"Head∧Eye MAE={_fmt4(metrics_both['MAE'])}, BCEA={_fmt6(metrics_both['BCEA'])}"
        )

        md = (
            f"### 분석 결과 — `{video.filename}`\n"
            f"- 영상 길이(원본): {dur:.1f}s\n"
            f"- 기준 소스: {base_src}\n"
            f"- 정면 주시율(머리만): {head_gaze_rate:.1f}%\n"
            f"- 정면 주시율(눈만): {eye_only_gaze_rate:.1f}% (유효 눈 프레임 {stats['frames_eye_valid']}개)\n"
            f"- 정면 주시율(머리∧눈): {head_eye_gaze_rate:.1f}%\n"
            f"- 깜빡임 총횟수: {stats['blinks_count']}회 ({blinks_per_min:.1f}회/분, 기준 {blink_limit_per_min}/분)\n"
            f"- 시선 지표(요약): {metrics_summary}\n\n"
            f"피드백\n- {gaze_msg_head}\n- {gaze_msg_eye}\n- {gaze_msg_both}\n"
        )

        def m_s(lst):
            if not lst:
                return (None, None)
            arr = np.array(lst, dtype=np.float32)
            return (_safe_num(arr.mean()), _safe_num(arr.std()))
        stats_detail = {
            "yaw_mean_std": m_s(accum["yaw"]),
            "pitch_mean_std": m_s(accum["pitch"]),
            "ear_mean_std": m_s(accum["ear"]),
            "mouth_mean_std": m_s(accum["mouth"]),
            "eye_h_mean_std": m_s(accum["eye_h"]),
            "eye_v_mean_std": m_s(accum["eye_v"]),
        }

        resp = {
            "report_md": md,
            "baseline_used": base,
            "processed_frames": stats["frames"],
            "frame_stride": frame_stride,
            "head_gaze_rate": head_gaze_rate,
            "eye_only_gaze_rate": eye_only_gaze_rate,
            "eye_valid_frames": stats["frames_eye_valid"],
            "head_eye_gaze_rate": head_eye_gaze_rate,
            "blinks_count": stats["blinks_count"],
            "blinks_per_min": blinks_per_min,
            "stats_detail": stats_detail,
            "annotated_video_url": annotated_url,
            "gaze_metrics": {
                "head_only": metrics_head,
                "head_and_eye": metrics_both
            }
        }

        # NaN/Inf → None 치환 (JSON 직렬화 안전)
        resp = sanitize_for_json(resp)
        return resp

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    
@app.post("/analyze_surprise_photos")
async def analyze_surprise_photos(
    session_id: str = Form(...)
):
    """
    이 세션에서 업로드된 중간 스냅샷들의 mouth(입꼬리 상대 높이)를
    캘리브레이션 기준과 비교·집계하여 피드백을 반환.
    """
    try:
        base = SESSION_BASELINE.get(session_id)
        if not base:
            return JSONResponse(status_code=400, content={"detail": "캘리브레이션이 없습니다. 먼저 /finalize_calibration 을 완료하세요."})

        items = SURPRISE_BUFFER.get(session_id, [])
        if not items:
            return JSONResponse(status_code=400, content={"detail": "분석할 surprise 스냅샷이 없습니다. /upload_snapshot(phase='surprise')로 업로드하세요."})

        mouths = [it["mouth"] for it in items if isinstance(it.get("mouth"), (int, float))]
        if not mouths:
            return JSONResponse(status_code=400, content={"detail": "유효한 mouth 값이 없습니다."})

        m_arr = np.array(mouths, dtype=np.float32)
        mouth_mean = float(m_arr.mean())
        mouth_std  = float(m_arr.std())
        mouth_min  = float(m_arr.min())
        mouth_max  = float(m_arr.max())
        mouth_med  = float(np.median(m_arr))
        mouth_delta = mouth_mean - base["mouth"]  # (-) 미소 경향, (+) 하강 경향

        # 간단 등급 (영상 분석과 동일 임계 MOUTH_DELTA 사용)
        if mouth_delta <= -MOUTH_DELTA:
            mouth_msg = f"입꼬리 상승(미소 경향) — Δ={mouth_delta:+.3f}, σ={mouth_std:.3f}"
        elif mouth_delta >=  MOUTH_DELTA:
            mouth_msg = f"입꼬리 하강 경향 — Δ={mouth_delta:+.3f}, σ={mouth_std:.3f}"
        else:
            mouth_msg = f"입꼬리 중립 — Δ={mouth_delta:+.3f}, σ={mouth_std:.3f}"

        # 간단 리포트 (마크다운)
        md = (
            f"### 중간 스냅샷 입꼬리 분석 — 세션 `{session_id}`\n"
            f"- 스냅샷 개수: {len(mouths)}장\n"
            f"- 캘리브레이션 기준 mouth: {base['mouth']:.3f}\n"
            f"- mouth 평균/중앙값: {mouth_mean:.3f} / {mouth_med:.3f}\n"
            f"- 분산(σ)/최소/최대: {mouth_std:.3f} / {mouth_min:.3f} / {mouth_max:.3f}\n"
            f"- Δ(평균−기준): {mouth_delta:+.3f}  (음수=미소, 양수=하강)\n\n"
            f"피드백\n- {mouth_msg}\n"
        )

        details = {
            "baseline_mouth": float(base["mouth"]),
            "surprise_count": int(len(mouths)),
            "mouth_mean": float(mouth_mean),
            "mouth_median": float(mouth_med),
            "mouth_std": float(mouth_std),
            "mouth_min": float(mouth_min),
            "mouth_max": float(mouth_max),
            "delta_vs_baseline": float(mouth_delta),
            "message": mouth_msg,
            # 원하면 각 사진별 값/경로도 반환
            "per_image": [
                {"path": it["path"], "mouth": float(it["mouth"]), "ts": it.get("ts")}
                for it in items
            ]
        }

        resp = {
            "report_md": md,
            "summary": details
        }
        return sanitize_for_json(resp)

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
