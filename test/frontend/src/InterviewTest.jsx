import React, { useMemo, useRef, useState } from "react";

/**
 * Interview Test UI (React)
 * Endpoints expected:
 *  POST /upload_snapshot (FormData: session_id, phase, image)
 *  POST /finalize_calibration (FormData: session_id)
 *  POST /analyze_video (FormData: session_id, blink_limit_per_min, frame_stride, draw_overlay, video)
 *  POST /analyze_surprise_photos (FormData: session_id)
 */

const API_BASE = import.meta.env.VITE_API_BASE || "http://your-server-address";

function useLogs() {
  const [lines, setLines] = useState([]);
  const log = (msg) => {
    const ts = new Date().toLocaleTimeString();
    const line = `[${ts}] ${msg}`;
    setLines((prev) => [...prev, line]);
    // eslint-disable-next-line no-console
    console.log(msg);
  };
  return { lines, log };
}

function Row({ children, right }) {
  return (
    <div className="row">
      <div className="row-l">{children}</div>
      {right ? <div className="row-r">{right}</div> : null}
    </div>
  );
}

function Card({ title, subtitle, children, footer }) {
  return (
    <section className="card">
      <header className="card-h">
        <div className="card-title">{title}</div>
        {subtitle ? <div className="card-sub">{subtitle}</div> : null}
      </header>
      <div className="card-b">{children}</div>
      {footer ? <footer className="card-f">{footer}</footer> : null}
    </section>
  );
}

function KV({ k, v }) {
  return (
    <div className="kv">
      <div className="kv-k">{k}</div>
      <div className="kv-v">{v}</div>
    </div>
  );
}

export default function InterviewUI() {
  // Refs/State
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [sessionId, setSessionId] = useState("");
  const [calibMsg, setCalibMsg] = useState("");

  const [qInput, setQInput] = useState("");
  const [questions, setQuestions] = useState([]);
  const [qIdx, setQIdx] = useState(0);
  const [captureIdxs, setCaptureIdxs] = useState([]);
  const [capturesDone, setCapturesDone] = useState(0);

  const [file, setFile] = useState(null);
  const [blinkLimit, setBlinkLimit] = useState(30);
  const [stride, setStride] = useState(5);
  const [drawOverlay, setDrawOverlay] = useState(true);

  const [reportHTML, setReportHTML] = useState("");       // 영상 분석 리포트
  const [metrics, setMetrics] = useState(null);
  const [annotatedSrc, setAnnotatedSrc] = useState("");

  const [surpriseHTML, setSurpriseHTML] = useState("");   
  const [surpriseSummary, setSurpriseSummary] = useState(null); 

  const { lines, log } = useLogs();

  const ensureSession = () => {
    if (!sessionId) {
      const id = `sess_${Math.random().toString(36).slice(2)}_${Date.now()}`;
      setSessionId(id);
      return id;
    }
    return sessionId;
  };

  const api = (p) => `${API_BASE.replace(/\/$/, "")}${p.startsWith("/") ? p : `/${p}`}`;

  // Camera 
  const startCam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      log("카메라 시작");
    } catch (e) {
      log(`❌ 카메라 시작 실패: ${e?.message || e}`);
    }
  };
  const stopCam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    log("카메라 중지");
  };

  const drawToCanvas = async (w = 640, h = 360) => {
    const c = canvasRef.current;
    const v = videoRef.current;
    if (!c || !v) return null;
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d");
    ctx.drawImage(v, 0, 0, w, h);
    return new Promise((res) => c.toBlob((b) => res(b), "image/jpeg", 0.9));
  };

  const uploadSnapshot = async (blob, phase) => {
    const id = ensureSession();
    const fd = new FormData();
    fd.append("session_id", id);
    fd.append("phase", phase);
    fd.append("image", blob, `${phase}_${Date.now()}.jpg`);
    const r = await fetch(api("/upload_snapshot"), { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "업로드 실패");
    return j;
  };

  const finalizeCalibration = async (id) => {
    const fd = new FormData();
    fd.append("session_id", id);
    const r = await fetch(api("/finalize_calibration"), { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "캘리브레이션 실패");
    return j;
  };

  const calibrate2s = async () => {
    if (!streamRef.current) {
      log("❌ 카메라가 꺼져 있습니다.");
      return;
    }
    const id = ensureSession();
    setCalibMsg("캘리브레이션 중… 2초");
    log(`세션 생성: ${id} — 캘리브레이션 시작`);

    let sent = 0;
    const started = Date.now();
    const iv = setInterval(async () => {
      try {
        if (!streamRef.current) throw new Error("카메라 중지됨");
        const blob = await drawToCanvas();
        const j = await uploadSnapshot(blob, "calib");
        sent += 1;
        setCalibMsg(`수집 프레임: ${sent}`);
        if (Date.now() - started >= 2000) {
          clearInterval(iv);
          const out = await finalizeCalibration(id);
          setCalibMsg(`✅ 완료 (frames=${out.frames_used}) baseline=${JSON.stringify(out.baseline)}`);
          log("캘리브레이션 완료");
        }
      } catch (e) {
        clearInterval(iv);
        setCalibMsg(`⚠️ 에러: ${e.message}`);
        log(`캘리브레이션 에러: ${e.message}`);
      }
    }, 200);
  };

  // Q&A + Surprise
  const chooseRandomIndices = (n, kMin = 2, kMax = 3) => {
    const target = Math.min(n, Math.max(kMin, Math.min(kMax, kMin + Math.floor(Math.random() * (kMax - kMin + 1)))));
    const s = new Set();
    while (s.size < target) s.add(Math.floor(Math.random() * n));
    return Array.from(s).sort((a, b) => a - b);
  };

  const resetSession = () => {
    const raw = qInput
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    if (!raw.length) {
      log("❌ 질문을 입력하세요.");
      return;
    }
    const id = ensureSession();
    setQuestions(raw);
    setQIdx(0);
    const picks = chooseRandomIndices(raw.length, 2, 3);
    setCaptureIdxs(picks);
    setCapturesDone(0);
    setSurpriseHTML("");    
    setSurpriseSummary(null);
    log(`세션 시작 — 질문 ${raw.length}개, 서프라이즈=${picks.map((x) => x + 1).join(", ")}`);
  };

  const triggerSurpriseIfNeeded = async () => {
    if (!streamRef.current) return;
    if (captureIdxs.includes(qIdx) && capturesDone < captureIdxs.length) {
      try {
        const blob = await drawToCanvas();
        const j = await uploadSnapshot(blob, "surprise");
        setCapturesDone((c) => c + 1);
        const extra = typeof j.surprise_count === "number" ? ` (누적 ${j.surprise_count}장)` : "";
        log(`📸 서프라이즈(Q${qIdx + 1}): ${(j.message || "OK")}${extra}`);
      } catch (e) {
        log(`❌ 서프라이즈 실패: ${e.message}`);
      }
    }
  };

  const nextQuestion = async () => {
    if (!questions.length) return;
    await triggerSurpriseIfNeeded();
    setQIdx((i) => Math.min(i + 1, questions.length));
  };

  const progressText = useMemo(() => {
    if (!questions.length) return "";
    const left = Math.max(0, captureIdxs.length - capturesDone);
    return `${Math.min(qIdx + 1, questions.length)} / ${questions.length} · 남은 서프라이즈 ${left}`;
  }, [questions.length, qIdx, captureIdxs.length, capturesDone]);

  // Analyze Video 
  const analyzeVideo = async () => {
    if (!file) {
      log("❌ MP4 파일을 선택하세요.");
      return;
    }
    const id = ensureSession();
    setReportHTML("분석 중…");
    setMetrics(null);
    setAnnotatedSrc("");

    const fd = new FormData();
    fd.append("session_id", id);
    fd.append("blink_limit_per_min", String(parseInt(String(blinkLimit) || "30", 10)));
    fd.append("frame_stride", String(parseInt(String(stride) || "5", 10)));
    fd.append("draw_overlay", drawOverlay ? "1" : "0");
    fd.append("video", file);

    const r = await fetch(api("/analyze_video"), { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) {
      setReportHTML(`<div class='warn'>❌ 분석 실패: ${j.detail}</div>`);
      log(`분석 실패: ${j.detail}`);
      return;
    }

    setReportHTML(j.report_md.replace(/\\n/g, "<br>"));
    setMetrics(j);
    if (j.annotated_video_url) {
      const src = `${API_BASE.replace(/\/$/, "")}${j.annotated_video_url}`;
      setAnnotatedSrc(src);
      log(`오버레이 영상: ${src}`);
    } else {
      log("오버레이 영상 없음");
    }
    log(`영상 분석 완료 (frames=${j.processed_frames}, stride=${j.frame_stride})`);
  };

  // Analyze Surprise Photos
  const analyzeSurprisePhotos = async () => {
    const id = ensureSession();
    setSurpriseHTML("분석 중…");
    setSurpriseSummary(null);

    const fd = new FormData();
    fd.append("session_id", id);

    const r = await fetch(api("/analyze_surprise_photos"), { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) {
      setSurpriseHTML(`<div class='warn'>❌ 분석 실패: ${j.detail}</div>`);
      log(`서프라이즈 분석 실패: ${j.detail}`);
      return;
    }

    setSurpriseHTML(j.report_md.replace(/\\n/g, "<br>"));
    setSurpriseSummary(j.summary || null);
    log(`서프라이즈 스냅샷 분석 완료 (count=${j?.summary?.surprise_count ?? "?"})`);
  };

  // Render
  return (
    <div className="wrap">
      <style>{CSS}</style>
      <h1 className="h1">Interview Test — Clean UI</h1>

      <div className="grid">
        {/* Left column */}
        <div className="col">
          <Card
            title="① 카메라 & 2초 캘리브레이션"
            subtitle="200ms 간격으로 프레임 수집 → 기준 생성"
            footer={<div className="muted">{calibMsg || "상태 메시지 표시"}</div>}
          >
            <Row>
              <button className="btn" onClick={startCam}>🎬 시작</button>
              <button className="btn" onClick={calibrate2s} disabled={!streamRef.current}>🎯 캘리브레이션(2s)</button>
              <button className="btn" onClick={stopCam} disabled={!streamRef.current}>⏹ 중지</button>
            </Row>
            <div className="videoBox">
              <video ref={videoRef} className="video" playsInline autoPlay muted />
            </div>
            <canvas ref={canvasRef} width={640} height={360} className="hide" />
          </Card>

          <Card title="② 질문 진행 + 랜덤 서프라이즈">
            <Row>
              <textarea
                className="ta"
                placeholder="한 줄에 하나씩 질문을 입력"
                value={qInput}
                onChange={(e) => setQInput(e.target.value)}
              />
            </Row>
            <Row right={<div className="muted">{questions.length ? progressText : ""}</div>}>
              <button className="btn" onClick={resetSession}>세션 시작</button>
              <button className="btn" onClick={nextQuestion} disabled={!questions.length || qIdx >= questions.length}>다음 ▶</button>
            </Row>
            {questions.length ? (
              <div className="hl">
                {qIdx < questions.length ? `Q${qIdx + 1}) ${questions[qIdx]}` : "🎉 모든 질문 완료"}
              </div>
            ) : null}
          </Card>
        </div>

        {/* Right column */}
        <div className="col">
          <Card title="③ 업로드 영상 분석" subtitle="머리/눈 정면주시율 + 깜빡임 + 오버레이">
            <Row>
              <input type="file" accept="video/mp4" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            </Row>
            <Row>
              <label className="lbl">깜빡임 기준(회/분)
                <input className="in" type="number" min={5} max={60} value={blinkLimit} onChange={(e) => setBlinkLimit(Number(e.target.value || 30))} />
              </label>
              <label className="lbl">프레임 stride
                <input className="in" type="number" min={1} max={10} value={stride} onChange={(e) => setStride(Number(e.target.value || 5))} />
              </label>
              <label className="chk">
                <input type="checkbox" checked={drawOverlay} onChange={(e) => setDrawOverlay(e.target.checked)} /> 오버레이 저장
              </label>
              <button className="btn" onClick={analyzeVideo} disabled={!file}>분석 실행</button>
            </Row>

            <div className="report" dangerouslySetInnerHTML={{ __html: reportHTML }} />

            {metrics ? (
              <div className="table">
                <KV k="처리 프레임" v={metrics.processed_frames} />
                <KV k="프레임 stride" v={metrics.frame_stride} />
                <KV k="정면 주시율(머리)" v={`${metrics.head_gaze_rate?.toFixed?.(1)}%`} />
                <KV k="정면 주시율(머리∧눈)" v={`${metrics.head_eye_gaze_rate?.toFixed?.(1)}%`} />
                <KV k="깜빡임 총횟수" v={metrics.blinks_count} />
                <KV k="깜빡임/분" v={metrics.blinks_per_min?.toFixed?.(1)} />
                <KV k="Baseline" v={<code>{JSON.stringify(metrics.baseline_used || {})}</code>} />
              </div>
            ) : null}

            {annotatedSrc ? (
              <div className="videoBox mt">
                <video src={annotatedSrc} className="video" controls />
              </div>
            ) : null}
          </Card>

          {/* Surprise photos analysis */}
          <Card title="④ Surprise 스냅샷 분석" subtitle="캘리브레이션 기준 vs 사진의 입꼬리(미소) 변화">
            <Row>
              <button className="btn" onClick={analyzeSurprisePhotos}>📸 분석 실행</button>
            </Row>

            <div className="report" dangerouslySetInnerHTML={{ __html: surpriseHTML }} />

            {surpriseSummary ? (
              <div className="table">
                <KV k="스냅샷 개수" v={surpriseSummary.surprise_count} />
                <KV k="Baseline mouth" v={surpriseSummary.baseline_mouth?.toFixed?.(3)} />
                <KV k="평균 / 중앙값" v={`${surpriseSummary.mouth_mean?.toFixed?.(3)} / ${surpriseSummary.mouth_median?.toFixed?.(3)}`} />
                <KV k="표준편차 σ" v={surpriseSummary.mouth_std?.toFixed?.(3)} />
                <KV k="최소 / 최대" v={`${surpriseSummary.mouth_min?.toFixed?.(3)} / ${surpriseSummary.mouth_max?.toFixed?.(3)}`} />
                <KV k="Δ(평균−기준)" v={surpriseSummary.delta_vs_baseline?.toFixed?.(3)} />
              </div>
            ) : null}
          </Card>

          <Card title="로그">
            <div className="log">
              {lines.length ? lines.map((l, i) => <div key={i}>{l}</div>) : <span className="muted">로그가 여기에 표시됩니다</span>}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

const CSS = `
:root { --pad: 14px; --gap: 12px; --radius: 14px; --bg:#fff; --fg:#111; --muted:#6a6a6a; --b:#e8e8e8; --accent:#2563eb; }
* { box-sizing: border-box; }
html, body, #root { height: 100%; }
body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, Apple SD Gothic Neo, sans-serif; color:var(--fg); background:#fafafa; }
.wrap { max-width: 1120px; margin: 18px auto; padding: 0 14px 24px; }
.h1 { font-size: 20px; font-weight: 700; margin: 4px 0 14px; }
.grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
@media (min-width: 980px) { .grid { grid-template-columns: 1fr 1fr; } }
.col { display: flex; flex-direction: column; gap: 14px; }
.card { background: var(--bg); border:1px solid var(--b); border-radius: var(--radius); overflow: hidden; }
.card-h { padding: 12px var(--pad) 6px; border-bottom:1px solid var(--b); }
.card-title { font-weight: 700; }
.card-sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
.card-b { padding: var(--pad); display:flex; flex-direction: column; gap: 10px; }
.card-f { border-top:1px solid var(--b); padding: 8px var(--pad); font-size:13px; }
.muted { color: var(--muted); }
.row { display:flex; gap: var(--gap); align-items: center; justify-content: flex-start; flex-wrap: wrap; }
.row-l { display:flex; gap: var(--gap); align-items:center; flex-wrap: wrap; }
.row-r { margin-left:auto; color:var(--muted); font-size:13px; }
.btn { padding: 9px 12px; border:1px solid var(--b); border-radius: 10px; background:#fff; cursor:pointer; font-weight:600; }
.btn:disabled { opacity: .55; cursor: not-allowed; }
.lbl { display:flex; flex-direction: column; gap:6px; font-size: 12px; color:#333; }
.in { width: 90px; padding:8px; border:1px solid var(--b); border-radius: 8px; }
.chk { display:flex; align-items:center; gap:8px; font-size:14px; }
.ta { width:100%; min-height:110px; padding:10px; border:1px solid var(--b); border-radius: 10px; resize: vertical; }
.videoBox { width:100%; border:1px solid var(--b); border-radius: 12px; background:#000; overflow:hidden; }
.video { display:block; width:100%; height:auto; max-height: 380px; }
.hide { display:none; }
.hl { padding:10px 12px; background:#f6fafc; border:1px solid var(--b); border-radius:10px; font-weight:600; }
.table { display:grid; grid-template-columns: 1fr 1fr; gap:8px; }
.kv { display:grid; grid-template-columns: 140px 1fr; align-items:center; border:1px solid var(--b); border-radius: 10px; overflow:hidden; }
.kv-k { background:#f8f8f8; font-size:13px; padding:8px 10px; border-right:1px solid var(--b); }
.kv-v { padding:8px 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap:anywhere; }
.report { font-size:14px; color:#222; line-height: 1.5; }
.report h3 { margin: 10px 0 6px; }
.log { min-height:120px; max-height: 260px; overflow:auto; background:#f7f7f7; border:1px solid var(--b); border-radius: 10px; padding:10px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; }
.mt { margin-top: 10px; }
`;
