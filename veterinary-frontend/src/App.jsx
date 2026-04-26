import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Upload, Activity, ShieldAlert, User, Thermometer, FileText,
  CheckCircle2, Loader2, AlertTriangle, Weight, FlaskConical,
  Stethoscope, MessageSquare, Send, X, Crop, Eye, BarChart3,
  Droplets, Sun, Heart, Zap, ChevronRight, Calendar
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { submitPrediction, checkHealth, agentChat, analyzeCrop, clinicalQA, bovineIqChat } from './api';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs) { return twMerge(clsx(inputs)); }

const LOADING_STEPS = [
  "Running CowReIDModel identification...",
  "Extracting ViT + ArcFace embeddings...",
  "Predicting milk yield (Transformer)...",
  "Analyzing heat stress (THI + Behavior)...",
  "Computing health score (Fusion + Anomaly)...",
  "Querying Neo4j Knowledge Graph...",
  "Retrieving PubMed evidence...",
  "Synthesizing IDSS report (Llama 3.3 70B)..."
];

export default function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [weight, setWeight] = useState(600);
  const [age, setAge] = useState("");
  const [cowId, setCowId] = useState("");
  const [temperature, setTemperature] = useState("");
  const [heartRate, setHeartRate] = useState("");
  const [dayIndex, setDayIndex] = useState(0);
  const [showClinicalFields, setShowClinicalFields] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiOnline, setApiOnline] = useState(true);
  
  // Refinement states
  const [selectedCowId, setSelectedCowId] = useState(null);
  const [showRefinementModal, setShowRefinementModal] = useState(false);
  const [refinementLoading, setRefinementLoading] = useState(false);

  // Crop state
  const [isCropping, setIsCropping] = useState(false);
  const [cropStart, setCropStart] = useState(null);
  const [cropRect, setCropRect] = useState(null);
  const [cropDescription, setCropDescription] = useState("");
  const [cropResult, setCropResult] = useState(null);
  const [cropLoading, setCropLoading] = useState(false);
  const imageWrapperRef = useRef(null);

  const [chatInput, setChatInput] = useState("");
  const [chatImage, setChatImage] = useState(null);
  const [chatPreview, setChatPreview] = useState(null);
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', content: "Hello! I'm your **Veterinary AI Agent Assistant**. Upload a cow image here or use the main uploader to begin our clinical analysis. 🐄✨" }
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);
  const idssEndRef = useRef(null);

  useEffect(() => { checkHealth().then(setApiOnline); }, []);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [chatMessages]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) { setImage(file); setPreview(URL.createObjectURL(file)); setError(null); setResult(null); setCropRect(null); setCropResult(null); }
  };

  const handleSubmit = async () => {
    if (!image) { setError("Please upload an image first."); return; }
    setLoading(true); setResult(null); setError(null); setLoadingStep(0);
    const stepInterval = setInterval(() => { setLoadingStep(prev => (prev < LOADING_STEPS.length - 1 ? prev + 1 : prev)); }, 900);
    try {
      const extraParams = JSON.stringify({
        temperature: parseFloat(temperature) || null,
        heart_rate: parseFloat(heartRate) || null
      });
      const data = await submitPrediction(image, weight, age, cowId, extraParams, true, dayIndex);
      setResult(data);
      setShowClinicalFields(true);
      const gate = data.stages?.gate;
      if (gate?.status === "PASSED") {
        setChatMessages(prev => [...prev, { role: 'assistant', content: `✅ **Cow #${data.cow_id}** identified. Pipeline complete. I have generated the clinical assessment and IDSS report below. How can I assist you with this case?` }]);
      } else {
        const rejectMsg = `🚫 **Pipeline Rejected:** ${gate?.reason || 'Analysis incomplete'}\n\n${gate?.action || 'Please upload a valid MMCOWS cow image.'}\n\n_Chat is disabled until a valid cow is identified._`;
        setChatMessages(prev => [...prev, { role: 'assistant', content: rejectMsg }]);
      }
    } catch (err) { setError(err.message); }
    finally { clearInterval(stepInterval); setLoading(false); }
  };

  // Crop handlers
  const handleMouseDown = (e) => {
    if (!isCropping || !imageWrapperRef.current) return;
    const rect = imageWrapperRef.current.getBoundingClientRect();
    setCropStart({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    setCropRect(null);
  };
  const handleMouseMove = (e) => {
    if (!isCropping || !cropStart || !imageWrapperRef.current) return;
    const rect = imageWrapperRef.current.getBoundingClientRect();
    const x = Math.min(cropStart.x, e.clientX - rect.left);
    const y = Math.min(cropStart.y, e.clientY - rect.top);
    const w = Math.abs(e.clientX - rect.left - cropStart.x);
    const h = Math.abs(e.clientY - rect.top - cropStart.y);
    setCropRect({ x: Math.round(x), y: Math.round(y), width: Math.round(w), height: Math.round(h) });
  };
  const handleMouseUp = () => { setCropStart(null); };

  const handleCropAnalysis = async () => {
    if (!image || !cropRect) return;
    setCropLoading(true); setCropResult(null);
    const imgElement = imageWrapperRef.current.querySelector('img');
    let scaleX = 1;
    let scaleY = 1;
    if (imgElement) {
        scaleX = imgElement.naturalWidth / imgElement.clientWidth;
        scaleY = imgElement.naturalHeight / imgElement.clientHeight;
    }
    
    const scaledCropRect = {
        x: Math.round(cropRect.x * scaleX),
        y: Math.round(cropRect.y * scaleY),
        width: Math.round(cropRect.width * scaleX),
        height: Math.round(cropRect.height * scaleY),
    };

    try {
      const data = await analyzeCrop(image, scaledCropRect.x, scaledCropRect.y, scaledCropRect.width, scaledCropRect.height, cropDescription);
      setCropResult(data);
      const idInfo = data.identification;
      
      const summaryMsg = idInfo?.is_known_cow
        ? `🔍 **Crop Analysis:** Cow #${idInfo.cow_id} detected (${(idInfo.confidence * 100).toFixed(1)}% confidence). ${data.agent_context || ''}`
        : `🔍 **Crop Analysis:** ${idInfo?.message || 'No known cow in this region.'}${data.vision_analysis?.overall_health_assessment ? '\n\nVision: ' + data.vision_analysis.overall_health_assessment : ''}`;
      
      setChatMessages(prev => [...prev, { role: 'assistant', content: summaryMsg }]);
      
      // Automatically trigger agent response to the crop analysis with vision context
      const agentPrompt = `Analyze the clinical implications of this cropped region: ${summaryMsg}. Provide a veterinary differential diagnosis based on the visual and sensor data.`;
      
      // Pass the specific crop base64 to the agent for targeted vision analysis
      const cropB64 = idInfo?.crop_b64 || (preview ? preview.split(',')[1] : null);
      handleChatSubmit(null, agentPrompt, cropB64);
    } catch (err) {
      setCropResult({ error: err.message });
    }
    finally { setCropLoading(false); setIsCropping(false); }
  };

  const handleChatSubmit = async (e, directText = null, imageOverride = null) => {
    if (e) e.preventDefault();
    const textToSend = directText || chatInput.trim();
    if (!textToSend && !chatImage && !imageOverride) return;
    
    const displayMsg = textToSend || "Image Uploaded";
    const userMsg = { role: 'user', content: displayMsg };
    if (chatPreview) {
      userMsg.content += `\n\n[USER_IMAGE: ${chatPreview}]`;
    }
    setChatMessages(prev => [...prev, userMsg]);
    setChatInput(""); 
    
    const finalImage = imageOverride || (chatImage ? await fileToBase64(chatImage) : null);
    setChatImage(null); setChatPreview(null);
    setIsTyping(true);
    
    try {
      const data = await agentChat(textToSend || "Analyze this image.", result?.cow_id, result?.stages?.report?.report || "", finalImage);
      const formattedAnswer = data.answer || 'No response available.';
      setChatMessages(prev => [...prev, { role: 'assistant', content: formattedAnswer }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: "I'm having trouble connecting to the Veterinary Agent. Please try again." }]);
    }
    finally { setIsTyping(false); }
  };

  const fileToBase64 = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = error => reject(error);
  });

  const handleRefineCow = (cid) => {
    setSelectedCowId(cid);
    setCowId(cid); // Set the global cowId as well
    setShowRefinementModal(true);
  };

  const submitRefinement = async () => {
    if (!image || !selectedCowId) return;
    setRefinementLoading(true);
    setLoading(true); // Trigger global loading state for visual feedback
    try {
      const extraParams = JSON.stringify({
        temperature: parseFloat(temperature) || null,
        heart_rate: parseFloat(heartRate) || null
      });
      const data = await submitPrediction(image, weight, age, selectedCowId, extraParams, true, dayIndex);
      setResult(data);
      setShowRefinementModal(false);
      setChatMessages(prev => [...prev, { role: 'assistant', content: `🔄 **Refinement Complete:** Cow #${selectedCowId} analysis has been updated with your clinical data (Weight: ${weight}kg, Age: ${age}y).` }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setRefinementLoading(false);
      setLoading(false);
    }
  };


  const gate = result?.stages?.gate;
  const isRejected = gate && gate.status === "REJECTED";
  const summaries = result?.stages?.clinical_summaries || [];
  const currentMessages = chatMessages;
  const currentEndRef = chatEndRef;
  const chatDisabled = isRejected || isTyping;

  return (
    <div className="min-h-screen flex flex-col">
      {/* HEADER */}
      <header className="sticky top-0 z-30 border-b" style={{background:'rgba(255,255,255,0.95)', backdropFilter:'blur(12px)', borderColor:'rgba(42,157,92,0.15)'}}>
        <div className="max-w-[1600px] mx-auto px-6 py-3 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="relative flex items-center justify-center w-10 h-10 rounded-xl" style={{background:'linear-gradient(135deg,#2a9d5c,#0d3f24)'}}>
              <span className="text-xl">🐄</span>
              <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-emerald-400 border-2 border-white" />
            </div>
            <div>
              <h1 className="text-base font-black tracking-tight" style={{fontFamily:'Outfit,sans-serif',color:'#0d3f24'}}>Veterinary AI <span className="font-light">Intelligence</span></h1>
              <p className="text-[10px] font-semibold uppercase tracking-[0.15em]" style={{color:'#5d8f8b'}}>MMCOWS · 16 Cows · 5 Models · Llama 4</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className={cn("flex items-center gap-2 text-[11px] font-bold px-3 py-1.5 rounded-full",
              apiOnline ? "badge-online" : "badge-danger")}>
              <div className={cn("w-2 h-2 rounded-full", apiOnline ? "status-dot-online" : "bg-rose-500 animate-pulse")} />
              {apiOnline ? 'System Online' : 'Offline'}
            </div>
          </div>
        </div>
      </header>

      {/* MAIN 3-COLUMN LAYOUT */}
      <main className="flex-1 max-w-[1600px] mx-auto px-4 pt-5 pb-10 grid grid-cols-1 lg:grid-cols-12 gap-5 w-full">

        {/* ══ LEFT: UPLOAD & INPUTS ══ */}
        <section className="lg:col-span-3 space-y-4 animate-enter">
          <div className="vet-card p-4">
            <p className="section-label"><Upload className="w-3 h-3" /> Upload & Configure</p>
            <div className={cn("border-2 border-dashed rounded-xl p-6 transition-all flex flex-col items-center justify-center text-center gap-3 cursor-pointer",
              preview ? "border-clinical-200 bg-slate-50" : "border-slate-300 hover:border-clinical-400")}
              onClick={() => document.getElementById('fileInput').click()}>
              {preview ? <img src={preview} alt="Preview" className="w-full h-36 object-cover rounded-lg" />
                : <><div className="bg-slate-100 p-3 rounded-full"><Upload className="w-6 h-6 text-slate-400" /></div>
                  <p className="text-xs font-semibold">Drop cattle image here</p></>}
              <input id="fileInput" type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
            </div>

            <div className="mt-4 space-y-3">
              {/* Clinical Fields - Hidden until toggle or result */}
              <div className="flex items-center justify-between">
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Clinical Data</h3>
                <button 
                  onClick={() => setShowClinicalFields(!showClinicalFields)}
                  className="text-[10px] text-clinical-600 font-bold hover:underline"
                >
                  {showClinicalFields ? "Hide" : "Show Optional Fields"}
                </button>
              </div>

              {showClinicalFields && (
                <motion.div 
                  initial={{ height: 0, opacity: 0 }} 
                  animate={{ height: "auto", opacity: 1 }}
                  className="space-y-3 overflow-hidden"
                >
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1 block">Weight (kg)</label>
                      <input type="number" value={weight} onChange={(e) => setWeight(e.target.value)}
                        className="w-full px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-sm font-semibold focus:ring-2 focus:ring-clinical-500 outline-none" />
                    </div>
                    <div>
                      <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1 block">Age (yrs)</label>
                      <input type="number" step="0.1" value={age} onChange={(e) => setAge(e.target.value)}
                        className="w-full px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-sm font-semibold focus:ring-2 focus:ring-clinical-500 outline-none" />
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1 block flex items-center gap-1">
                        <Thermometer className="w-2 h-2" /> Temp (°C)
                      </label>
                      <input type="number" step="0.1" value={temperature} onChange={(e) => setTemperature(e.target.value)} placeholder="38.5"
                        className="w-full px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-sm font-semibold focus:ring-2 focus:ring-clinical-500 outline-none" />
                    </div>
                    <div>
                      <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1 block flex items-center gap-1">
                        <Heart className="w-2 h-2" /> Heart (bpm)
                      </label>
                      <input type="number" value={heartRate} onChange={(e) => setHeartRate(e.target.value)} placeholder="60"
                        className="w-full px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-sm font-semibold focus:ring-2 focus:ring-clinical-500 outline-none" />
                    </div>
                  </div>

                  <div>
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1 block">Day Index (0-13)</label>
                    <input type="number" min="0" max="13" value={dayIndex} onChange={(e) => setDayIndex(parseInt(e.target.value) || 0)}
                      className="w-full px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-sm font-semibold focus:ring-2 focus:ring-clinical-500 outline-none" />
                  </div>
                  <div>
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1 block">Cow ID Override</label>
                    <input type="number" value={cowId} onChange={(e) => setCowId(e.target.value)} placeholder="Auto-detect"
                      className="w-full px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-sm font-semibold focus:ring-2 focus:ring-clinical-500 outline-none" />
                  </div>
                </motion.div>
              )}

              <button onClick={handleSubmit} disabled={loading || !image} className={cn("btn-primary w-full", (loading || !image) && 'opacity-50 cursor-not-allowed')}>
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <><Activity className="w-4 h-4" /> Run Full Pipeline</>}
              </button>

              {error && <div className="bg-rose-50 text-rose-700 p-2 rounded-lg text-xs border border-rose-100 flex gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />{error}</div>}
            </div>
          </div>

          {/* Crop Tool */}
          {preview && (
            <div className="glass-card p-4">
              <h3 className="text-xs font-bold flex items-center gap-2 mb-2"><Crop className="w-3 h-3" /> Interactive Crop</h3>
              <button onClick={() => setIsCropping(!isCropping)}
                className={cn("w-full py-1.5 rounded-lg text-xs font-bold transition-all",
                  isCropping ? "bg-sky-500 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200")}>
                {isCropping ? "✂️ Cropping Mode ON" : "Enable Crop Mode"}
              </button>
              {cropRect && (
                <div className="mt-2 space-y-2">
                  <p className="text-[10px] text-slate-500">Region: {cropRect.width}×{cropRect.height}px</p>
                  <input value={cropDescription} onChange={(e) => setCropDescription(e.target.value)}
                    placeholder="Describe what you see..."
                    className="w-full px-2 py-1 text-xs bg-slate-50 border rounded-lg outline-none" />
                  <button onClick={handleCropAnalysis} disabled={cropLoading}
                    className="w-full py-1.5 bg-sky-600 text-white rounded-lg text-xs font-bold disabled:opacity-50">
                    {cropLoading ? '⏳ Analyzing...' : '🔬 Analyze This Region'}
                  </button>
                </div>
              )}
              {/* Structured Crop Result */}
              {cropResult && !cropResult.error && (
                <div className="crop-result-card mt-2 p-2 bg-slate-50 rounded-lg border text-xs">
                  <h4 className="font-bold mb-1">🔍 Crop Analysis Result</h4>
                  {cropResult.identification?.is_known_cow ? (
                    <div className="space-y-1">
                      <p className="font-bold text-emerald-700">✅ Cow #{cropResult.identification.cow_id} detected</p>
                      <p>Confidence: <span className="font-black">{(cropResult.identification.confidence * 100).toFixed(1)}%</span></p>
                      <p>{cropResult.identification.message}</p>
                    </div>
                  ) : (
                    <p className="text-amber-700 font-bold">⚠️ {cropResult.identification?.message || 'No known cow in region'}</p>
                  )}
                  {cropResult.agent_context && <p className="text-slate-600 italic mt-1">{cropResult.agent_context}</p>}
                  
                  {cropResult.identification?.cow_id && (
                    <button onClick={() => handleRefineCow(cropResult.identification.cow_id)}
                      className="mt-3 bg-clinical-900 text-white px-4 py-2 rounded-xl font-bold hover:bg-black transition-all flex items-center gap-2 text-xs w-full justify-center">
                      <Stethoscope className="w-4 h-4" /> Analyze Clinical Case
                    </button>
                  )}

                  <button onClick={() => setCropResult(null)} className="text-[10px] text-slate-400 hover:text-slate-600 underline mt-2">Dismiss</button>
                </div>
              )}
              {cropResult?.error && (
                <div className="mt-2 bg-rose-50 border border-rose-200 rounded-lg p-2 text-xs text-rose-700">
                  ❌ {cropResult.error}
                  <button onClick={() => setCropResult(null)} className="ml-2 underline">Dismiss</button>
                </div>
              )}
            </div>
          )}

          {/* Known Cows Grid */}
          <div className="vet-card p-4 animate-enter animate-enter-delay-2">
            <p className="section-label"><User className="w-3 h-3" /> Known Cows (16)</p>
            <div className="grid grid-cols-8 gap-1.5">
              {Array.from({ length: 16 }, (_, i) => i + 1).map(id => (
                <button key={id} onClick={() => setCowId(id)}
                  className={cn("w-full aspect-square rounded-lg text-[10px] font-black transition-all",
                    result?.cow_id === id ? "text-white scale-110 shadow" : cowId == id ? "text-white" : "text-slate-600 hover:scale-105")}
                  style={result?.cow_id === id ? {background:'linear-gradient(135deg,#2a9d5c,#1d7f49)'}
                    : cowId == id ? {background:'#0d3f24'}
                    : {background:'rgba(42,157,92,0.07)', border:'1px solid rgba(42,157,92,0.15)'}}>
                  {id}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* ══ CENTER: RESULTS ══ */}
        <section className="lg:col-span-6 space-y-5 animate-enter animate-enter-delay-1">
          <AnimatePresence mode="wait">
            {/* Empty state */}
            {!loading && !result && (
              <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="hero-empty h-[520px] rounded-3xl flex flex-col items-center justify-center gap-6 p-8 text-center">
                <div className="cow-float">
                  <span className="text-8xl select-none">🐄</span>
                </div>
                <div>
                  <p className="text-lg font-black" style={{fontFamily:'Outfit,sans-serif',color:'#0d3f24'}}>Upload a cattle image to begin</p>
                  <p className="text-sm mt-1" style={{color:'#5d8f8b'}}>5 AI models · 16 known cows · 14-day MMCOWS dataset</p>
                </div>
                <div className="flex gap-3 text-xs">
                  {['Re-ID','Milk Yield','Heat Stress','Health Score','Anomaly'].map(m => (
                    <span key={m} className="px-2.5 py-1 rounded-full font-semibold" style={{background:'rgba(42,157,92,0.10)',color:'#1d7f49',border:'1px solid rgba(42,157,92,0.2)'}}>{m}</span>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Loading */}
            {loading && (
              <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="pipeline-loader h-[520px] rounded-3xl flex flex-col items-center justify-center p-8 text-center border" style={{borderColor:'rgba(42,157,92,0.15)'}}>
                <div className="relative mb-6">
                  <div className="absolute inset-0 rounded-full" style={{background:'rgba(42,157,92,0.15)',filter:'blur(20px)',transform:'scale(1.8)'}} />
                  <div className="cow-float"><span className="text-6xl">🐄</span></div>
                </div>
                <h2 className="text-xl font-black mb-1" style={{fontFamily:'Outfit,sans-serif',color:'#0d3f24'}}>Running MMCOWS Pipeline</h2>
                <p className="text-xs mb-5" style={{color:'#5d8f8b'}}>AI is analyzing your cattle image…</p>
                <div className="h-2 w-64 rounded-full overflow-hidden mb-4" style={{background:'rgba(42,157,92,0.12)'}}>
                  <motion.div initial={{ width: '0%' }} animate={{ width: `${((loadingStep+1)/LOADING_STEPS.length)*100}%` }}
                    className="h-full rounded-full progress-shimmer" />
                </div>
                <AnimatePresence mode="wait">
                  <motion.p key={loadingStep} initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }}
                    className="text-xs font-bold" style={{color:'#2a9d5c'}}>{LOADING_STEPS[loadingStep]}</motion.p>
                </AnimatePresence>
              </motion.div>
            )}

            {/* Results */}
            {result && (
              <motion.div key="results" initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">

                {/* Gate Status */}
                {isRejected ? (
                  <div className="gate-rejected flex items-center gap-4">
                    <ShieldAlert className="w-10 h-10 shrink-0 opacity-80" />
                    <div>
                      <p className="text-xs font-black uppercase tracking-widest mb-1">Pipeline Rejected</p>
                      <p className="text-sm font-bold">{gate.reason}</p>
                      <p className="text-xs opacity-80 mt-1">{gate.action}</p>
                    </div>
                  </div>
                ) : gate?.status === "PASSED" && (
                  <div className="gate-passed flex items-center gap-3">
                    <CheckCircle2 className="w-5 h-5" />
                    <span className="text-sm font-bold">Pipeline PASSED — Cow #{result.cow_id} identified</span>
                    <span className="text-xs opacity-80 ml-auto">{result.total_latency_ms?.toFixed(0)}ms</span>
                  </div>
                )}

                {/* Annotated Image */}
                {result.stages?.annotated_image_b64 && (
                  <div className="glass-card p-3" style={{ cursor: isCropping ? 'crosshair' : 'default' }}>
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2 flex items-center gap-1">
                      <Eye className="w-3 h-3" /> Annotated Detection
                    </p>
                    <div ref={imageWrapperRef} 
                         onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}
                         style={{ position: 'relative', display: 'inline-block', width: '100%' }}>
                        <img src={`data:image/jpeg;base64,${result.stages.annotated_image_b64}`} alt="Annotated"
                          className="w-full rounded-lg" draggable={false} />
                        {cropRect && (
                          <div className="crop-overlay absolute border-2 border-sky-500 bg-sky-500/20" 
                               style={{ left: cropRect.x, top: cropRect.y, width: cropRect.width, height: cropRect.height, position: 'absolute' }} />
                        )}
                    </div>
                  </div>
                )}

                {/* Multi-Cow Model Results Grid */}
                {!isRejected && summaries.map((cowSummary, idx) => {
                  const hsVal = cowSummary.health?.health_score;
                  const riskLevel = cowSummary.health?.risk_level || 'unknown';
                  const myVal = cowSummary.milk?.predicted_yield_kg;
                  const heatLevel = cowSummary.heat_stress?.heat_stress_level || 'unknown';
                  const anomaly = cowSummary.health?.anomaly_detected || false;
                  const hsNum = typeof hsVal === 'number' ? hsVal : (hsVal === 'Insufficient Data' ? 0.5 : parseFloat(hsVal) || 0.5);
                  const riskColor = riskLevel === 'high' ? '#dc2626' : riskLevel === 'medium' ? '#d97706' : '#2a9d5c';
                  const riskBg   = riskLevel === 'high' ? 'rgba(220,38,38,0.08)' : riskLevel === 'medium' ? 'rgba(217,119,6,0.08)' : 'rgba(42,157,92,0.08)';

                  return (
                    <motion.div key={idx} initial={{opacity:0,y:12}} animate={{opacity:1,y:0}} transition={{delay:idx*0.07}}
                      className="cow-card" style={{borderLeft:`4px solid ${riskColor}`}}>
                      {/* Cow card header */}
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <span className="text-2xl">🐄</span>
                          <h4 className="font-black text-lg" style={{fontFamily:'Outfit,sans-serif',color:'#0d3f24'}}>Cow #{cowSummary.cow_id}</h4>
                        </div>
                        <div className="flex items-center gap-2">
                          <button onClick={() => handleRefineCow(cowSummary.cow_id)}
                            className="btn-dark text-xs px-3 py-1.5">
                            <Stethoscope className="w-3 h-3" /> Analyze
                          </button>
                          <span className="text-[10px] font-bold px-2 py-1 rounded-full" style={{background:riskBg,color:riskColor,border:`1px solid ${riskColor}30`}}>
                            {riskLevel.toUpperCase()} RISK
                          </span>
                        </div>
                      </div>
                      {/* 4 metric chips */}
                      <div className="grid grid-cols-4 gap-2 mb-3">
                        <div className="metric-chip">
                          <Heart className="w-3.5 h-3.5 mb-0.5" style={{color:'#10b981'}} />
                          <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Health</span>
                          <span className={cn('text-base font-black', hsVal?.includes?.('Provide') ? 'text-[9px] text-slate-400 leading-tight' : hsNum>0.6?'text-emerald-600':hsNum>0.3?'text-amber-600':'text-rose-600')}>
                            {typeof hsVal==='string'?hsVal:`${(hsVal*100).toFixed(0)}%`}
                          </span>
                        </div>
                        <div className="metric-chip">
                          <Droplets className="w-3.5 h-3.5 mb-0.5" style={{color:'#0ea5e9'}} />
                          <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Milk</span>
                          <span className={cn('text-sm font-black text-sky-600',myVal?.includes?.('Provide')?'text-[9px] text-slate-400 leading-tight':'')}>
                            {myVal?`${myVal}${typeof myVal==='number'?'kg':''}`:'N/A'}
                          </span>
                        </div>
                        <div className="metric-chip">
                          <Sun className="w-3.5 h-3.5 mb-0.5" style={{color:'#f59e0b'}} />
                          <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Heat</span>
                          <span className={cn('text-sm font-black capitalize',heatLevel?.includes?.('Provide')?'text-[9px] text-slate-400 leading-tight normal-case':heatLevel==='no_stress'?'text-emerald-600':heatLevel==='severe'?'text-rose-600':'text-amber-600')}>
                            {heatLevel.replace('_',' ')}
                          </span>
                        </div>
                        <div className={cn('metric-chip', anomaly && 'border-rose-300')} style={anomaly?{background:'rgba(244,63,94,0.06)'}:{}}>
                          <Zap className="w-3.5 h-3.5 mb-0.5" style={{color:anomaly?'#f43f5e':'#7c3aed'}} />
                          <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400">Anomaly</span>
                          <span className={cn('text-sm font-black',anomaly?'text-rose-600':'text-emerald-600')}>
                            {anomaly?'DETECTED':'Normal'}
                          </span>
                        </div>
                      </div>

                      {/* Per-Cow Clinical Report */}
                      {result.stages?.report?.per_cow_reports?.[cowSummary.cow_id] && (
                        <div className="mt-3 p-4 rounded-xl" style={{background:'rgba(42,157,92,0.04)',border:'1px solid rgba(42,157,92,0.12)'}}>
                          <p className="section-label mb-2"><FileText className="w-3 h-3" /> Clinical Case Assessment</p>
                          <div className="text-xs leading-relaxed max-h-48 overflow-y-auto vet-scroll report-prose" style={{color:'#1a2e1e'}}>
                            <ReactMarkdown>{result.stages.report.per_cow_reports[cowSummary.cow_id]}</ReactMarkdown>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  );
                })}

                {/* IDSS Full Report Summary */}
                {result.stages?.report?.full_report && (
                  <div className="vet-card overflow-hidden">
                    <div className="px-4 py-3 border-b flex items-center justify-between" style={{borderColor:'rgba(42,157,92,0.12)',background:'rgba(42,157,92,0.04)'}}>
                      <p className="section-label mb-0"><Activity className="w-3 h-3" /> Herd Intelligence Summary</p>
                      <span className="text-[10px] font-bold" style={{color:'#5d8f8b'}}>Llama 4 Scout</span>
                    </div>
                    <div className="p-5 max-h-[400px] overflow-y-auto vet-scroll">
                      <div className="prose prose-sm max-w-none text-xs report-prose">
                        <ReactMarkdown>{result.stages.report.full_report.split('## INDIVIDUAL CASE ASSESSMENTS')[0]}</ReactMarkdown>
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </section>

        {/* ══ RIGHT: AI AGENT PANEL ══ */}
        <section className="lg:col-span-3 animate-enter animate-enter-delay-3">
          <div className="vet-card h-[calc(100vh-100px)] sticky top-[72px] flex flex-col">
            {/* Agent Header */}
            <div className="p-4 text-white rounded-t-2xl" style={{background:'linear-gradient(135deg,#0d3f24,#2a9d5c)'}}>
              <div className="flex items-center gap-2">
                <div className="bg-white/20 p-1.5 rounded-lg">
                  <MessageSquare className="w-4 h-4 text-white" />
                </div>
                <div>
                  <h4 className="font-black text-sm" style={{fontFamily:'Outfit,sans-serif'}}>Veterinary AI Assistant</h4>
                  <p className="text-[10px] text-emerald-200 font-medium">
                    Clinical &amp; Diagnostic Agent
                    {isRejected && <span className="text-rose-300 ml-1">• BLOCKED</span>}
                  </p>
                </div>
                <span className="ml-auto text-lg">🐄</span>
              </div>
            </div>

            {/* Rejection banner */}
            {isRejected && (
              <div className="bg-rose-50 border-b border-rose-200 px-3 py-2 flex items-center gap-2 text-rose-700 text-[10px] font-bold">
                <ShieldAlert className="w-3 h-3" /> Chat disabled — pipeline rejected
              </div>
            )}

            {/* Messages area */}
            <div className={cn("agent-messages vet-scroll", isRejected && 'opacity-60')} style={{background:'#f9fdf9'}}>
              {currentMessages.map((msg, i) => (
                <motion.div key={i} initial={{opacity:0,y:6}} animate={{opacity:1,y:0}} transition={{duration:0.2}}
                  className={cn('flex flex-col max-w-[92%]', msg.role==='user' ? 'ml-auto items-end' : 'mr-auto items-start')}>
                  <div className={cn('p-2.5 text-xs leading-relaxed', msg.role==='user' ? 'chat-user-bubble' : 'chat-ai-bubble')}>
                    {msg.content.split('\n').map((line, i) => {
                      let formatted = line;
                      // Bold
                      formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong class="font-black">$1</strong>');
                      
                      // Check for specific UI elements
                      if (formatted.includes('[SHOW_IMAGE:')) {
                        const match = formatted.match(/\[SHOW_IMAGE:\s*(.*?)\]/);
                        if (match) {
                          const src = match[1].startsWith('http') ? match[1] : `http://localhost:8001${match[1]}`;
                          return <img key={i} src={src} alt="Agent Output" className="mt-2 rounded-lg max-w-full" />;
                        }
                      }

                      if (formatted.includes('[USER_IMAGE:')) {
                        const match = formatted.match(/\[USER_IMAGE:\s*(.*?)\]/);
                        if (match) {
                          return <img key={i} src={match[1]} alt="User Upload" className="mt-2 rounded-lg max-w-full border border-white/20" />;
                        }
                      }

                      // Header
                      const headerMatch = formatted.match(/^(#{1,6})\s+(.*)/);
                      if (headerMatch) {
                        return <h4 key={i} className="font-bold text-sm mt-2 mb-1 border-b text-clinical-900" dangerouslySetInnerHTML={{ __html: headerMatch[2] }} />;
                      }
                      // Bullet
                      if (formatted.startsWith('* ')) {
                        return <div key={i} className="flex gap-2 ml-2 mb-0.5">
                          <span>•</span>
                          <span dangerouslySetInnerHTML={{ __html: formatted.substring(2) }} />
                        </div>;
                      }
                      return <p key={i} className="mb-1" dangerouslySetInnerHTML={{ __html: formatted }} />;
                    })}
                  </div>
                </motion.div>
              ))}
              {isTyping && (
                <div className="flex gap-1 p-2.5 rounded-2xl w-fit" style={{background:'white',border:'1px solid rgba(42,157,92,0.12)',boxShadow:'0 2px 8px rgba(13,63,36,0.06)'}}>
                  <div className="typing-dot" /><div className="typing-dot" /><div className="typing-dot" />
                </div>
              )}
              <div ref={currentEndRef} />
            </div>

            {/* Chat image preview */}
            {chatPreview && (
              <div className="px-3 py-2 flex items-center gap-2 border-t" style={{background:'rgba(42,157,92,0.04)',borderColor:'rgba(42,157,92,0.12)'}}>
                <div className="relative">
                  <img src={chatPreview} alt="Chat Preview" className="w-10 h-10 rounded-xl object-cover" style={{border:'2px solid rgba(42,157,92,0.3)'}} />
                  <button onClick={()=>{setChatImage(null);setChatPreview(null);}} className="absolute -top-1 -right-1 bg-rose-500 text-white rounded-full p-0.5"><X className="w-2 h-2" /></button>
                </div>
                <p className="text-[10px] italic" style={{color:'#5d8f8b'}}>Image attached for clinical vision analysis...</p>
              </div>
            )}
            <form onSubmit={handleChatSubmit} className="p-3 flex items-center gap-2 border-t" style={{background:'white',borderColor:'rgba(42,157,92,0.12)'}}>
              <label className="cursor-pointer transition-colors" style={{color:'#5d8f8b'}}>
                <input type="file" className="hidden" accept="image/*" onChange={(e)=>{
                  const f=e.target.files[0];
                  if(f){setChatImage(f);setChatPreview(URL.createObjectURL(f));}
                }} />
                <Upload className="w-4 h-4" />
              </label>
              <input value={chatInput} onChange={(e)=>setChatInput(e.target.value)}
                disabled={isRejected}
                placeholder={isRejected ? 'Chat disabled...' : 'Ask for diagnostics...'}
                className="flex-1 px-3 py-2 rounded-xl text-xs outline-none" style={{background:'rgba(244,248,245,0.8)',border:'1.5px solid rgba(42,157,92,0.15)'}} />
              <button type="submit" disabled={isTyping||(!chatInput.trim()&&!chatImage)}
                className="p-2 rounded-xl text-white transition-all disabled:opacity-40"
                style={{background:'linear-gradient(135deg,#2a9d5c,#1d7f49)'}}>
                <Send className="w-4 h-4" />
              </button>
            </form>
          </div>
        </section>
      </main>

      {/* REFINEMENT MODAL */}
      <AnimatePresence>
        {showRefinementModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}
              onClick={()=>setShowRefinementModal(false)} className="absolute inset-0 backdrop-blur-sm" style={{background:'rgba(13,63,36,0.5)'}} />
            <motion.div initial={{opacity:0,scale:0.95,y:20}} animate={{opacity:1,scale:1,y:0}} exit={{opacity:0,scale:0.95,y:20}}
              className="relative bg-white rounded-3xl shadow-2xl w-full max-w-md overflow-hidden">
              <div className="p-6 text-white" style={{background:'linear-gradient(135deg,#0d3f24,#2a9d5c)'}}>
                <div className="flex justify-between items-center mb-2">
                  <h3 className="text-xl font-black flex items-center gap-2" style={{fontFamily:'Outfit,sans-serif'}}><Stethoscope className="w-5 h-5" /> Clinical Case Entry</h3>
                  <button onClick={()=>setShowRefinementModal(false)} className="text-white/60 hover:text-white"><X /></button>
                </div>
                <p className="text-sm text-white/80">Enter clinical parameters for Cow #{selectedCowId} to refine diagnostic accuracy.</p>
              </div>
              <div className="p-6 space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-wider mb-1" style={{color:'#5d8f8b'}}>Body Weight (kg)</label>
                    <div className="relative">
                      <Weight className="absolute left-3 top-2.5 w-4 h-4" style={{color:'#5d8f8b'}} />
                      <input type="number" value={weight} onChange={(e)=>setWeight(e.target.value)}
                        className="vet-input pl-9" />
                    </div>
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-wider mb-1" style={{color:'#5d8f8b'}}>Animal Age (years)</label>
                    <div className="relative">
                      <Calendar className="absolute left-3 top-2.5 w-4 h-4" style={{color:'#5d8f8b'}} />
                      <input type="number" value={age} onChange={(e)=>setAge(e.target.value)}
                        className="vet-input pl-9" />
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-wider mb-1" style={{color:'#5d8f8b'}}>Body Temp (°C)</label>
                    <div className="relative">
                      <Thermometer className="absolute left-3 top-2.5 w-4 h-4" style={{color:'#5d8f8b'}} />
                      <input type="number" step="0.1" value={temperature} onChange={(e)=>setTemperature(e.target.value)}
                        className="vet-input pl-9" />
                    </div>
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-wider mb-1" style={{color:'#5d8f8b'}}>Heart Rate (bpm)</label>
                    <div className="relative">
                      <Activity className="absolute left-3 top-2.5 w-4 h-4" style={{color:'#5d8f8b'}} />
                      <input type="number" value={heartRate} onChange={(e)=>setHeartRate(e.target.value)}
                        className="vet-input pl-9" />
                    </div>
                  </div>
                </div>
                <button onClick={submitRefinement} disabled={refinementLoading} className="btn-primary w-full py-3">
                  {refinementLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <><Zap className="w-4 h-4" /> Run Accurate IDSS Analysis</>}
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
