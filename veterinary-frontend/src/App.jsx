import React, { useState, useEffect } from 'react';
import { 
  Upload, 
  Activity, 
  ShieldAlert, 
  User, 
  Thermometer, 
  FileText, 
  CheckCircle2, 
  Loader2,
  AlertTriangle,
  Weight,
  FlaskConical,
  Stethoscope,
  MessageSquare,
  ChevronDown,
  ChevronUp,
  Database,
  Send,
  X
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { submitPrediction, checkHealth, clinicalQA } from './api';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Helper for tailwind classes
function cn(...inputs) {
  return twMerge(clsx(inputs));
}

const LOADING_STEPS = [
  "Initializing multi-modal pipeline...",
  "Running YOLO identity engine...",
  "Extracting ViT embeddings...",
  "Analyzing clinical signs with MaxViT...",
  "Querying Neo4j Knowledge Graph...",
  "Retrieving PubMed evidence...",
  "Synthesizing clinical report with Llama 3.3..."
];

export default function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [weight, setWeight] = useState(600);
  const [age, setAge] = useState("");
  const [cowId, setCowId] = useState("");
  const [bodyTemp, setBodyTemp] = useState("");
  const [heartRate, setHeartRate] = useState("");
  
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiOnline, setApiOnline] = useState(true);

  // Chat State
  const [chatOpen, setChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', content: "Hello! I'm your Veterinary AI Agent. How can I help you with this case today?" }
  ]);
  const [isTyping, setIsTyping] = useState(false);

  // Check backend health on mount
  useEffect(() => {
    checkHealth().then(setApiOnline);
  }, []);

  // Handle file selection
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!image) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);
    setLoadingStep(0);

    // Simulated step progression
    const stepInterval = setInterval(() => {
      setLoadingStep(prev => (prev < LOADING_STEPS.length - 1 ? prev + 1 : prev));
    }, 850);

    try {
      const sensorObj = {};
      if (bodyTemp) sensorObj.body_temp = parseFloat(bodyTemp);
      if (heartRate) sensorObj.heart_rate = parseFloat(heartRate);
      
      const computedSensorJson = Object.keys(sensorObj).length > 0 ? JSON.stringify(sensorObj) : "{}";
      
      const data = await submitPrediction(image, weight, age, cowId, computedSensorJson);
      setResult(data);
      // Reset chat messages when new result arrives
      setChatMessages([
        { role: 'assistant', content: `Diagnosis complete for Cow #${data.cow_id}. How can I assist you with this report?` }
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      clearInterval(stepInterval);
      setLoading(false);
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMsg = { role: 'user', content: chatInput };
    const currentMessages = [...chatMessages, userMsg];
    setChatMessages(currentMessages);
    setChatInput("");
    setIsTyping(true);

    try {
      const context = result ? `Primary Finding: ${result.stages.clinical_summary.primary_finding}. Risk Level: ${result.stages.clinical_summary.risk_level}.` : "";
      const data = await clinicalQA(userMsg.content, context);
      setChatMessages([...currentMessages, { role: 'assistant', content: data.answer }]);
    } catch (err) {
      setChatMessages([...currentMessages, { role: 'assistant', content: "I'm sorry, I'm having trouble connecting to the clinical engine right now." }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="min-h-screen pb-12">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 py-4 px-6 sticky top-0 z-10 backdrop-blur-sm bg-white/90">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="bg-clinical-900 p-2 rounded-lg">
              <Activity className="text-white w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900 tracking-tight">VETAI IDSS</h1>
              <p className="text-xs text-slate-500 font-medium uppercase tracking-widest">Advanced Decision Support</p>
            </div>
          </div>
          
          <div className={cn(
            "flex items-center gap-2 text-xs font-semibold px-3 py-1 rounded-full",
            apiOnline ? "bg-emerald-50 text-emerald-700 border border-emerald-100" : "bg-rose-50 text-rose-700 border border-rose-100"
          )}>
            <div className={cn("w-2 h-2 rounded-full", apiOnline ? "bg-emerald-500" : "bg-rose-500 animate-pulse")} />
            {apiOnline ? "SYSTEM ONLINE" : "OFFLINE"}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 pt-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* LEFT COLUMN: UPLOAD & INPUTS */}
        <section className="lg:col-span-4 space-y-6">
          <div className="glass-card p-6">
            <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
              <Upload className="w-5 h-5 text-clinical-500" />
              Analyze Case
            </h2>
            
            <div 
              className={cn(
                "border-2 border-dashed rounded-xl p-8 transition-all flex flex-col items-center justify-center text-center gap-4 cursor-pointer",
                preview ? "border-clinical-200 bg-slate-50" : "border-slate-300 hover:border-clinical-400 hover:bg-slate-50"
              )}
              onClick={() => document.getElementById('fileInput').click()}
            >
              {preview ? (
                <img src={preview} alt="Cattle Preview" className="w-full h-48 object-cover rounded-lg shadow-sm" />
              ) : (
                <>
                  <div className="bg-slate-100 p-4 rounded-full">
                    <Upload className="w-8 h-8 text-slate-400" />
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-semibold">Drop cattle image here</p>
                    <p className="text-xs text-slate-500">Supports JPEG, PNG up to 10MB</p>
                  </div>
                </>
              )}
              <input 
                id="fileInput"
                type="file" 
                className="hidden" 
                accept="image/*"
                onChange={handleFileChange}
              />
            </div>

            <div className="mt-6 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5 block">Animal Weight (kg)</label>
                  <div className="relative">
                    <Weight className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                    <input 
                      type="number"
                      value={weight}
                      onChange={(e) => setWeight(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-clinical-500 outline-none transition-all font-semibold"
                      placeholder="e.g. 600"
                    />
                  </div>
                </div>
                <div>
                  <label className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5 block">Animal Age (Years)</label>
                  <div className="relative">
                    <Activity className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                    <input 
                      type="number"
                      step="0.1"
                      value={age}
                      onChange={(e) => setAge(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-clinical-500 outline-none transition-all font-semibold"
                      placeholder="e.g. 2.5"
                    />
                  </div>
                </div>
              </div>

              <div>
                <label className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5 block">Cow ID (Optional Override)</label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                  <input 
                    type="number"
                    value={cowId}
                    onChange={(e) => setCowId(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-clinical-500 outline-none transition-all font-semibold"
                    placeholder="e.g. 101 (leave blank for auto)"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5 block">Body Temp (°C)</label>
                  <div className="relative">
                    <Thermometer className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                    <input 
                      type="number"
                      step="0.1"
                      value={bodyTemp}
                      onChange={(e) => setBodyTemp(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-clinical-500 outline-none transition-all font-semibold"
                      placeholder="e.g. 38.5"
                    />
                  </div>
                </div>
                <div>
                  <label className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5 block">Heart Rate (bpm)</label>
                  <div className="relative">
                    <Activity className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                    <input 
                      type="number"
                      value={heartRate}
                      onChange={(e) => setHeartRate(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-clinical-500 outline-none transition-all font-semibold"
                      placeholder="e.g. 62"
                    />
                  </div>
                </div>
              </div>

              <button
                onClick={handleSubmit}
                disabled={loading || !image}
                className={cn(
                  "w-full py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all shadow-md active:scale-[0.98]",
                  loading || !image 
                    ? "bg-slate-100 text-slate-400 cursor-not-allowed" 
                    : "bg-clinical-900 text-white hover:bg-black"
                )}
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <>
                    <Activity className="w-5 h-5" />
                    Run Diagnosis
                  </>
                )}
              </button>
              
              {error && (
                <div className="bg-rose-50 text-rose-700 p-3 rounded-lg text-sm border border-rose-100 flex gap-2">
                  <AlertTriangle className="w-5 h-5 shrink-0" />
                  {error}
                </div>
              )}
            </div>
          </div>

          <div className="glass-card p-6 bg-slate-900 text-white border-none">
            <h3 className="font-bold flex items-center gap-2 mb-2 text-emerald-400">
              <CheckCircle2 className="w-4 h-4" />
              Expert System Specs
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Utilizing a multi-agent orchestrated pipeline featuring MaxViT-B Classifier, Llama-3.2 Vision, and Llama-3.3 70B clinical logic.
            </p>
          </div>
        </section>

        {/* RIGHT COLUMN: LOADING & RESULTS */}
        <section className="lg:col-span-8">
          <AnimatePresence mode="wait">
            {!loading && !result && (
              <motion.div 
                key="empty"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="h-[600px] border-2 border-dashed border-slate-200 rounded-2xl flex flex-col items-center justify-center text-slate-400 gap-4"
              >
                <Stethoscope className="w-16 h-16 opacity-20" />
                <p className="text-lg font-medium">Capture or upload an image to begin clinical analysis</p>
              </motion.div>
            )}

            {loading && (
              <motion.div 
                key="loading"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="glass-card h-[600px] flex flex-col items-center justify-center p-12 text-center"
              >
                <div className="relative mb-8">
                   <div className="absolute inset-0 bg-clinical-500/20 blur-2xl rounded-full scale-150 animate-pulse" />
                   <div className="bg-clinical-900 p-6 rounded-3xl relative">
                     <Loader2 className="w-12 h-12 text-white animate-spin" />
                   </div>
                </div>
                <h2 className="text-2xl font-bold mb-2">Analyzing Cattle IDSS...</h2>
                <div className="h-4 w-64 bg-slate-100 rounded-full overflow-hidden mb-6">
                  <motion.div 
                    initial={{ width: "0%" }}
                    animate={{ width: `${((loadingStep + 1) / LOADING_STEPS.length) * 100}%` }}
                    className="h-full bg-clinical-900"
                  />
                </div>
                <AnimatePresence mode="wait">
                  <motion.p 
                    key={loadingStep}
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -5 }}
                    className="text-clinical-600 font-bold"
                  >
                    {LOADING_STEPS[loadingStep]}
                  </motion.p>
                </AnimatePresence>
              </motion.div>
            )}

            {result && (
              <motion.div 
                key="results"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                {/* SAFETY BANNER */}
                {(result.stages.disease.safety.safety_level === 'blocked' || result.stages.report.urgency_score >= 8) && (
                  <motion.div 
                    initial={{ x: -20, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    className="bg-rose-600 text-white p-5 rounded-xl shadow-lg shadow-rose-200 flex items-center gap-5 relative overflow-hidden"
                  >
                    <div className="absolute top-0 right-0 p-4 opacity-10">
                      <ShieldAlert className="w-24 h-24" />
                    </div>
                    <div className="bg-white/20 p-3 rounded-xl animate-pulse">
                      <ShieldAlert className="w-8 h-8" />
                    </div>
                    <div className="relative z-10">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-black uppercase tracking-widest bg-white/20 px-2 py-0.5 rounded">Critical Alert</span>
                        <span className="text-xs font-bold">Urgency: {result.stages.report.urgency_score}/10</span>
                      </div>
                      <h3 className="text-lg font-black uppercase leading-tight">
                        {result.stages.disease.safety.safety_level === 'blocked' ? "BIOSECURITY BLOCK: HIGH RISK SUSPECTED" : "IMMEDIATE VETERINARY ACTION REQUIRED"}
                      </h3>
                      <div className="mt-2 text-sm text-rose-100">
                        {result.stages.disease.safety.safety_flags.map((flag, i) => (
                           <div key={i} className="flex gap-2 items-start py-0.5">
                             <div className="w-1 h-1 rounded-full bg-white mt-2 shrink-0" />
                             {flag}
                           </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* IDENTITY CARD */}
                  <div className="glass-card p-6 health-border-healthy">
                    <div className="flex justify-between items-start mb-6">
                      <div className="flex items-center gap-3">
                        <div className="bg-slate-100 p-2.5 rounded-xl">
                          <User className="w-5 h-5 text-slate-600" />
                        </div>
                        <div>
                          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-none mb-1">Cattle Identity</p>
                          <h3 className="text-xl font-black text-slate-900">COW #{result.stages.identity.cow_id}</h3>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Confidence</p>
                        <p className="text-lg font-black text-emerald-600">{Math.round(result.stages.identity.confidence * 100)}%</p>
                      </div>
                    </div>
                    
                    <div className="space-y-4">
                      <div className="flex justify-between text-xs font-bold">
                        <span className="text-slate-500">Recognition Method</span>
                        <span className="text-slate-900">YOLO-ViT Hybrid</span>
                      </div>
                      <div className="flex justify-between text-xs font-bold">
                        <span className="text-slate-500">History Status</span>
                        <span className="text-emerald-600">Active Records Found</span>
                      </div>
                    </div>
                  </div>

                  {/* DISEASE PROBABILITIES */}
                  <div className="glass-card p-6 health-border-warning">
                    <h3 className="text-xs font-black text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                       <FlaskConical className="w-4 h-4" />
                       Differential Diagnosis
                    </h3>
                    <div className="space-y-5">
                      {result.stages.disease.predictions.slice(0, 3).map((pred, i) => (
                        <div key={i} className="space-y-1.5">
                          <div className="flex justify-between text-xs font-black">
                            <span className="capitalize">{pred.disease.replace(/_/g, ' ')}</span>
                            <span className={cn(
                              pred.confidence > 0.7 ? "text-rose-600" : "text-amber-600"
                            )}>
                              {Math.round(pred.confidence * 100)}%
                            </span>
                          </div>
                          <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                            <motion.div 
                              initial={{ width: 0 }}
                              animate={{ width: `${pred.confidence * 100}%` }}
                              transition={{ duration: 1, delay: 0.2 * i }}
                              className={cn(
                                "h-full rounded-full",
                                pred.confidence > 0.7 ? "bg-health-rose" : "bg-health-amber"
                              )}
                            ></motion.div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* CLINICAL REPORT */}
                <div className="bg-white rounded-2xl overflow-hidden shadow-xl border border-slate-200">
                  <div className="bg-slate-50 p-4 border-b border-slate-100 flex items-center justify-between">
                    <div className="flex items-center gap-3 text-emerald-600">
                      <FileText className="w-5 h-5" />
                      <span className="text-xs font-black uppercase tracking-widest">Master IDSS Clinical Report</span>
                    </div>
                    <div className="text-[10px] font-bold text-slate-400">
                      Pipeline v {result.pipeline_version} | Llama-3.3 70B
                    </div>
                  </div>
                  <div className="p-8 prose-custom max-h-[1000px] overflow-y-auto bg-white">
                    <div className="prose prose-slate max-w-none 
                      prose-headings:font-black prose-headings:tracking-tight
                      prose-p:text-slate-600 prose-p:leading-relaxed
                      prose-strong:text-slate-900 prose-strong:font-black
                      prose-li:text-slate-600
                    ">
                      <ReactMarkdown
                        components={{
                          h1: ({node, ...props}) => <h1 className="text-slate-900 border-b border-slate-200 pb-2 mt-12 mb-6" {...props} />,
                          h2: ({node, ...props}) => {
                            const text = props.children?.toString() || "";
                            const color = text.toLowerCase().includes('summary') ? 'text-emerald-700' :
                                         text.toLowerCase().includes('diagnosis') ? 'text-sky-700' :
                                         text.toLowerCase().includes('treatment') ? 'text-violet-700' :
                                         text.toLowerCase().includes('biosecurity') ? 'text-rose-700' :
                                         text.toLowerCase().includes('evidence') ? 'text-indigo-800' : 'text-slate-900';
                            return <h2 className={cn(color, "border-b border-slate-100 pb-2 mt-10 mb-4")} {...props} />
                          },
                          h3: ({node, ...props}) => {
                            const text = props.children?.toString() || "";
                            const color = text.toLowerCase().includes('summary') ? 'text-emerald-700' :
                                         text.toLowerCase().includes('diagnosis') ? 'text-sky-700' :
                                         text.toLowerCase().includes('findings') ? 'text-amber-700' :
                                         text.toLowerCase().includes('treatment') ? 'text-violet-700' :
                                         text.toLowerCase().includes('biosecurity') ? 'text-rose-700' :
                                         text.toLowerCase().includes('disclaimer') ? 'text-slate-500 italic' : 'text-slate-900';
                            return <h3 className={cn(color, "mt-8 mb-3")} {...props} />
                          },
                          strong: ({node, ...props}) => {
                            return <strong className="text-slate-900 font-bold" {...props} />
                          }
                        }}
                      >
                        {result.stages.report.report}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </section>

      </main>

      {result && (
        <footer className="max-w-7xl mx-auto px-6 mt-12 text-center text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-8">
            Diagnostic Confidence: {(result.stages.disease.predictions[0]?.confidence * 100).toFixed(1)}% | 
            Report Severity: {result.stages.report.urgency_score}/10 |
            Process ID: {result.stages.report.report_id}
        </footer>
      )}

      {/* FLOATING CHAT AGENT */}
      <div className="fixed bottom-6 right-6 z-50">
        <AnimatePresence>
          {chatOpen && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="absolute bottom-16 right-0 w-[380px] h-[500px] bg-white rounded-2xl shadow-2xl border border-slate-200 overflow-hidden flex flex-col"
            >
              <div className="bg-clinical-900 p-4 text-white flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="bg-white/20 p-2 rounded-lg">
                    <MessageSquare className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="font-bold text-sm">Veterinary AI Agent</h4>
                    <p className="text-[10px] text-emerald-400 font-medium">Ready for clinical Q&A</p>
                  </div>
                </div>
                <button onClick={() => setChatOpen(false)} className="hover:bg-white/10 p-1 rounded-full transition-colors">
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50">
                {chatMessages.map((msg, i) => (
                  <div key={i} className={cn(
                    "flex flex-col max-w-[85%]",
                    msg.role === 'user' ? "ml-auto items-end" : "mr-auto items-start"
                  )}>
                    <div className={cn(
                      "p-3 rounded-2xl text-sm leading-relaxed",
                      msg.role === 'user' 
                        ? "bg-clinical-900 text-white rounded-br-none" 
                        : "bg-white text-slate-700 shadow-sm border border-slate-100 rounded-bl-none"
                    )}>
                      {msg.content}
                    </div>
                  </div>
                ))}
                {isTyping && (
                  <div className="flex gap-1.5 p-3 bg-white border border-slate-100 rounded-2xl w-fit">
                    <div className="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce" />
                    <div className="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce [animation-delay:0.2s]" />
                    <div className="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce [animation-delay:0.4s]" />
                  </div>
                )}
              </div>

              <form onSubmit={handleChatSubmit} className="p-4 bg-white border-t border-slate-100 gap-2 flex">
                <input 
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask the clinical agent..."
                  className="flex-1 bg-slate-100 px-4 py-2 rounded-xl text-sm outline-none focus:ring-2 focus:ring-clinical-500 transition-all"
                />
                <button 
                  type="submit"
                  disabled={!chatInput.trim() || isTyping}
                  className="bg-clinical-900 text-white p-2 rounded-xl hover:bg-black transition-colors disabled:opacity-50"
                >
                  <Send className="w-5 h-5" />
                </button>
              </form>
            </motion.div>
          )}
        </AnimatePresence>

        <button 
          onClick={() => setChatOpen(!chatOpen)}
          className={cn(
            "p-4 rounded-3xl shadow-xl transition-all duration-300 flex items-center gap-2 group",
            chatOpen ? "bg-white text-slate-900 rotate-90" : "bg-clinical-900 text-white hover:scale-105 active:scale-95"
          )}
        >
          {chatOpen ? <X className="w-6 h-6" /> : (
            <>
              <MessageSquare className="w-6 h-6" />
              {!chatOpen && <span className="max-w-0 overflow-hidden group-hover:max-w-[100px] transition-all duration-500 whitespace-nowrap text-sm font-bold ml-2">Talk to Agent</span>}
            </>
          )}
        </button>
      </div>
    </div>
  );
}
