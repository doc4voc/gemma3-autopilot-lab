import { useState, useEffect } from "react";
import { generateAIReview, downloadHTMLReport } from "../services/analysisService";

export default function AutoAnalyst({ telemetry, isRecording, onStart, onStop, onAnalysisChange }) {
    const [sampleCount, setSampleCount] = useState(0);
    const [status, setStatus] = useState("IDLE"); // IDLE, RECORDING, ANALYZING, DONE
    const [review, setReview] = useState(null);

    // Monitor Telemetry Length directly (no interval reset race).
    useEffect(() => {
        if (!isRecording) return;
        setSampleCount(telemetry ? telemetry.length : 0);
    }, [isRecording, telemetry]);

    const handleStart = () => {
        setStatus("RECORDING");
        setSampleCount(0);
        onStart();
    };

    const handleStopAnalysis = async () => {
        setStatus("ANALYZING");
        onStop(); // Stop the car/recording

        // Notify Parent to Block Driving
        if (onAnalysisChange) onAnalysisChange(true);

        // 1. Get Review
        const history = telemetry;
        if (!history || history.length < 10) {
            alert("Not enough data to analyze.");
            setStatus("IDLE");
            if (onAnalysisChange) onAnalysisChange(false);
            return;
        }

        const aiReview = await generateAIReview(history);
        setReview(aiReview);

        // 2. Generate Report
        downloadHTMLReport(history, aiReview);

        setStatus("DONE");
        if (onAnalysisChange) onAnalysisChange(false);
    };

    if (status === "IDLE") {
        return (
            <button
                onClick={handleStart}
                className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg font-bold shadow-lg border border-indigo-400 flex items-center gap-2"
            >
                <span>🧪</span> Start Data Session
            </button>
        );
    }

    if (status === "RECORDING") {
        return (
            <div className="bg-black/80 p-4 rounded-xl border border-indigo-500 animate-pulse-slow w-64">
                <div className="text-indigo-400 text-xs font-bold uppercase mb-2">Recording Telemetry</div>
                <div className="flex justify-between items-end mb-2">
                    <span className="text-3xl font-mono text-white">{sampleCount}</span>
                    <span className="text-xs text-gray-400">samples</span>
                </div>
                <div className="h-2 bg-gray-700 rounded-full overflow-hidden mb-4">
                    <div className="h-full bg-indigo-500" style={{ width: `${Math.min(100, (sampleCount / 600) * 100)}%` }} />
                </div>
                <button
                    onClick={handleStopAnalysis}
                    className="w-full bg-red-600 hover:bg-red-700 text-white py-2 rounded font-bold"
                >
                    ⏹ Stop & Analyze
                </button>
            </div>
        );
    }

    if (status === "ANALYZING") {
        return (
            <div className="bg-black/90 p-6 rounded-xl border border-indigo-500 w-80 text-center">
                <div className="text-4xl mb-4 animate-bounce">🤖</div>
                <h3 className="text-indigo-400 font-bold text-xl mb-2">Gemma is Thinking...</h3>
                <p className="text-gray-400 text-sm">Reviewing {sampleCount} data points.</p>
                <p className="text-gray-500 text-xs mt-2">Generating HTML Report...</p>
            </div>
        );
    }

    if (status === "DONE") {
        return (
            <div className="bg-indigo-900/90 p-6 rounded-xl border border-indigo-400 w-80 shadow-2xl">
                <div className="flex justify-between items-start mb-4">
                    <h3 className="text-white font-bold text-lg">Analysis Complete</h3>
                    <button onClick={() => { setStatus("IDLE"); if (onAnalysisChange) onAnalysisChange(false); }} className="text-gray-400 hover:text-white">✕</button>
                </div>
                <div className="bg-black/50 p-3 rounded text-sm text-gray-300 mb-4 max-h-40 overflow-y-auto" dangerouslySetInnerHTML={{ __html: review }} />
                <button
                    onClick={() => downloadHTMLReport(telemetry, review)}
                    className="w-full bg-indigo-600 hover:bg-indigo-500 text-white py-2 rounded font-bold mb-2"
                >
                    📥 Download Again
                </button>
            </div>
        );
    }

    return null;
}
