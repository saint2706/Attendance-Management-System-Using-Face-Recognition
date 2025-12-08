import { useState, useRef, useCallback, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { markAttendance } from '../api/attendance';
import type { RecognitionResult } from '../api/attendance';
import {
    Camera,
    CameraOff,
    Clock,
    CheckCircle,
    XCircle,
    AlertTriangle,
    Loader2,
    RefreshCw
} from 'lucide-react';
import './MarkAttendance.css';

export const MarkAttendance = () => {
    const [searchParams] = useSearchParams();
    const direction = (searchParams.get('direction') as 'in' | 'out') || 'in';

    const [stream, setStream] = useState<MediaStream | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setResult] = useState<RecognitionResult | null>(null);

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Start camera
    const startCamera = useCallback(async () => {
        setError(null);
        setResult(null);

        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                }
            });

            setStream(mediaStream);

            if (videoRef.current) {
                videoRef.current.srcObject = mediaStream;
                await videoRef.current.play();
            }
        } catch (err) {
            console.error('Camera error:', err);
            setError('Unable to access camera. Please ensure camera permissions are granted.');
        }
    }, []);

    // Stop camera
    const stopCamera = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
        }
    }, [stream]);

    // Capture and process
    const captureAndRecognize = async () => {
        if (!videoRef.current || !canvasRef.current) return;

        setIsProcessing(true);
        setResult(null);

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        if (!ctx) return;

        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0);

        // Get base64 image
        const imageBase64 = canvas.toDataURL('image/jpeg', 0.9);

        try {
            const recognitionResult = await markAttendance(imageBase64, direction);
            setResult(recognitionResult);
        } catch (err) {
            console.error('Recognition error:', err);
            setResult({
                recognized: false,
                spoofDetected: false,
                message: 'Recognition failed. Please try again.'
            });
        } finally {
            setIsProcessing(false);
        }
    };

    // Auto-start camera on mount
    useEffect(() => {
        startCamera();
        return () => stopCamera();
    }, []);

    // Reset for another attempt
    const resetAttempt = () => {
        setResult(null);
        setError(null);
        if (!stream) {
            startCamera();
        }
    };

    return (
        <div className="mark-attendance animate-fade-in">
            <div className="attendance-container">
                <header className="attendance-header">
                    <Clock size={32} className="header-icon" />
                    <h1>Mark Time-{direction === 'in' ? 'In' : 'Out'}</h1>
                    <p className="text-muted">
                        Position your face in the camera and click capture
                    </p>
                </header>

                {/* Camera View */}
                <div className="camera-container card card-elevated">
                    {error ? (
                        <div className="camera-error">
                            <CameraOff size={48} />
                            <p>{error}</p>
                            <button onClick={startCamera} className="btn btn-primary">
                                <RefreshCw size={18} />
                                Retry
                            </button>
                        </div>
                    ) : (
                        <>
                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                muted
                                className="camera-video"
                            />
                            <canvas ref={canvasRef} className="hidden-canvas" />

                            {/* Face guide overlay */}
                            <div className="face-guide">
                                <div className="face-oval"></div>
                            </div>
                        </>
                    )}
                </div>

                {/* Result Display */}
                {result && (
                    <div className={`result-card card ${result.recognized ? 'result-success' :
                        result.spoofDetected ? 'result-warning' : 'result-error'
                        }`}>
                        <div className="result-content">
                            {result.recognized ? (
                                <>
                                    <CheckCircle size={32} />
                                    <div>
                                        <h3>Attendance Marked!</h3>
                                        <p>Welcome, {result.username}</p>
                                        {result.confidence && (
                                            <span className="badge badge-success">
                                                Confidence: {(result.confidence * 100).toFixed(1)}%
                                            </span>
                                        )}
                                    </div>
                                </>
                            ) : result.spoofDetected ? (
                                <>
                                    <AlertTriangle size={32} />
                                    <div>
                                        <h3>Liveness Check Failed</h3>
                                        <p>Please try again with your actual face</p>
                                    </div>
                                </>
                            ) : (
                                <>
                                    <XCircle size={32} />
                                    <div>
                                        <h3>Not Recognized</h3>
                                        <p>{result.message}</p>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                )}

                {/* Action Buttons */}
                <div className="attendance-actions">
                    {!result ? (
                        <button
                            onClick={captureAndRecognize}
                            disabled={!stream || isProcessing}
                            className="btn btn-primary btn-lg capture-button"
                        >
                            {isProcessing ? (
                                <>
                                    <Loader2 size={20} className="animate-spin" />
                                    Processing...
                                </>
                            ) : (
                                <>
                                    <Camera size={20} />
                                    Capture & Recognize
                                </>
                            )}
                        </button>
                    ) : (
                        <button
                            onClick={resetAttempt}
                            className="btn btn-secondary btn-lg"
                        >
                            <RefreshCw size={20} />
                            Try Again
                        </button>
                    )}
                </div>

                {/* Instructions */}
                <div className="instructions card">
                    <div className="card-body">
                        <h4>Tips for best results:</h4>
                        <ul>
                            <li>Ensure good lighting on your face</li>
                            <li>Look directly at the camera</li>
                            <li>Remove hats or sunglasses</li>
                            <li>Keep your face within the oval guide</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};
