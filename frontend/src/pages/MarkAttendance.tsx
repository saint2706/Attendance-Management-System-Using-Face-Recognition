import { useState, useRef, useCallback, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
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
    RefreshCw,
    Home,
    UserCheck,
    Keyboard
} from 'lucide-react';
import './MarkAttendance.css';

export const MarkAttendance = () => {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const direction = (searchParams.get('direction') as 'in' | 'out') || 'in';

    const [stream, setStream] = useState<MediaStream | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isInitializing, setIsInitializing] = useState(true);
    const [result, setResult] = useState<RecognitionResult | null>(null);
    const [showFlash, setShowFlash] = useState(false);
    const [countdown, setCountdown] = useState<number | null>(null);

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);

    // Start camera
    const startCamera = useCallback(async () => {
        setError(null);
        setResult(null);
        setIsInitializing(true);

        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                }
            });

            setStream(mediaStream);
            streamRef.current = mediaStream;

            if (videoRef.current) {
                videoRef.current.srcObject = mediaStream;
                await videoRef.current.play();
            }
        } catch (err) {
            console.error('Camera error:', err);
            setError('Unable to access camera. Please ensure camera permissions are granted.');
        } finally {
            setIsInitializing(false);
        }
    }, []);

    // Handle flash effect timer
    useEffect(() => {
        if (showFlash) {
            const timer = setTimeout(() => setShowFlash(false), 150);
            return () => clearTimeout(timer);
        }
    }, [showFlash]);

    // Capture and process
    const captureAndRecognize = useCallback(async () => {
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

        // Trigger flash effect after capture
        setShowFlash(true);

        // Freeze video to show captured frame
        video.pause();

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
    }, [direction]);

    // Handle countdown logic
    useEffect(() => {
        if (countdown === null) return;

        if (countdown > 0) {
            const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
            return () => clearTimeout(timer);
        } else if (countdown === 0) {
            captureAndRecognize();
            setCountdown(null);
        }
    }, [countdown, captureAndRecognize]);

    const startCaptureSequence = useCallback(() => {
        if (!stream || isProcessing || countdown !== null) return;
        setCountdown(3);
    }, [stream, isProcessing, countdown]);

    // Reset for another attempt
    const resetAttempt = useCallback(() => {
        setResult(null);
        setError(null);
        setCountdown(null);
        if (!stream) {
            startCamera();
        } else if (videoRef.current) {
            videoRef.current.play().catch(console.error);
        }
    }, [stream, startCamera]);

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Only trigger if we are in the capture state (no result yet)
            if (!result && !isProcessing && countdown === null) {
                if (e.code === 'Space') {
                    e.preventDefault(); // Prevent scrolling
                    startCaptureSequence();
                }
            } else if (result) {
                if (e.code === 'Escape') {
                    resetAttempt();
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [result, isProcessing, countdown, startCaptureSequence, resetAttempt]);

    // Auto-start camera on mount (only once)
    useEffect(() => {
        startCamera();
        return () => {
            // Use ref to access current stream value for cleanup
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
                streamRef.current = null;
            }
        };
        // startCamera is intentionally omitted from dependencies
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

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
                            {isInitializing && (
                                <div className="flex flex-col items-center justify-center text-muted" role="status" style={{ position: 'absolute', inset: 0, zIndex: 10, backgroundColor: '#000' }}>
                                    <Loader2 size={48} className="animate-spin mb-md" style={{ color: 'var(--color-primary)' }} />
                                    <p>Starting camera...</p>
                                </div>
                            )}

                            {/* Countdown Overlay */}
                            {countdown !== null && countdown > 0 && (
                                <div className="countdown-overlay" aria-live="assertive">
                                    <span className="countdown-number" key={countdown}>{countdown}</span>
                                </div>
                            )}

                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                muted
                                className="camera-video"
                                aria-label="Camera preview"
                                aria-describedby="attendance-instructions"
                            />
                            <canvas ref={canvasRef} className="hidden-canvas" />

                            {/* Flash effect overlay */}
                            <div
                                className={`camera-flash ${showFlash ? 'active' : ''}`}
                                aria-hidden="true"
                            />

                            {/* Face guide overlay */}
                            <div className="face-guide" aria-hidden="true">
                                <div className="face-oval"></div>
                            </div>
                        </>
                    )}
                </div>

                {/* Result Display */}
                {result && (
                    <div
                        className={`result-card card ${result.recognized ? 'result-success' :
                            result.spoofDetected ? 'result-warning' : 'result-error'
                        }`}
                        role="alert"
                        aria-live="assertive"
                    >
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
                        <div className="flex flex-col items-center gap-sm">
                            <button
                                onClick={startCaptureSequence}
                                disabled={!stream || isProcessing || countdown !== null}
                                className="btn btn-primary btn-lg capture-button"
                                aria-label={
                                    countdown !== null && countdown > 0
                                        ? `Capturing in ${countdown}...`
                                        : countdown === 0
                                            ? "Capturing now..."
                                            : "Start capture sequence"
                                }
                            >
                                {isProcessing ? (
                                    <>
                                        <Loader2 size={20} className="animate-spin" />
                                        Processing...
                                    </>
                                ) : countdown !== null ? (
                                    <>
                                        <Camera size={20} />
                                        Capturing in {countdown}...
                                    </>
                                ) : (
                                    <>
                                        <Camera size={20} />
                                        Capture & Recognize
                                    </>
                                )}
                            </button>
                            <p className="text-muted text-xs flex items-center gap-xs">
                                <Keyboard size={14} />
                                Press <strong>Space</strong> to capture
                            </p>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center gap-sm">
                            <div className="flex gap-md">
                                {result.recognized ? (
                                    <>
                                        <button
                                            onClick={() => navigate('/')}
                                            className="btn btn-secondary btn-lg"
                                            aria-label="Return to Home Page"
                                        >
                                            <Home size={20} />
                                            Return Home
                                        </button>
                                        <button
                                            onClick={resetAttempt}
                                            className="btn btn-primary btn-lg"
                                            aria-label="Mark attendance for another person"
                                        >
                                            <UserCheck size={20} />
                                            Mark Another
                                        </button>
                                    </>
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
                            <div role="status" aria-live="polite">
                                <p className="text-muted text-xs flex items-center gap-xs">
                                    <Keyboard size={14} />
                                    Press <strong>Escape</strong> to {result.recognized ? 'mark another' : 'try again'}
                                </p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Instructions */}
                <div className="instructions card" id="attendance-instructions">
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
