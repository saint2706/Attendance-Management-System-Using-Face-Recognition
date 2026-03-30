import { useState, useRef, useEffect } from 'react';
import type { FormEvent } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { LogIn, AlertCircle, Loader2, Eye, EyeOff } from 'lucide-react';
import './Login.css';

/**
 * The login page component.
 * Allows users to authenticate and access protected areas of the application.
 * @returns {JSX.Element} The login form UI.
 */
export const Login = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const errorRef = useRef<HTMLDivElement>(null);

    const { login } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();

    const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/dashboard';

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            await login({ username, password });
            navigate(from, { replace: true });
        } catch {
            setError('Invalid username or password. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        if (error && errorRef.current) {
            errorRef.current.focus();
        }
    }, [error]);

    return (
        <div className="login-page animate-fade-in">
            <title>Login - Smart Attendance System</title>
            <meta name="description" content="Login page for the Smart Attendance System dashboard." />
            <script
                type="application/ld+json"
                dangerouslySetInnerHTML={{
                    __html: JSON.stringify({
                        "@context": "https://schema.org",
                        "@type": "WebPage",
                        "name": "Login - Smart Attendance System",
                        "description": "Login page for the Smart Attendance System dashboard.",
                        "url": "https://attendance-system.example.com/login"
                    })
                }}
            />
            <div className="login-card card card-elevated">
                <div className="login-header">
                    <LogIn size={32} className="login-icon" aria-hidden="true" />
                    <h1 className="login-title">Welcome Back</h1>
                    <p className="login-subtitle text-muted">
                        Sign in to access your dashboard
                    </p>
                </div>

                {error && (
                    <div className="login-error" role="alert" id="login-error" aria-live="assertive" ref={errorRef} tabIndex={-1}>
                        <AlertCircle size={18} aria-hidden="true" />
                        <span>{error}</span>
                    </div>
                )}

                <form onSubmit={handleSubmit} className="login-form">
                    <div className="form-group">
                        <label htmlFor="username" className="input-label">
                            Username <span className="text-danger" aria-hidden="true">*</span>
                        </label>
                        <input
                            type="text"
                            id="username"
                            name="username"
                            className="input"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            placeholder="Enter your username"
                            required
                            aria-required="true"
                            autoFocus
                            autoComplete="username"
                            disabled={isLoading}
                            aria-invalid={Boolean(error)}
                            aria-describedby={error ? "login-error" : undefined}
                        />
                    </div>

                    <div className="form-group">
                        <label htmlFor="password" className="input-label">
                            Password <span className="text-danger" aria-hidden="true">*</span>
                        </label>
                        <div className="input-with-icon">
                            <input
                                type={showPassword ? 'text' : 'password'}
                                id="password"
                                name="password"
                                className="input"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="Enter your password"
                                required
                                aria-required="true"
                                autoComplete="current-password"
                                disabled={isLoading}
                                aria-invalid={Boolean(error)}
                                aria-describedby={error ? "login-error" : undefined}
                            />
                            <button
                                type="button"
                                className="password-toggle"
                                onClick={() => setShowPassword(!showPassword)}
                                aria-label={showPassword ? 'Hide password' : 'Show password'}
                                title={showPassword ? 'Hide password' : 'Show password'}
                                disabled={isLoading}
                            >
                                {showPassword ? (
                                    <EyeOff size={20} aria-hidden="true" />
                                ) : (
                                    <Eye size={20} aria-hidden="true" />
                                )}
                            </button>
                        </div>
                    </div>

                    <button
                        type="submit"
                        className="btn btn-primary btn-lg login-button"
                        disabled={isLoading}
                        aria-live="polite"
                    >
                        {isLoading ? (
                            <>
                                <Loader2 size={18} className="animate-spin" aria-hidden="true" />
                                Signing in...
                            </>
                        ) : (
                            <>
                                <LogIn size={18} aria-hidden="true" />
                                Sign In
                            </>
                        )}
                    </button>
                </form>

                <div className="login-footer">
                    <Link to="/" className="text-muted">← Back to Home</Link>
                </div>
            </div>
        </div>
    );
};
