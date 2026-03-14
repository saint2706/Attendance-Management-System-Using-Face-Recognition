import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import {
    ArrowRight,
    Clock,
    Shield,
    ChartBar,
    Zap,
    UserCheck,
    LogIn
} from 'lucide-react';
import './Home.css';

/**
 * The landing page component for the application.
 * Displays project overview, key features, and quick links.
 * @returns {JSX.Element} The home page UI.
 */
export const Home = () => {
    const { isAuthenticated } = useAuth();

    return (
        <div className="home animate-fade-in">
            {/* Hero Section */}
            <section className="hero">
                <div className="hero-content">
                    <img
                        src="/static/icons/icon-512.png"
                        alt=""
                        width="128"
                        height="128"
                        fetchPriority="high"
                        className="hero-image"
                        aria-hidden="true"
                    />
                    <h1 className="hero-title">
                        Smart Attendance System
                    </h1>
                    <p className="hero-subtitle">
                        Automated attendance tracking using advanced face recognition technology.
                        Fast, secure, and accurate.
                    </p>
                    <div className="hero-actions">
                        <Link to="/mark-attendance" className="btn btn-primary btn-lg">
                            <Clock size={20} aria-hidden="true" />
                            Mark Time-In
                        </Link>
                        <Link to="/mark-attendance?direction=out" className="btn btn-secondary btn-lg">
                            <ArrowRight size={20} aria-hidden="true" />
                            Mark Time-Out
                        </Link>
                        {!isAuthenticated && (
                            <Link to="/login" className="btn btn-secondary btn-lg">
                                <LogIn size={20} aria-hidden="true" />
                                Dashboard Login
                            </Link>
                        )}
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="features">
                <h2 className="features-title text-center">Why Choose Our System?</h2>
                <div className="features-grid">
                    <article className="feature-card card card-elevated">
                        <div className="card-body text-center">
                            <Zap size={32} className="feature-icon" aria-hidden="true" />
                            <h3>Fast & Accurate</h3>
                            <p className="text-muted">
                                Advanced AI-powered face recognition ensures quick and precise attendance marking.
                            </p>
                        </div>
                    </article>
                    <article className="feature-card card card-elevated">
                        <div className="card-body text-center">
                            <Shield size={32} className="feature-icon" aria-hidden="true" />
                            <h3>Secure & Private</h3>
                            <p className="text-muted">
                                Your biometric data is encrypted and stored securely with industry-standard protection.
                            </p>
                        </div>
                    </article>
                    <article className="feature-card card card-elevated">
                        <div className="card-body text-center">
                            <ChartBar size={32} className="feature-icon" aria-hidden="true" />
                            <h3>Detailed Reports</h3>
                            <p className="text-muted">
                                Generate comprehensive attendance reports with visual analytics and export capabilities.
                            </p>
                        </div>
                    </article>
                </div>
            </section>

            {/* Privacy Section */}
            <section className="privacy-notice card">
                <div className="card-body text-center">
                    <UserCheck size={32} className="feature-icon" aria-hidden="true" />
                    <h2>Your Privacy Matters</h2>
                    <p className="text-muted">
                        By using this system, you consent to the collection and processing of your facial data
                        for attendance purposes only. Your images are encrypted and stored securely.
                        We do not share your biometric data with third parties.
                    </p>
                </div>
            </section>
        </div>
    );
};
