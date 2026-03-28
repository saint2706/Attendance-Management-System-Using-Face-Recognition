import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import {
    ArrowRight,
    Clock,
    Shield,
    ChartBar,
    Zap,
    UserCheck,
    LogIn,
    ScanFace
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
            <script
                type="application/ld+json"
                dangerouslySetInnerHTML={{
                    __html: JSON.stringify({
                        "@context": "https://schema.org",
                        "@graph": [
                            {
                                "@type": "SoftwareApplication",
                                "name": "Smart Attendance System",
                                "applicationCategory": "BusinessApplication",
                                "description": "Automated attendance tracking using face recognition technology"
                            },
                            {
                                "@type": "WebSite",
                                "name": "Smart Attendance System",
                                "description": "Automated attendance tracking using face recognition technology",
                                "url": "https://attendance-system.example.com/",
                                "potentialAction": {
                                    "@type": "SearchAction",
                                    "target": "https://attendance-system.example.com/search?q={search_term_string}",
                                    "query-input": "required name=search_term_string"
                                }
                            },
                            {
                                "@type": "FAQPage",
                                "mainEntity": [
                                    {
                                        "@type": "Question",
                                        "name": "How does the Smart Attendance System work?",
                                        "acceptedAnswer": {
                                            "@type": "Answer",
                                            "text": "The system uses advanced AI-powered face recognition technology to quickly and accurately mark attendance when you scan your face."
                                        }
                                    },
                                    {
                                        "@type": "Question",
                                        "name": "Is my biometric data secure?",
                                        "acceptedAnswer": {
                                            "@type": "Answer",
                                            "text": "Yes, your biometric data is encrypted and stored securely with industry-standard protection. We do not share your biometric data with third parties."
                                        }
                                    }
                                ]
                            }
                        ]
                    })
                }}
            />
            {/* Hero Section */}
            <section className="hero" aria-labelledby="hero-title">
                <div className="hero-content">
                    <ScanFace
                        size={128}
                        className="hero-icon mb-md mx-auto"
                        aria-hidden="true"
                        style={{ color: 'var(--color-primary)' }}
                    />
                    <h1 className="hero-title" id="hero-title">
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
            <section className="features" aria-labelledby="features-title">
                <h2 className="features-title text-center" id="features-title">Why Choose Our System?</h2>
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

            {/* FAQ Section */}
            <section className="faq-section card mb-lg" aria-labelledby="faq-title">
                <div className="card-body">
                    <h2 id="faq-title" className="text-center mb-md">Frequently Asked Questions</h2>
                    <div className="faq-item mb-md">
                        <h3 className="text-lg font-semibold mb-sm">How does the Smart Attendance System work?</h3>
                        <p className="text-muted">
                            The system uses advanced AI-powered face recognition technology to quickly and accurately mark attendance when you scan your face.
                        </p>
                    </div>
                    <div className="faq-item">
                        <h3 className="text-lg font-semibold mb-sm">Is my biometric data secure?</h3>
                        <p className="text-muted">
                            Yes, your biometric data is encrypted and stored securely with industry-standard protection. We do not share your biometric data with third parties.
                        </p>
                    </div>
                </div>
            </section>

            {/* Privacy Section */}
            <section className="privacy-notice card" aria-labelledby="privacy-title">
                <div className="card-body text-center">
                    <UserCheck size={32} className="feature-icon" aria-hidden="true" />
                    <h2 id="privacy-title">Your Privacy Matters</h2>
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
