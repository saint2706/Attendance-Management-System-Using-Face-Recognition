import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { useTheme } from '../../contexts/ThemeContext';
import {
    UserCheck,
    Moon,
    Sun,
    LogOut,
    LogIn,
    LayoutDashboard,
    Menu,
    X
} from 'lucide-react';
import { useState } from 'react';
import './Navbar.css';

export const Navbar = () => {
    const { isAuthenticated, user, logout } = useAuth();
    const { resolvedTheme, toggleTheme } = useTheme();
    const navigate = useNavigate();
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

    const handleLogout = async () => {
        await logout();
        navigate('/');
    };

    return (
        <nav className="navbar">
            <div className="navbar-container">
                <Link to="/" className="navbar-brand">
                    <UserCheck size={24} />
                    <span>Smart Attendance</span>
                </Link>

                <div className="navbar-actions">
                    <ul className={`navbar-nav ${mobileMenuOpen ? 'open' : ''}`}>
                        {isAuthenticated ? (
                            <>
                                <li>
                                    <Link to="/dashboard" className="nav-link">
                                        <LayoutDashboard size={18} />
                                        <span>Dashboard</span>
                                    </Link>
                                </li>
                                <li>
                                    <button onClick={handleLogout} className="nav-link nav-button">
                                        <LogOut size={18} />
                                        <span>Logout</span>
                                    </button>
                                </li>
                                {user && (
                                    <li className="nav-user">
                                        <span className="badge badge-info">{user.username}</span>
                                    </li>
                                )}
                            </>
                        ) : (
                            <li>
                                <Link to="/login" className="nav-link">
                                        <LogIn size={18} />
                                    <span>Login</span>
                                </Link>
                            </li>
                        )}
                    </ul>

                    <button
                        onClick={toggleTheme}
                        className="btn btn-icon theme-toggle"
                            aria-label={resolvedTheme === 'dark' ? "Switch to light mode" : "Switch to dark mode"}
                            title={resolvedTheme === 'dark' ? "Switch to light mode" : "Switch to dark mode"}
                    >
                        {resolvedTheme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
                    </button>

                    <button
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                        className="btn btn-icon mobile-menu-toggle"
                            aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
                            aria-expanded={mobileMenuOpen}
                    >
                        {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
                    </button>
                </div>
            </div>
        </nav>
    );
};
