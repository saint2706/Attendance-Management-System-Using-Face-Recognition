import { Link } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
    UserPlus,
    Camera,
    Brain,
    ChartBar,
    Radio,
    Users,
    UserCheck,
    Activity
} from 'lucide-react';
import './Dashboard.css';

/**
 * The protected admin dashboard page.
 * Displays system statistics and provides links to management functions.
 * @returns {JSX.Element} The admin dashboard UI.
 */
export const Dashboard = () => {
    const { user } = useAuth();

    const [isLoadingStats, setIsLoadingStats] = useState(true);
    // Mock stats - in real app, these would come from API
    const [stats, setStats] = useState({
        totalEmployees: 0,
        presentToday: 0,
        status: 'Loading...'
    });

    useEffect(() => {
        // Simulate API call
        const timer = setTimeout(() => {
            setStats({
                totalEmployees: 25,
                presentToday: 18,
                status: 'Active'
            });
            setIsLoadingStats(false);
        }, 1000);
        return () => clearTimeout(timer);
    }, []);

    const getGreeting = () => {
        const hour = new Date().getHours();
        if (hour < 12) return 'Good morning';
        if (hour < 18) return 'Good afternoon';
        return 'Good evening';
    };

    return (
        <div className="dashboard animate-fade-in">
            <header className="dashboard-header">
                <div>
                    <h1 className="dashboard-title">Admin Dashboard</h1>
                    <p className="text-muted">{getGreeting()}, {user?.username}!</p>
                </div>
                <div className="header-actions">
                    <Link to="/setup-wizard" className="btn btn-primary" title="Start the Setup Wizard">
                        Setup Wizard
                    </Link>
                </div>
            </header>

            {/* Quick Stats */}
            <section className="stats-section" aria-labelledby="stats-title">
                <h2 className="section-title" id="stats-title">Quick Overview</h2>
                <div className="stats-grid" aria-live="polite">
                    <div className="stat-card card card-elevated">
                        <div className="stat-content">
                            <div>
                                <p className="stat-label">Total Employees</p>
                                {isLoadingStats ? (
                                    <div className="animate-pulse h-8 w-16 skeleton rounded mt-1" />
                                ) : (
                                    <p className="stat-value">{stats.totalEmployees}</p>
                                )}
                            </div>
                            <Users size={32} className="stat-icon" aria-hidden="true" />
                        </div>
                    </div>
                    <div className="stat-card card card-elevated stat-success">
                        <div className="stat-content">
                            <div>
                                <p className="stat-label">Present Today</p>
                                {isLoadingStats ? (
                                    <div className="animate-pulse h-8 w-16 skeleton rounded mt-1" />
                                ) : (
                                    <p className="stat-value">{stats.presentToday}</p>
                                )}
                            </div>
                            <UserCheck size={32} className="stat-icon" aria-hidden="true" />
                        </div>
                    </div>
                    <div className="stat-card card card-elevated stat-info">
                        <div className="stat-content">
                            <div>
                                <p className="stat-label">System Status</p>
                                {isLoadingStats ? (
                                    <div className="animate-pulse h-8 w-24 skeleton rounded mt-1" />
                                ) : (
                                    <p className="stat-value stat-status">
                                        <Activity size={20} aria-hidden="true" />
                                        {stats.status}
                                    </p>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Admin Actions */}
            <section className="actions-section" aria-labelledby="actions-title">
                <h2 className="section-title" id="actions-title">Admin Actions</h2>
                <div className="actions-grid">
                    <Link to="/employees/register" className="action-card card card-elevated" title="Register a new employee">
                        <div className="card-body">
                            <UserPlus size={32} className="action-icon" aria-hidden="true" />
                            <h3>Register Employee</h3>
                            <p className="text-muted text-sm">
                                Add a new employee to the system
                            </p>
                        </div>
                    </Link>

                    <Link to="/add-photos" className="action-card card card-elevated" title="Capture photos for face recognition">
                        <div className="card-body">
                            <Camera size={32} className="action-icon" aria-hidden="true" />
                            <h3>Add Photos</h3>
                            <p className="text-muted text-sm">
                                Capture photos for face recognition
                            </p>
                        </div>
                    </Link>

                    <Link to="/train" className="action-card card card-elevated" title="Update the recognition model">
                        <div className="card-body">
                            <Brain size={32} className="action-icon" aria-hidden="true" />
                            <h3>Train Model</h3>
                            <p className="text-muted text-sm">
                                Update the recognition model
                            </p>
                        </div>
                    </Link>

                    <Link to="/attendance" className="action-card card card-elevated" title="Access attendance reports">
                        <div className="card-body">
                            <ChartBar size={32} className="action-icon" aria-hidden="true" />
                            <h3>View Attendance</h3>
                            <p className="text-muted text-sm">
                                Access attendance reports
                            </p>
                        </div>
                    </Link>

                    <Link to="/session" className="action-card card card-elevated" title="Monitor live recognition">
                        <div className="card-body">
                            <Radio size={32} className="action-icon" aria-hidden="true" />
                            <h3>Attendance Session</h3>
                            <p className="text-muted text-sm">
                                Monitor live recognition
                            </p>
                        </div>
                    </Link>
                </div>
            </section>
        </div>
    );
};
