import { Link } from 'react-router-dom';
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

export const Dashboard = () => {
    const { user } = useAuth();

    // Mock stats - in real app, these would come from API
    const stats = {
        totalEmployees: 25,
        presentToday: 18,
        status: 'Active'
    };

    return (
        <div className="dashboard animate-fade-in">
            <header className="dashboard-header">
                <div>
                    <h1 className="dashboard-title">Admin Dashboard</h1>
                    <p className="text-muted">Welcome back, {user?.username}!</p>
                </div>
                <div className="header-actions">
                    <Link to="/setup-wizard" className="btn btn-primary">
                        Setup Wizard
                    </Link>
                </div>
            </header>

            {/* Quick Stats */}
            <section className="stats-section">
                <h2 className="section-title">Quick Overview</h2>
                <div className="stats-grid">
                    <div className="stat-card card card-elevated">
                        <div className="stat-content">
                            <div>
                                <p className="stat-label">Total Employees</p>
                                <p className="stat-value">{stats.totalEmployees}</p>
                            </div>
                            <Users size={32} className="stat-icon" />
                        </div>
                    </div>
                    <div className="stat-card card card-elevated stat-success">
                        <div className="stat-content">
                            <div>
                                <p className="stat-label">Present Today</p>
                                <p className="stat-value">{stats.presentToday}</p>
                            </div>
                            <UserCheck size={32} className="stat-icon" />
                        </div>
                    </div>
                    <div className="stat-card card card-elevated stat-info">
                        <div className="stat-content">
                            <div>
                                <p className="stat-label">System Status</p>
                                <p className="stat-value stat-status">
                                    <Activity size={20} />
                                    {stats.status}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Admin Actions */}
            <section className="actions-section">
                <h2 className="section-title">Admin Actions</h2>
                <div className="actions-grid">
                    <Link to="/employees/register" className="action-card card card-elevated">
                        <div className="card-body">
                            <UserPlus size={32} className="action-icon" />
                            <h3>Register Employee</h3>
                            <p className="text-muted text-sm">
                                Add a new employee to the system
                            </p>
                        </div>
                    </Link>

                    <Link to="/add-photos" className="action-card card card-elevated">
                        <div className="card-body">
                            <Camera size={32} className="action-icon" />
                            <h3>Add Photos</h3>
                            <p className="text-muted text-sm">
                                Capture photos for face recognition
                            </p>
                        </div>
                    </Link>

                    <Link to="/train" className="action-card card card-elevated">
                        <div className="card-body">
                            <Brain size={32} className="action-icon" />
                            <h3>Train Model</h3>
                            <p className="text-muted text-sm">
                                Update the recognition model
                            </p>
                        </div>
                    </Link>

                    <Link to="/attendance" className="action-card card card-elevated">
                        <div className="card-body">
                            <ChartBar size={32} className="action-icon" />
                            <h3>View Attendance</h3>
                            <p className="text-muted text-sm">
                                Access attendance reports
                            </p>
                        </div>
                    </Link>

                    <Link to="/session" className="action-card card card-elevated">
                        <div className="card-body">
                            <Radio size={32} className="action-icon" />
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
