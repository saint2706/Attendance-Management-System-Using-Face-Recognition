import { Link } from 'react-router-dom';
import { ActionCard } from '../components/ActionCard';

import { useAuth } from '../contexts/AuthContext';
import { getAttendanceStats } from '../api/attendance';
import { useQuery } from '@tanstack/react-query';
import {
    UserPlus,
    Camera,
    Brain,
    ChartBar,
    Radio,
    Users,
    Inbox,
    UserCheck,
    Activity,
    AlertTriangle,
    RefreshCw,
    Loader2
} from 'lucide-react';
import './Dashboard.css';

/**
 * The protected admin dashboard page.
 * Displays system statistics and provides links to management functions.
 * @returns {JSX.Element} The admin dashboard UI.
 */
export const Dashboard = () => {
    const { user } = useAuth();

    // ⚡ Bolt: Implemented React Query to cache expensive API calls for dashboard stats.
    // This reduces redundant network requests and prevents unnecessary re-renders
    // when navigating back and forth to the dashboard.
    const {
        data,
        isFetching: isLoadingStats, // use isFetching to show loading state during manual refetch
        isError: hasError,
        refetch: fetchStats,
    } = useQuery({
        queryKey: ['attendanceStats'],
        queryFn: getAttendanceStats,
        staleTime: 5 * 60 * 1000, // 5 minutes cache to prevent rapid refetching
    });

    const stats = {
        totalEmployees: data?.totalEmployees ?? 0,
        presentToday: data?.presentToday ?? 0,
        status: hasError ? 'Error loading stats' : (isLoadingStats ? 'Loading...' : 'Active')
    };

    const getGreeting = () => {
        const hour = new Date().getHours();
        if (hour < 12) return 'Good morning';
        if (hour < 18) return 'Good afternoon';
        return 'Good evening';
    };

    return (
        <main className="dashboard animate-fade-in">
            <title>Dashboard - Smart Attendance System</title>
            <meta name="description" content="Admin dashboard for managing the Smart Attendance System." />
            <script
                type="application/ld+json"
                dangerouslySetInnerHTML={{
                    __html: JSON.stringify({
                        "@context": "https://schema.org",
                        "@type": "WebPage",
                        "name": "Admin Dashboard - Smart Attendance System",
                        "description": "Admin dashboard for managing the Smart Attendance System.",
                        "url": "https://attendance-system.example.com/dashboard"
                    }).replace(/</g, '\\u003c')
                }}
            />
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
                {hasError ? (
                    <div className="text-center py-12 w-full card card-elevated" style={{ gridColumn: '1 / -1' }}>
                        <AlertTriangle size={48} className="mx-auto text-warning mb-sm" aria-hidden="true" />
                        <h3 className="text-lg font-semibold mb-xs">Failed to load statistics</h3>
                        <p className="text-muted mb-md">We couldn't retrieve the latest dashboard data.</p>
                        <button onClick={() => fetchStats()} className="btn btn-secondary" title="Retry loading statistics" disabled={isLoadingStats}>
                            {isLoadingStats ? (
                                <Loader2 size={18} className="animate-spin" aria-hidden="true" />
                            ) : (
                                <RefreshCw size={18} aria-hidden="true" />
                            )}
                            {isLoadingStats ? 'Retrying...' : 'Try Again'}
                        </button>
                    </div>
                ) : !isLoadingStats && stats.totalEmployees === 0 ? (
                    <div className="text-center py-12 w-full card card-elevated" style={{ gridColumn: '1 / -1' }}>
                        <Inbox size={48} className="mx-auto text-muted mb-sm" aria-hidden="true" />
                        <h3 className="text-lg font-semibold mb-xs">No employees yet</h3>
                        <p className="text-muted mb-md">Get started by registering your first employee.</p>
                        <Link to="/employees/register" className="btn btn-primary" title="Register your first employee">
                            <UserPlus size={18} aria-hidden="true" />
                            Register Employee
                        </Link>
                    </div>
                ) : (
                    <div className="stats-grid" aria-live="polite">
                        <div className="stat-card card card-elevated">
                            <div className="stat-content">
                                <div>
                                    <p className="stat-label">Total Employees</p>
                                    {isLoadingStats ? (
                                        <div className="animate-pulse h-8 w-16 skeleton rounded mt-1" aria-hidden="true" />
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
                                        <div className="animate-pulse h-8 w-16 skeleton rounded mt-1" aria-hidden="true" />
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
                                        <div className="animate-pulse h-8 w-24 skeleton rounded mt-1" aria-hidden="true" />
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
                )}
            </section>

            {/* Admin Actions */}
            <section className="actions-section" aria-labelledby="actions-title">
                <h2 className="section-title" id="actions-title">Admin Actions</h2>
                <div className="actions-grid">
                    <ActionCard to="/employees/register" title="Register a new employee" icon={UserPlus} heading="Register Employee" description="Add a new employee to the system" />
                    <ActionCard to="/add-photos" title="Capture photos for face recognition" icon={Camera} heading="Add Photos" description="Capture photos for face recognition" />
                    <ActionCard to="/train" title="Update the recognition model" icon={Brain} heading="Train Model" description="Update the recognition model" />
                    <ActionCard to="/attendance" title="Access attendance reports" icon={ChartBar} heading="View Attendance" description="Access attendance reports" />
                    <ActionCard to="/session" title="Monitor live recognition" icon={Radio} heading="Attendance Session" description="Monitor live recognition" />
                </div>
            </section>
        </main>
    );
};
