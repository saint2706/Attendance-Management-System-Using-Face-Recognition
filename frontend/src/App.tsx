import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { Loader2 } from 'lucide-react';
import React, { Suspense } from 'react';
import { ThemeProvider } from './contexts/ThemeContext';
import { Navbar } from './components/layout/Navbar';
import './index.css';

const Home = React.lazy(() => import('./pages/Home').then(module => ({ default: module.Home })));
const Login = React.lazy(() => import('./pages/Login').then(module => ({ default: module.Login })));
const Dashboard = React.lazy(() => import('./pages/Dashboard').then(module => ({ default: module.Dashboard })));
const MarkAttendance = React.lazy(() => import('./pages/MarkAttendance').then(module => ({ default: module.MarkAttendance })));

/**
 * A wrapper component that protects routes requiring authentication.
 * If the user is not authenticated, it redirects them to the login page.
 * @param {Object} props - The component props.
 * @param {React.ReactNode} props.children - The child components to render if authenticated.
 * @returns {JSX.Element} The child components or a redirect to the login page.
 */
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center" style={{ minHeight: '60vh' }} role="status" aria-live="polite" aria-busy="true">
        <Loader2 size={48} className="animate-spin mb-md" style={{ color: 'var(--color-primary)' }} aria-hidden="true" />
        <div className="animate-pulse text-muted">Loading...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

/**
 * The main application content component containing navigation and routing.
 * @returns {JSX.Element} The main layout and route structure of the application.
 */
const AppContent = () => {
  return (
    <div className="app">
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
      <Navbar />
      <main id="main-content" className="container" style={{ paddingTop: 'var(--spacing-lg)', paddingBottom: 'var(--spacing-xl)' }} tabIndex={-1}>
        <Suspense fallback={
          <div className="flex flex-col items-center justify-center" style={{ minHeight: '60vh' }} role="status" aria-live="polite" aria-busy="true">
            <Loader2 size={48} className="animate-spin mb-md" style={{ color: 'var(--color-primary)' }} aria-hidden="true" />
            <div className="animate-pulse text-muted">Loading...</div>
          </div>
        }>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<Login />} />
            <Route path="/mark-attendance" element={<MarkAttendance />} />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              }
            />
            <Route path="/attendance" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
            <Route path="/session" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
            <Route path="/add-photos" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
            <Route path="/train" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
            <Route path="/employees/register" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
            <Route path="/setup-wizard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />

            {/* Fallback - redirect to home */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </main>
      <footer className="text-center text-muted text-sm" style={{ padding: 'var(--spacing-lg)', borderTop: '1px solid var(--color-border)' }}>
        © 2024 Smart Attendance System. All rights reserved.
      </footer>
    </div>
  );
};

/**
 * The root component of the React application.
 * Sets up routing, theming, and authentication providers.
 * @returns {JSX.Element} The configured React application.
 */
function App() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <ThemeProvider>
        <AuthProvider>
          <AppContent />
        </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}

/**
 * The main application component that sets up routing and context providers.
 * @returns {JSX.Element} The rendered application component.
 */
export default App;
