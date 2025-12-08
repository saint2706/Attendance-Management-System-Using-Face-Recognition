import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { Navbar } from './components/layout/Navbar';
import { Home } from './pages/Home';
import { Login } from './pages/Login';
import { Dashboard } from './pages/Dashboard';
import { MarkAttendance } from './pages/MarkAttendance';
import './index.css';

// Protected route wrapper
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center" style={{ minHeight: '60vh' }}>
        <div className="animate-pulse text-muted">Loading...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

// Main app content
const AppContent = () => {
  return (
    <div className="app">
      <Navbar />
      <main className="container" style={{ paddingTop: 'var(--spacing-lg)', paddingBottom: 'var(--spacing-xl)' }}>
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
          {/* Placeholder routes - to be implemented */}
          <Route path="/attendance" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/session" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/add-photos" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/train" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/employees/register" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/setup-wizard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />

          {/* Fallback - redirect to home */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="text-center text-muted text-sm" style={{ padding: 'var(--spacing-lg)', borderTop: '1px solid var(--color-border)' }}>
        © 2024 Smart Attendance System. All rights reserved.
      </footer>
    </div>
  );
};

function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <AuthProvider>
          <AppContent />
        </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}

export default App;
