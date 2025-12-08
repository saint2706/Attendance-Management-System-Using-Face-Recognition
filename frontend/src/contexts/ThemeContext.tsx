import { createContext, useContext, useState, useEffect, useMemo } from 'react';
import type { ReactNode } from 'react';

type Theme = 'light' | 'dark' | 'system';

interface ThemeContextType {
    theme: Theme;
    resolvedTheme: 'light' | 'dark';
    setTheme: (theme: Theme) => void;
    toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const THEME_KEY = 'attendance_theme';

const getSystemTheme = (): 'light' | 'dark' => {
    if (typeof window !== 'undefined' && window.matchMedia) {
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return 'light';
};

export const ThemeProvider = ({ children }: { children: ReactNode }) => {
    const [theme, setThemeState] = useState<Theme>(() => {
        if (typeof window !== 'undefined') {
            return (localStorage.getItem(THEME_KEY) as Theme) || 'system';
        }
        return 'system';
    });

    // Track system theme changes to trigger re-calculation
    const [systemTheme, setSystemTheme] = useState<'light' | 'dark'>(getSystemTheme);

    // Calculate resolved theme from theme state (memoized to avoid recalculation on every render)
    const resolvedTheme = useMemo(() => {
        return theme === 'system' ? systemTheme : theme;
    }, [theme, systemTheme]);

    // Update DOM and localStorage when resolved theme changes
    useEffect(() => {
        document.documentElement.setAttribute('data-theme', resolvedTheme);
        localStorage.setItem(THEME_KEY, theme);
    }, [resolvedTheme, theme]);

    // Listen for system theme changes
    useEffect(() => {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
        const handleChange = () => {
            setSystemTheme(getSystemTheme());
        };

        mediaQuery.addEventListener('change', handleChange);
        return () => mediaQuery.removeEventListener('change', handleChange);
    }, []);

    const setTheme = (newTheme: Theme) => {
        setThemeState(newTheme);
    };

    const toggleTheme = () => {
        setTheme(resolvedTheme === 'light' ? 'dark' : 'light');
    };

    return (
        <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    );
};

// Export the hook separately to fix react-refresh/only-export-components
export function useTheme() {
    const context = useContext(ThemeContext);
    if (context === undefined) {
        throw new Error('useTheme must be used within a ThemeProvider');
    }
    return context;
}
