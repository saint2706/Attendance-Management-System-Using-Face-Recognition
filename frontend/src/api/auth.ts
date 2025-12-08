import apiClient from './client';
import { setToken, setRefreshToken, removeToken, removeRefreshToken } from './client';

export interface LoginCredentials {
    username: string;
    password: string;
}

export interface LoginResponse {
    access: string;
    refresh: string;
    user: User;
}

export interface User {
    id: number;
    username: string;
    email: string;
    firstName: string;
    lastName: string;
    isStaff: boolean;
    isActive: boolean;
}

export interface RegisterData {
    username: string;
    email: string;
    password: string;
    firstName?: string;
    lastName?: string;
}

// Login
export const login = async (credentials: LoginCredentials): Promise<LoginResponse> => {
    const response = await apiClient.post<LoginResponse>('/auth/login/', credentials);

    // Store tokens
    setToken(response.data.access);
    setRefreshToken(response.data.refresh);

    return response.data;
};

// Logout
export const logout = async (): Promise<void> => {
    try {
        await apiClient.post('/auth/logout/');
    } finally {
        removeToken();
        removeRefreshToken();
    }
};

// Get current user
export const getCurrentUser = async (): Promise<User> => {
    const response = await apiClient.get<User>('/auth/me/');
    return response.data;
};

// Register new employee (admin only)
export const registerEmployee = async (data: RegisterData): Promise<User> => {
    const response = await apiClient.post<User>('/auth/register/', data);
    return response.data;
};

// Verify token is still valid
export const verifyToken = async (): Promise<boolean> => {
    try {
        await apiClient.post('/auth/verify/');
        return true;
    } catch {
        return false;
    }
};
