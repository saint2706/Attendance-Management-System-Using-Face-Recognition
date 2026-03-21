import apiClient from './client';
import { setToken, setRefreshToken, removeToken, removeRefreshToken } from './client';
import { z } from 'zod';

/**
 * Credentials required for user login.
 */
export interface LoginCredentials {
    username: string;
    password: string;
}

export const LoginCredentialsSchema = z.object({
    username: z.string().min(1, 'Username is required').max(150, 'Username is too long'),
    password: z.string().min(1, 'Password is required'),
});

/**
 * The response received after a successful login.
 */
export interface LoginResponse {
    access: string;
    refresh: string;
    user: User;
}

/**
 * Represents a user profile in the system.
 */
export interface User {
    id: number;
    username: string;
    email: string;
    firstName: string;
    lastName: string;
    isStaff: boolean;
    isActive: boolean;
}

/**
 * Data required to register a new employee.
 */
export interface RegisterData {
    username: string;
    email: string;
    password: string;
    firstName?: string;
    lastName?: string;
}

/**
 * Authenticates a user and stores their tokens.
 * @param {LoginCredentials} credentials - The user's login credentials.
 * @returns {Promise<LoginResponse>} A promise resolving to the login response containing tokens and user data.
 */
export const login = async (credentials: LoginCredentials): Promise<LoginResponse> => {
    // Validate credentials using Zod schema
    LoginCredentialsSchema.parse(credentials);

    const response = await apiClient.post<LoginResponse>('/auth/login/', credentials);

    // Store tokens
    setToken(response.data.access);
    setRefreshToken(response.data.refresh);

    return response.data;
};

/**
 * Logs out the current user and clears stored tokens.
 * @returns {Promise<void>} A promise that resolves when logout is complete.
 */
export const logout = async (): Promise<void> => {
    try {
        await apiClient.post('/auth/logout/');
    } finally {
        removeToken();
        removeRefreshToken();
    }
};

/**
 * Retrieves the currently authenticated user's profile information.
 * @returns {Promise<User>} A promise resolving to the user profile.
 */
export const getCurrentUser = async (): Promise<User> => {
    const response = await apiClient.get<User>('/auth/me/');
    return response.data;
};

/**
 * Registers a new employee in the system (requires admin privileges).
 * @param {RegisterData} data - The new employee's registration data.
 * @returns {Promise<User>} A promise resolving to the created user's profile.
 */
export const registerEmployee = async (data: RegisterData): Promise<User> => {
    const response = await apiClient.post<User>('/auth/register/', data);
    return response.data;
};

/**
 * Verifies if the current authentication token is still valid.
 * @returns {Promise<boolean>} A promise resolving to true if valid, false otherwise.
 */
export const verifyToken = async (): Promise<boolean> => {
    try {
        await apiClient.post('/auth/verify/');
        return true;
    } catch {
        return false;
    }
};
