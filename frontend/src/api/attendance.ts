import apiClient from './client';

/**
 * Represents a single attendance record.
 */
export interface AttendanceRecord {
    id: number;
    userId: number;
    username: string;
    direction: 'in' | 'out';
    timestamp: string;
    confidence: number;
    successful: boolean;
    spoofDetected: boolean;
}

/**
 * Represents the daily attendance statistics.
 */
export interface AttendanceStats {
    totalEmployees: number;
    presentToday: number;
    checkedOutToday: number;
    pendingCheckout: number;
}

/**
 * Represents the result of a facial recognition attempt.
 */
export interface RecognitionResult {
    recognized: boolean;
    username?: string;
    distance?: number;
    confidence?: number;
    spoofDetected: boolean;
    message: string;
}

/**
 * Represents an event in the live attendance feed.
 */
export interface AttendanceFeedEvent {
    eventType: 'attempt' | 'outcome';
    username: string;
    direction: string;
    timestamp: string;
    accepted?: boolean;
    confidence?: number;
    liveness?: 'passed' | 'failed' | 'skipped';
}

/**
 * Retrieves attendance records based on optional filtering parameters.
 * @param {Object} [params] - Optional filtering parameters.
 * @param {string} [params.date] - The date to filter by (YYYY-MM-DD).
 * @param {number} [params.userId] - The ID of the user to filter by.
 * @param {'in' | 'out'} [params.direction] - The direction of attendance ('in' or 'out').
 * @returns {Promise<AttendanceRecord[]>} A promise resolving to an array of attendance records.
 */
export const getAttendanceRecords = async (params?: {
    date?: string;
    userId?: number;
    direction?: 'in' | 'out';
}): Promise<AttendanceRecord[]> => {
    const response = await apiClient.get<AttendanceRecord[]>('/attendance/', { params });
    return response.data;
};

/**
 * Retrieves today's attendance statistics.
 * @returns {Promise<AttendanceStats>} A promise resolving to the attendance statistics.
 */
export const getAttendanceStats = async (): Promise<AttendanceStats> => {
    const response = await apiClient.get<AttendanceStats>('/dashboard/stats/');
    return response.data;
};

/**
 * Marks attendance by sending a captured face image for recognition.
 * @param {string} imageBase64 - The base64-encoded image data.
 * @param {'in' | 'out'} direction - The direction of attendance.
 * @returns {Promise<RecognitionResult>} A promise resolving to the recognition result.
 */
export const markAttendance = async (
    imageBase64: string,
    direction: 'in' | 'out'
): Promise<RecognitionResult> => {
    const response = await apiClient.post<RecognitionResult>('/recognition/', {
        image: imageBase64,
        direction,
    });
    return response.data;
};

/**
 * Retrieves the live attendance feed events.
 * @returns {Promise<{ events: AttendanceFeedEvent[] }>} A promise resolving to the feed events.
 */
export const getAttendanceFeed = async (): Promise<{ events: AttendanceFeedEvent[] }> => {
    // This uses the existing Django endpoint
    const response = await apiClient.get<{ events: AttendanceFeedEvent[] }>(
        '/attendance_session/feed/'
    );
    return response.data;
};

/**
 * Retrieves attendance records for a specific date.
 * @param {string} date - The date to fetch records for (YYYY-MM-DD).
 * @returns {Promise<AttendanceRecord[]>} A promise resolving to an array of attendance records.
 */
export const getAttendanceByDate = async (date: string): Promise<AttendanceRecord[]> => {
    const response = await apiClient.get<AttendanceRecord[]>('/attendance/', {
        params: { date },
    });
    return response.data;
};

/**
 * Retrieves the attendance history for a specific employee.
 * @param {number} userId - The ID of the user.
 * @param {string} [startDate] - The start date for filtering (YYYY-MM-DD).
 * @param {string} [endDate] - The end date for filtering (YYYY-MM-DD).
 * @returns {Promise<AttendanceRecord[]>} A promise resolving to an array of attendance records.
 */
export const getEmployeeAttendance = async (
    userId: number,
    startDate?: string,
    endDate?: string
): Promise<AttendanceRecord[]> => {
    const response = await apiClient.get<AttendanceRecord[]>(`/employees/${userId}/attendance/`, {
        params: { startDate, endDate },
    });
    return response.data;
};
