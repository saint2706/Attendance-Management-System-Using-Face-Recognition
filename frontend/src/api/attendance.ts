import apiClient from './client';

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

export interface AttendanceStats {
    totalEmployees: number;
    presentToday: number;
    checkedOutToday: number;
    pendingCheckout: number;
}

export interface RecognitionResult {
    recognized: boolean;
    username?: string;
    distance?: number;
    confidence?: number;
    spoofDetected: boolean;
    message: string;
}

export interface AttendanceFeedEvent {
    eventType: 'attempt' | 'outcome';
    username: string;
    direction: string;
    timestamp: string;
    accepted?: boolean;
    confidence?: number;
    liveness?: 'passed' | 'failed' | 'skipped';
}

// Get attendance records
export const getAttendanceRecords = async (params?: {
    date?: string;
    userId?: number;
    direction?: 'in' | 'out';
}): Promise<AttendanceRecord[]> => {
    const response = await apiClient.get<AttendanceRecord[]>('/attendance/', { params });
    return response.data;
};

// Get today's attendance statistics
export const getAttendanceStats = async (): Promise<AttendanceStats> => {
    const response = await apiClient.get<AttendanceStats>('/dashboard/stats/');
    return response.data;
};

// Mark attendance with face image
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

// Get live attendance feed
export const getAttendanceFeed = async (): Promise<{ events: AttendanceFeedEvent[] }> => {
    // This uses the existing Django endpoint
    const response = await apiClient.get<{ events: AttendanceFeedEvent[] }>(
        '/attendance_session/feed/'
    );
    return response.data;
};

// Get attendance by date
export const getAttendanceByDate = async (date: string): Promise<AttendanceRecord[]> => {
    const response = await apiClient.get<AttendanceRecord[]>('/attendance/', {
        params: { date },
    });
    return response.data;
};

// Get employee attendance history
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
