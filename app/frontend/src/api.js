import axios from 'axios';

// The REST API (app/backend). Set VITE_API_URL to serve it from elsewhere.
export const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export const api = axios.create({ baseURL: API_URL });

export const imageUrl = (name) => `${API_URL}/api/images/${encodeURIComponent(name)}`;
