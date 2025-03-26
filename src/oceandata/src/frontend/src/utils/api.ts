import axios from 'axios';

// Basis-URL für API-Anfragen
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api' 
  : 'http://localhost:5000/api';

// Axios-Instanz mit Standardkonfiguration
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor für Authentifizierung
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('oceandata_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// API-Endpunkte
export const endpoints = {
  // Authentifizierung
  auth: {
    login: '/auth/login',
    register: '/auth/register',
    logout: '/auth/logout',
    profile: '/auth/profile',
  },
  
  // Datenquellen
  dataSources: {
    list: '/data-sources',
    add: '/data-sources',
    remove: (id: string) => `/data-sources/${id}`,
    sync: (id: string) => `/data-sources/${id}/sync`,
  },
  
  // Analysen
  analytics: {
    analyze: (id: string) => `/analytics/${id}`,
    results: (id: string) => `/analytics/${id}/results`,
  },
  
  // Tokenisierung
  tokenization: {
    tokenize: '/tokenization',
    list: '/tokenization',
    details: (id: string) => `/tokenization/${id}`,
  },
  
  // Marktplatz
  marketplace: {
    list: '/marketplace',
    details: (id: string) => `/marketplace/${id}`,
    buy: (id: string) => `/marketplace/${id}/buy`,
  },
};

// API-Funktionen
export const apiService = {
  // Authentifizierung
  auth: {
    login: (email: string, password: string) => 
      api.post(endpoints.auth.login, { email, password }),
    register: (username: string, email: string, password: string) => 
      api.post(endpoints.auth.register, { username, email, password }),
    logout: () => 
      api.post(endpoints.auth.logout),
    getProfile: () => 
      api.get(endpoints.auth.profile),
  },
  
  // Datenquellen
  dataSources: {
    list: () => 
      api.get(endpoints.dataSources.list),
    add: (source: any) => 
      api.post(endpoints.dataSources.add, source),
    remove: (id: string) => 
      api.delete(endpoints.dataSources.remove(id)),
    sync: (id: string) => 
      api.post(endpoints.dataSources.sync(id)),
  },
  
  // Analysen
  analytics: {
    analyze: (id: string) => 
      api.post(endpoints.analytics.analyze(id)),
    getResults: (id: string) => 
      api.get(endpoints.analytics.results(id)),
  },
  
  // Tokenisierung
  tokenization: {
    tokenize: (data: any) => 
      api.post(endpoints.tokenization.tokenize, data),
    list: () => 
      api.get(endpoints.tokenization.list),
    getDetails: (id: string) => 
      api.get(endpoints.tokenization.details(id)),
  },
  
  // Marktplatz
  marketplace: {
    list: (filters?: any) => 
      api.get(endpoints.marketplace.list, { params: filters }),
    getDetails: (id: string) => 
      api.get(endpoints.marketplace.details(id)),
    buy: (id: string, paymentDetails: any) => 
      api.post(endpoints.marketplace.buy(id), paymentDetails),
  },
};

export default apiService;