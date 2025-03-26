import { Routes, Route } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Layout from '@components/layout/Layout';
import Dashboard from '@pages/Dashboard';
import DataSources from '@pages/DataSources';
import Analytics from '@pages/Analytics';
import Tokenization from '@pages/Tokenization';
import Marketplace from '@pages/Marketplace';
import Settings from '@pages/Settings';
import NotFound from '@pages/NotFound';
import { AuthProvider } from '@context/AuthContext';
import { DataProvider } from '@context/DataContext';

function App() {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simuliere Ladezeit für die Anwendung
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <div className="inline-block w-16 h-16 border-4 border-ocean-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="mt-4 text-lg text-gray-700">OceanData wird geladen...</p>
        </div>
      </div>
    );
  }

  return (
    <AuthProvider>
      <DataProvider>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="data-sources" element={<DataSources />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="tokenization" element={<Tokenization />} />
            <Route path="marketplace" element={<Marketplace />} />
            <Route path="settings" element={<Settings />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </DataProvider>
    </AuthProvider>
  );
}

export default App;