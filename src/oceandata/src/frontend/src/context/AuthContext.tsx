import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  id: string;
  username: string;
  email: string;
  walletAddress?: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => void;
  connectWallet: (address: string) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Überprüfe, ob ein Benutzer im lokalen Speicher vorhanden ist
    const storedUser = localStorage.getItem('oceandata_user');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
    setIsLoading(false);
  }, []);

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      // Für die Demo verwenden wir einen simulierten Login
      // In einer echten Anwendung würde hier eine API-Anfrage erfolgen
      setIsLoading(true);
      
      // Simuliere API-Anfrage
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Demo-Benutzer
      const demoUser: User = {
        id: '1',
        username: 'demo_user',
        email: email
      };
      
      setUser(demoUser);
      localStorage.setItem('oceandata_user', JSON.stringify(demoUser));
      setIsLoading(false);
      return true;
    } catch (error) {
      console.error('Login fehlgeschlagen:', error);
      setIsLoading(false);
      return false;
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('oceandata_user');
  };

  const connectWallet = (address: string) => {
    if (user) {
      const updatedUser = { ...user, walletAddress: address };
      setUser(updatedUser);
      localStorage.setItem('oceandata_user', JSON.stringify(updatedUser));
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        logout,
        connectWallet
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth muss innerhalb eines AuthProviders verwendet werden');
  }
  return context;
}