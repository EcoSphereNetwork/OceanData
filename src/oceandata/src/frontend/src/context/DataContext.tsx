import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useAuth } from './AuthContext';

// Typdefinitionen
export interface DataSource {
  id: string;
  name: string;
  type: 'browser' | 'smartwatch' | 'iot_thermostat' | 'iot_light' | 'iot_security_camera' | string;
  status: 'connected' | 'disconnected' | 'error';
  lastSync: string | null;
  recordCount: number;
  privacyLevel: 'low' | 'medium' | 'high' | 'compute_only';
  estimatedValue: number;
}

export interface AnalyticsResult {
  sourceId: string;
  statistics: {
    recordCount: number;
    dateRange: {
      start: string | null;
      end: string | null;
    };
    fieldCount: number;
  };
  insights: Record<string, any>;
}

export interface TokenizedDataset {
  id: string;
  name: string;
  description: string;
  sourceIds: string[];
  assetId: string;
  tokenAddress: string;
  price: number;
  createdAt: string;
  marketplaceUrl: string;
  privacyLevel: string;
}

interface DataContextType {
  dataSources: DataSource[];
  analyticsResults: Record<string, AnalyticsResult>;
  tokenizedDatasets: TokenizedDataset[];
  isLoading: boolean;
  addDataSource: (source: Omit<DataSource, 'id'>) => Promise<DataSource>;
  removeDataSource: (id: string) => Promise<boolean>;
  syncDataSource: (id: string) => Promise<boolean>;
  analyzeDataSource: (id: string) => Promise<AnalyticsResult | null>;
  tokenizeDataset: (name: string, description: string, sourceIds: string[], price: number, privacyLevel: string) => Promise<TokenizedDataset | null>;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

export function DataProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [analyticsResults, setAnalyticsResults] = useState<Record<string, AnalyticsResult>>({});
  const [tokenizedDatasets, setTokenizedDatasets] = useState<TokenizedDataset[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Lade Datenquellen, wenn ein Benutzer angemeldet ist
    if (user) {
      loadInitialData();
    } else {
      setDataSources([]);
      setAnalyticsResults({});
      setTokenizedDatasets([]);
      setIsLoading(false);
    }
  }, [user]);

  const loadInitialData = async () => {
    setIsLoading(true);
    
    try {
      // In einer echten Anwendung würden hier API-Anfragen erfolgen
      // Für die Demo verwenden wir simulierte Daten
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Demo-Datenquellen
      const demoSources: DataSource[] = [
        {
          id: 'browser-chrome',
          name: 'Chrome Browser History',
          type: 'browser',
          status: 'connected',
          lastSync: new Date().toISOString(),
          recordCount: 1245,
          privacyLevel: 'medium',
          estimatedValue: 0.8
        },
        {
          id: 'smartwatch-fitbit',
          name: 'Fitbit Health Data',
          type: 'smartwatch',
          status: 'connected',
          lastSync: new Date().toISOString(),
          recordCount: 4320,
          privacyLevel: 'medium',
          estimatedValue: 2.1
        },
        {
          id: 'iot_thermostat-nest',
          name: 'Nest Thermostat',
          type: 'iot_thermostat',
          status: 'connected',
          lastSync: new Date().toISOString(),
          recordCount: 720,
          privacyLevel: 'low',
          estimatedValue: 0.5
        }
      ];
      
      setDataSources(demoSources);
      
      // Demo-Tokenisierte Datensätze
      const demoTokenizedDatasets: TokenizedDataset[] = [
        {
          id: 'dataset-1',
          name: 'Meine Gesundheitsdaten',
          description: 'Fitbit-Gesundheitsdaten mit Herzfrequenz und Aktivitätsinformationen',
          sourceIds: ['smartwatch-fitbit'],
          assetId: 'did:op:' + Math.random().toString(36).substring(2, 15),
          tokenAddress: '0x' + Math.random().toString(36).substring(2, 15),
          price: 2.5,
          createdAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
          marketplaceUrl: 'https://market.oceanprotocol.com/asset/did:op:123456',
          privacyLevel: 'medium'
        }
      ];
      
      setTokenizedDatasets(demoTokenizedDatasets);
      
      setIsLoading(false);
    } catch (error) {
      console.error('Fehler beim Laden der Daten:', error);
      setIsLoading(false);
    }
  };

  const addDataSource = async (source: Omit<DataSource, 'id'>): Promise<DataSource> => {
    // Simuliere API-Anfrage
    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const newSource: DataSource = {
      ...source,
      id: `${source.type}-${Math.random().toString(36).substring(2, 9)}`,
      lastSync: null,
      recordCount: 0,
      estimatedValue: 0
    };
    
    setDataSources(prev => [...prev, newSource]);
    setIsLoading(false);
    
    return newSource;
  };

  const removeDataSource = async (id: string): Promise<boolean> => {
    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 500));
    
    setDataSources(prev => prev.filter(source => source.id !== id));
    setIsLoading(false);
    
    return true;
  };

  const syncDataSource = async (id: string): Promise<boolean> => {
    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    setDataSources(prev => 
      prev.map(source => 
        source.id === id 
          ? { 
              ...source, 
              lastSync: new Date().toISOString(),
              recordCount: source.recordCount + Math.floor(Math.random() * 100),
              estimatedValue: source.estimatedValue * (1 + Math.random() * 0.2)
            } 
          : source
      )
    );
    
    setIsLoading(false);
    return true;
  };

  const analyzeDataSource = async (id: string): Promise<AnalyticsResult | null> => {
    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const source = dataSources.find(s => s.id === id);
    if (!source) {
      setIsLoading(false);
      return null;
    }
    
    // Erstelle simulierte Analyseergebnisse basierend auf dem Quellentyp
    let insights: Record<string, any> = {};
    
    if (source.type === 'browser') {
      insights = {
        topDomains: {
          'google.com': 42,
          'youtube.com': 38,
          'github.com': 25,
          'stackoverflow.com': 18,
          'amazon.com': 15
        },
        usageByHour: Array.from({ length: 24 }, (_, i) => ({
          hour: i,
          visits: Math.floor(Math.random() * (i > 8 && i < 23 ? 30 : 10))
        })),
        categories: {
          'social': 30,
          'entertainment': 25,
          'shopping': 15,
          'news': 10,
          'development': 20
        }
      };
    } else if (source.type === 'smartwatch') {
      insights = {
        averageHeartRate: 72 + Math.floor(Math.random() * 10),
        stepsByDay: Array.from({ length: 7 }, (_, i) => ({
          day: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][i],
          steps: 5000 + Math.floor(Math.random() * 7000)
        })),
        sleepQuality: {
          'deep': 20,
          'light': 50,
          'rem': 15,
          'awake': 15
        },
        caloriesBurned: 1800 + Math.floor(Math.random() * 800)
      };
    } else if (source.type.startsWith('iot_')) {
      insights = {
        usagePatterns: Array.from({ length: 24 }, (_, i) => ({
          hour: i,
          activity: Math.floor(Math.random() * 100)
        })),
        energyConsumption: 120 + Math.floor(Math.random() * 80),
        anomalies: Math.floor(Math.random() * 5)
      };
    }
    
    const result: AnalyticsResult = {
      sourceId: id,
      statistics: {
        recordCount: source.recordCount,
        dateRange: {
          start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
          end: source.lastSync
        },
        fieldCount: 10 + Math.floor(Math.random() * 20)
      },
      insights
    };
    
    setAnalyticsResults(prev => ({
      ...prev,
      [id]: result
    }));
    
    setIsLoading(false);
    return result;
  };

  const tokenizeDataset = async (
    name: string, 
    description: string, 
    sourceIds: string[], 
    price: number,
    privacyLevel: string
  ): Promise<TokenizedDataset | null> => {
    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    const assetId = 'did:op:' + Math.random().toString(36).substring(2, 15);
    const tokenAddress = '0x' + Math.random().toString(36).substring(2, 15);
    
    const newDataset: TokenizedDataset = {
      id: `dataset-${Math.random().toString(36).substring(2, 9)}`,
      name,
      description,
      sourceIds,
      assetId,
      tokenAddress,
      price,
      createdAt: new Date().toISOString(),
      marketplaceUrl: `https://market.oceanprotocol.com/asset/${assetId.replace('did:op:', '')}`,
      privacyLevel
    };
    
    setTokenizedDatasets(prev => [...prev, newDataset]);
    setIsLoading(false);
    
    return newDataset;
  };

  return (
    <DataContext.Provider
      value={{
        dataSources,
        analyticsResults,
        tokenizedDatasets,
        isLoading,
        addDataSource,
        removeDataSource,
        syncDataSource,
        analyzeDataSource,
        tokenizeDataset
      }}
    >
      {children}
    </DataContext.Provider>
  );
}

export function useData() {
  const context = useContext(DataContext);
  if (context === undefined) {
    throw new Error('useData muss innerhalb eines DataProviders verwendet werden');
  }
  return context;
}