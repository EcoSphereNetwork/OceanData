// tests/components/DataTokenizationDashboard.test.js

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import DataTokenizationDashboard from '../../src/components/DataTokenizationDashboard';

// Mock-Daten für die Tests
const mockDataSources = [
  { 
    id: 'browser-chrome', 
    name: 'Chrome Browser History', 
    type: 'browser',
    recordCount: 4582,
    dateRange: 'Jan 1, 2024 - Mar 14, 2025',
    status: 'connected',
    lastSync: '2025-03-14T08:30:00Z',
    privacyLevel: 'medium',
    estimatedValue: 3.4
  },
  { 
    id: 'smartwatch-fitbit', 
    name: 'Fitbit Health Data', 
    type: 'smartwatch',
    recordCount: 8760,
    dateRange: 'Jan 1, 2024 - Mar 14, 2025',
    status: 'connected',
    lastSync: '2025-03-14T07:15:00Z',
    privacyLevel: 'high',
    estimatedValue: 5.2
  }
];

// Mock-Analyseergebnisse
const mockAnalysisResults = {
  sourceType: 'browser',
  recordCount: 4582,
  valueEstimate: 3.4,
  anomalies: {
    count: 32,
    percentage: 0.7,
    insights: [
      { feature: 'duration', description: 'Unusually long sessions detected' }
    ]
  },
  timePatterns: {
    peakUsageTimes: ['8:00 AM', '12:00 PM', '8:00 PM'],
    weekdayUsage: [65, 70, 68, 72, 75, 45, 40],
    weekdayLabels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
  },
  topFeatures: [
    { name: 'duration', importance: 0.85 },
    { name: 'time_of_day', importance: 0.72 },
    { name: 'domain_category', importance: 0.65 }
  ],
  valuationFactors: {
    dataSize: 0.68,
    dataQuality: 0.85,
    uniqueness: 0.72,
    timeRelevance: 0.95
  }
};

// Mock für API-Aufrufe
jest.mock('../../src/api/oceanDataApi', () => ({
  getDataSources: jest.fn().mockResolvedValue(mockDataSources),
  analyzeData: jest.fn().mockImplementation((dataSourceId) => {
    return new Promise((resolve) => {
      // Simuliere Verzögerung für die Analyse
      setTimeout(() => {
        resolve(mockAnalysisResults);
      }, 500);
    });
  }),
  tokenizeData: jest.fn().mockImplementation(() => {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          success: true,
          tokenSymbol: 'DT123456',
          tokenAddress: '0x1234567890abcdef1234567890abcdef12345678',
          marketplaceUrl: 'https://market.oceanprotocol.com/asset/12345',
          tokenPrice: 3.4,
          createdAt: '2025-03-14T10:30:00Z'
        });
      }, 500);
    });
  })
}));

describe('DataTokenizationDashboard', () => {
  // Rendere die Komponente vor jedem Test
  beforeEach(() => {
    render(<DataTokenizationDashboard />);
  });

  test('sollte die Komponente rendern und Datenquellen anzeigen', async () => {
    // Überprüfe, ob der Titel angezeigt wird
    expect(screen.getByText('Data Monetization Dashboard')).toBeInTheDocument();
    
    // Überprüfe, ob die Datenquellen geladen wurden
    await waitFor(() => {
      expect(screen.getByText('Chrome Browser History')).toBeInTheDocument();
      expect(screen.getByText('Fitbit Health Data')).toBeInTheDocument();
    });
  });

  test('sollte eine Datenquelle auswählen und analysieren können', async () => {
    // Warte auf das Laden der Datenquellen
    await waitFor(() => {
      expect(screen.getByText('Chrome Browser History')).toBeInTheDocument();
    });
    
    // Wähle eine Datenquelle aus
    fireEvent.click(screen.getByText('Chrome Browser History'));
    
    // Überprüfe, ob die Datenquelle ausgewählt wurde
    expect(screen.getByText('Chrome Browser History').closest('div')).toHaveClass('border-blue-500');
    
    // Klicke auf "Start Analysis"
    fireEvent.click(screen.getByText('Start Analysis'));
    
    // Überprüfe, ob die Analyse gestartet wurde (Ladeindikator angezeigt wird)
    expect(screen.getByText('Analyzing Chrome Browser History...')).toBeInTheDocument();
    
    // Warte auf das Abschließen der Analyse
    await waitFor(() => {
      expect(screen.getByText('Estimated Value:')).toBeInTheDocument();
      expect(screen.getByText('3.40 OCEAN')).toBeInTheDocument();
    }, { timeout: 1000 });
  });

  test('sollte Daten tokenisieren können, nachdem sie analysiert wurden', async () => {
    // Warte auf das Laden der Datenquellen
    await waitFor(() => {
      expect(screen.getByText('Chrome Browser History')).toBeInTheDocument();
    });
    
    // Wähle eine Datenquelle aus
    fireEvent.click(screen.getByText('Chrome Browser History'));
    
    // Starte die Analyse
    fireEvent.click(screen.getByText('Start Analysis'));
    
    // Warte auf das Abschließen der Analyse
    await waitFor(() => {
      expect(screen.getByText('3.40 OCEAN')).toBeInTheDocument();
    }, { timeout: 1000 });
    
    // Tokenisiere die Daten
    fireEvent.click(screen.getByText('Tokenize Data'));
    
    // Überprüfe, ob der Tokenisierungsprozess gestartet wurde
    expect(screen.getByText('Tokenizing data...')).toBeInTheDocument();
    
    // Warte auf das Abschließen der Tokenisierung
    await waitFor(() => {
      expect(screen.getByText('Successfully Tokenized!')).toBeInTheDocument();
      expect(screen.getByText('DT123456')).toBeInTheDocument();
    }, { timeout: 1000 });
    
    // Überprüfe, ob der "View on Ocean Market"-Button angezeigt wird
    expect(screen.getByText('View on Ocean Market')).toBeInTheDocument();
  });

  test('sollte die Analyse-Detailansicht anzeigen, wenn Analyseergebnisse verfügbar sind', async () => {
    // Warte auf das Laden der Datenquellen
    await waitFor(() => {
      expect(screen.getByText('Chrome Browser History')).toBeInTheDocument();
    });
    
    // Wähle eine Datenquelle aus und starte die Analyse
    fireEvent.click(screen.getByText('Chrome Browser History'));
    fireEvent.click(screen.getByText('Start Analysis'));
    
    // Warte auf das Abschließen der Analyse
    await waitFor(() => {
      expect(screen.getByText('3.40 OCEAN')).toBeInTheDocument();
    }, { timeout: 1000 });
    
    // Überprüfe, ob die Analyse-Detailansicht angezeigt wird
    expect(screen.getByText('Data Analysis Results')).toBeInTheDocument();
    
    // Überprüfe, ob die Tabs für verschiedene Analyseergebnisse angezeigt werden
    expect(screen.getByText('Key Insights')).toBeInTheDocument();
    expect(screen.getByText('Time Patterns')).toBeInTheDocument();
    expect(screen.getByText('Valuation Factors')).toBeInTheDocument();
    expect(screen.getByText('Potential Use Cases')).toBeInTheDocument();
    
    // Wechsle zum Tab "Valuation Factors"
    fireEvent.click(screen.getByText('Valuation Factors'));
    
    // Überprüfe, ob die Bewertungsfaktoren angezeigt werden
    await waitFor(() => {
      expect(screen.getByText('Value Estimate')).toBeInTheDocument();
      expect(screen.getByText('OCEAN Tokens')).toBeInTheDocument();
    });
  });
});
