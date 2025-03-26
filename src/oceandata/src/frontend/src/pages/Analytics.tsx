import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useData, DataSource, AnalyticsResult } from '@context/DataContext';
import { 
  AnalyticsIcon, 
  ChartIcon,
  DataSourceIcon,
  PlusIcon
} from '@components/icons';

// Smolitux-UI Komponenten importieren
import { BarChart, LineChart, DonutChart } from '@smolitux/charts';

// Komponente für die Visualisierung der Analyseergebnisse
const AnalyticsVisualizer = ({ source, result }: { source: DataSource; result: AnalyticsResult }) => {
  // Bestimme den Typ der Datenquelle und zeige entsprechende Visualisierungen an
  if (source.type === 'browser') {
    const { topDomains, usageByHour, categories } = result.insights;
    
    // Konvertiere die Daten für die Diagramme
    const domainData = Object.entries(topDomains).map(([domain, count]) => ({
      domain,
      visits: count
    })).sort((a, b) => (b.visits as number) - (a.visits as number)).slice(0, 5);
    
    const categoryData = Object.entries(categories).map(([category, percentage]) => ({
      category,
      percentage
    }));
    
    return (
      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Top-Domains</h3>
          <div className="h-64">
            <BarChart
              data={domainData}
              index="domain"
              categories={["visits"]}
              colors={["blue"]}
              valueFormatter={(value) => `${value} Besuche`}
              showLegend={false}
            />
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Nutzung nach Stunde</h3>
            <div className="h-64">
              <LineChart
                data={usageByHour}
                index="hour"
                categories={["visits"]}
                colors={["ocean"]}
                valueFormatter={(value) => `${value} Besuche`}
                showLegend={false}
              />
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Kategorien</h3>
            <div className="h-64">
              <DonutChart
                data={categoryData}
                index="category"
                category="percentage"
                colors={["blue", "green", "amber", "purple", "ocean"]}
                valueFormatter={(value) => `${value}%`}
              />
            </div>
          </div>
        </div>
      </div>
    );
  } else if (source.type === 'smartwatch') {
    const { averageHeartRate, stepsByDay, sleepQuality, caloriesBurned } = result.insights;
    
    // Konvertiere die Daten für die Diagramme
    const sleepData = Object.entries(sleepQuality).map(([type, percentage]) => ({
      type,
      percentage
    }));
    
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-red-100 text-red-600">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Durchschnittliche Herzfrequenz</p>
                <p className="text-2xl font-semibold text-gray-900">{averageHeartRate} BPM</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-green-100 text-green-600">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Gesamtschritte</p>
                <p className="text-2xl font-semibold text-gray-900">{stepsByDay.reduce((sum, day) => sum + (day.steps as number), 0).toLocaleString()}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-purple-100 text-purple-600">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Tiefschlafanteil</p>
                <p className="text-2xl font-semibold text-gray-900">{sleepQuality.deep}%</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-orange-100 text-orange-600">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.879 16.121A3 3 0 1012.015 11L11 14H9c0 .768.293 1.536.879 2.121z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Verbrannte Kalorien</p>
                <p className="text-2xl font-semibold text-gray-900">{caloriesBurned}</p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Schritte pro Tag</h3>
            <div className="h-64">
              <BarChart
                data={stepsByDay}
                index="day"
                categories={["steps"]}
                colors={["green"]}
                valueFormatter={(value) => `${value.toLocaleString()} Schritte`}
                showLegend={false}
              />
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Schlafqualität</h3>
            <div className="h-64">
              <DonutChart
                data={sleepData}
                index="type"
                category="percentage"
                colors={["purple", "blue", "teal", "gray"]}
                valueFormatter={(value) => `${value}%`}
              />
            </div>
          </div>
        </div>
      </div>
    );
  } else if (source.type.startsWith('iot_')) {
    const { usagePatterns, energyConsumption, anomalies } = result.insights;
    
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-amber-100 text-amber-600">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Energieverbrauch</p>
                <p className="text-2xl font-semibold text-gray-900">{energyConsumption} kWh</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-red-100 text-red-600">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Anomalien</p>
                <p className="text-2xl font-semibold text-gray-900">{anomalies}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-blue-100 text-blue-600">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Datensätze</p>
                <p className="text-2xl font-semibold text-gray-900">{source.recordCount.toLocaleString()}</p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Nutzungsmuster nach Stunde</h3>
          <div className="h-64">
            <LineChart
              data={usagePatterns}
              index="hour"
              categories={["activity"]}
              colors={["amber"]}
              valueFormatter={(value) => `${value}%`}
              showLegend={false}
            />
          </div>
        </div>
      </div>
    );
  }
  
  // Fallback für unbekannte Quellentypen
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="text-center py-8">
        <ChartIcon className="w-12 h-12 mx-auto text-gray-400" />
        <h3 className="mt-2 text-lg font-medium text-gray-900">Keine spezifischen Visualisierungen verfügbar</h3>
        <p className="mt-1 text-gray-500">
          Für diesen Datenquellentyp sind keine spezifischen Visualisierungen verfügbar.
        </p>
      </div>
      
      <div className="mt-4 border-t border-gray-200 pt-4">
        <h4 className="text-md font-medium text-gray-900 mb-2">Rohdaten-Statistiken</h4>
        <dl className="grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2">
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Datensätze</dt>
            <dd className="mt-1 text-sm text-gray-900">{result.statistics.recordCount.toLocaleString()}</dd>
          </div>
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Felder</dt>
            <dd className="mt-1 text-sm text-gray-900">{result.statistics.fieldCount}</dd>
          </div>
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Zeitraum Start</dt>
            <dd className="mt-1 text-sm text-gray-900">
              {result.statistics.dateRange.start 
                ? new Date(result.statistics.dateRange.start).toLocaleDateString('de-DE') 
                : 'Nicht verfügbar'}
            </dd>
          </div>
          <div className="sm:col-span-1">
            <dt className="text-sm font-medium text-gray-500">Zeitraum Ende</dt>
            <dd className="mt-1 text-sm text-gray-900">
              {result.statistics.dateRange.end 
                ? new Date(result.statistics.dateRange.end).toLocaleDateString('de-DE') 
                : 'Nicht verfügbar'}
            </dd>
          </div>
        </dl>
      </div>
    </div>
  );
};

// Hauptkomponente für die Analyseseite
const Analytics = () => {
  const { dataSources, analyticsResults, analyzeDataSource, isLoading } = useData();
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Wähle automatisch die erste Datenquelle, wenn keine ausgewählt ist
  useEffect(() => {
    if (dataSources.length > 0 && !selectedSourceId) {
      setSelectedSourceId(dataSources[0].id);
    }
  }, [dataSources, selectedSourceId]);

  const handleAnalyze = async () => {
    if (!selectedSourceId) return;
    
    setIsAnalyzing(true);
    try {
      await analyzeDataSource(selectedSourceId);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const selectedSource = dataSources.find(source => source.id === selectedSourceId);
  const analysisResult = selectedSourceId ? analyticsResults[selectedSourceId] : null;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Datenanalyse</h1>
      </div>

      <div className="bg-white shadow rounded-lg p-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <div className="mb-4 md:mb-0">
            <label htmlFor="data-source" className="block text-sm font-medium text-gray-700 mb-1">
              Datenquelle auswählen
            </label>
            <select
              id="data-source"
              value={selectedSourceId || ''}
              onChange={(e) => setSelectedSourceId(e.target.value || null)}
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-ocean-500 focus:border-ocean-500 sm:text-sm rounded-md"
              disabled={isLoading || isAnalyzing}
            >
              <option value="">Bitte wählen...</option>
              {dataSources.map((source) => (
                <option key={source.id} value={source.id}>
                  {source.name} ({source.recordCount.toLocaleString()} Datensätze)
                </option>
              ))}
            </select>
          </div>
          
          <button
            onClick={handleAnalyze}
            disabled={!selectedSourceId || isLoading || isAnalyzing}
            className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500 disabled:opacity-50"
          >
            {isAnalyzing ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Analysiere...
              </>
            ) : (
              <>
                <AnalyticsIcon className="w-5 h-5 mr-2" />
                Analysieren
              </>
            )}
          </button>
        </div>
      </div>

      {isLoading && !isAnalyzing && (
        <div className="flex justify-center items-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-ocean-600"></div>
          <span className="ml-3 text-lg text-gray-600">Wird geladen...</span>
        </div>
      )}

      {isAnalyzing && (
        <div className="bg-white shadow rounded-lg p-8">
          <div className="text-center">
            <AnalyticsIcon className="w-12 h-12 mx-auto text-ocean-500 animate-pulse" />
            <h3 className="mt-4 text-lg font-medium text-gray-900">Daten werden analysiert</h3>
            <p className="mt-2 text-gray-500">
              Bitte warten Sie, während wir Ihre Daten analysieren. Dies kann einige Momente dauern.
            </p>
            <div className="mt-6 max-w-md mx-auto">
              <div className="relative">
                <div className="overflow-hidden h-2 text-xs flex rounded bg-ocean-100">
                  <div className="animate-pulse-slow w-full flex flex-col text-center whitespace-nowrap text-white justify-center bg-ocean-500"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {!isLoading && !isAnalyzing && selectedSource && analysisResult && (
        <AnalyticsVisualizer source={selectedSource} result={analysisResult} />
      )}

      {!isLoading && !isAnalyzing && selectedSource && !analysisResult && (
        <div className="bg-white shadow rounded-lg p-8">
          <div className="text-center">
            <ChartIcon className="w-12 h-12 mx-auto text-gray-400" />
            <h3 className="mt-2 text-lg font-medium text-gray-900">Keine Analyseergebnisse</h3>
            <p className="mt-1 text-gray-500">
              Klicken Sie auf "Analysieren", um Ihre Daten zu analysieren und Erkenntnisse zu gewinnen.
            </p>
            <button
              onClick={handleAnalyze}
              className="mt-4 inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
            >
              <AnalyticsIcon className="w-5 h-5 mr-2" />
              Jetzt analysieren
            </button>
          </div>
        </div>
      )}

      {!isLoading && !isAnalyzing && !selectedSource && (
        <div className="bg-white shadow rounded-lg p-8">
          <div className="text-center">
            <DataSourceIcon className="w-12 h-12 mx-auto text-gray-400" />
            <h3 className="mt-2 text-lg font-medium text-gray-900">Keine Datenquelle ausgewählt</h3>
            <p className="mt-1 text-gray-500">
              Bitte wählen Sie eine Datenquelle aus, um mit der Analyse zu beginnen.
            </p>
            {dataSources.length === 0 && (
              <div className="mt-4">
                <p className="text-sm text-gray-500">
                  Sie haben noch keine Datenquellen hinzugefügt.
                </p>
                <Link to="/data-sources" className="mt-2 inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500">
                  <PlusIcon className="w-5 h-5 mr-2" />
                  Datenquelle hinzufügen
                </Link>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Analytics;