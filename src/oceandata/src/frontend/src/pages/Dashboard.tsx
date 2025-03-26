import { useEffect, useState } from 'react';
import { useData } from '@context/DataContext';
import { useAuth } from '@context/AuthContext';
import { 
  DataSourceIcon, 
  AnalyticsIcon, 
  TokenizationIcon, 
  MarketplaceIcon,
  ChartIcon
} from '@components/icons';
import { Link } from 'react-router-dom';

// Smolitux-UI Komponenten importieren
import { Card, Flex, Grid, Text, Title, Metric, ProgressBar } from '@smolitux/core';
import { AreaChart, BarChart, DonutChart } from '@smolitux/charts';

const Dashboard = () => {
  const { user } = useAuth();
  const { dataSources, tokenizedDatasets, analyticsResults } = useData();
  const [totalValue, setTotalValue] = useState(0);
  const [chartData, setChartData] = useState<any[]>([]);
  const [sourceDistribution, setSourceDistribution] = useState<any[]>([]);

  useEffect(() => {
    // Berechne den Gesamtwert aller Datenquellen
    const value = dataSources.reduce((sum, source) => sum + source.estimatedValue, 0);
    setTotalValue(value);

    // Erstelle Daten für das Wertdiagramm (letzte 7 Tage)
    const today = new Date();
    const last7Days = Array.from({ length: 7 }, (_, i) => {
      const date = new Date(today);
      date.setDate(date.getDate() - (6 - i));
      return {
        date: date.toLocaleDateString('de-DE', { weekday: 'short' }),
        value: value * (0.85 + (i * 0.025) + (Math.random() * 0.05))
      };
    });
    setChartData(last7Days);

    // Erstelle Daten für die Quellenverteilung
    const distribution = [
      { name: 'Browser', value: dataSources.filter(s => s.type === 'browser').length },
      { name: 'Smartwatch', value: dataSources.filter(s => s.type === 'smartwatch').length },
      { name: 'IoT', value: dataSources.filter(s => s.type.startsWith('iot_')).length }
    ].filter(item => item.value > 0);
    
    setSourceDistribution(distribution);
  }, [dataSources]);

  // Berechne die Gesamtanzahl der Datensätze
  const totalRecords = dataSources.reduce((sum, source) => sum + source.recordCount, 0);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <div className="text-sm text-gray-500">
          Letzte Aktualisierung: {new Date().toLocaleString('de-DE')}
        </div>
      </div>

      {/* Übersichtskarten */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center">
            <div className="p-3 rounded-full bg-blue-100 text-blue-600">
              <DataSourceIcon className="w-6 h-6" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Datenquellen</p>
              <p className="text-2xl font-semibold text-gray-900">{dataSources.length}</p>
            </div>
          </div>
          <div className="mt-4">
            <Link to="/data-sources" className="text-sm text-blue-600 hover:text-blue-800">
              Alle anzeigen →
            </Link>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center">
            <div className="p-3 rounded-full bg-green-100 text-green-600">
              <AnalyticsIcon className="w-6 h-6" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Datensätze</p>
              <p className="text-2xl font-semibold text-gray-900">{totalRecords.toLocaleString()}</p>
            </div>
          </div>
          <div className="mt-4">
            <Link to="/analytics" className="text-sm text-green-600 hover:text-green-800">
              Analysieren →
            </Link>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center">
            <div className="p-3 rounded-full bg-purple-100 text-purple-600">
              <TokenizationIcon className="w-6 h-6" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Tokenisierte Datensätze</p>
              <p className="text-2xl font-semibold text-gray-900">{tokenizedDatasets.length}</p>
            </div>
          </div>
          <div className="mt-4">
            <Link to="/tokenization" className="text-sm text-purple-600 hover:text-purple-800">
              Tokenisieren →
            </Link>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center">
            <div className="p-3 rounded-full bg-ocean-100 text-ocean-600">
              <MarketplaceIcon className="w-6 h-6" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Geschätzter Wert</p>
              <p className="text-2xl font-semibold text-gray-900">{totalValue.toFixed(2)} OCEAN</p>
            </div>
          </div>
          <div className="mt-4">
            <Link to="/marketplace" className="text-sm text-ocean-600 hover:text-ocean-800">
              Zum Marktplatz →
            </Link>
          </div>
        </div>
      </div>

      {/* Diagramme und Analysen */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Wertentwicklung */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Wertentwicklung</h2>
          <div className="h-64">
            <AreaChart
              data={chartData}
              index="date"
              categories={["value"]}
              colors={["ocean"]}
              valueFormatter={(value) => `${value.toFixed(2)} OCEAN`}
              showLegend={false}
              showGridLines={false}
              showAnimation={true}
            />
          </div>
        </div>

        {/* Quellenverteilung */}
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Datenquellenverteilung</h2>
          <div className="h-64 flex items-center justify-center">
            {sourceDistribution.length > 0 ? (
              <DonutChart
                data={sourceDistribution}
                category="value"
                index="name"
                colors={["blue", "green", "amber"]}
                showAnimation={true}
                showTooltip={true}
              />
            ) : (
              <div className="text-center text-gray-500">
                <ChartIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>Keine Daten verfügbar</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Datenquellen-Übersicht */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Aktive Datenquellen</h3>
        </div>
        <div className="divide-y divide-gray-200">
          {dataSources.length > 0 ? (
            dataSources.map((source) => (
              <div key={source.id} className="px-4 py-4 sm:px-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className={`p-2 rounded-md ${
                      source.type === 'browser' ? 'bg-blue-100 text-blue-600' :
                      source.type === 'smartwatch' ? 'bg-green-100 text-green-600' :
                      'bg-amber-100 text-amber-600'
                    }`}>
                      <DataSourceIcon className="w-5 h-5" />
                    </div>
                    <div className="ml-3">
                      <h4 className="text-sm font-medium text-gray-900">{source.name}</h4>
                      <p className="text-xs text-gray-500">
                        {source.recordCount.toLocaleString()} Datensätze
                        {source.lastSync && ` • Letzte Synchronisierung: ${new Date(source.lastSync).toLocaleString('de-DE')}`}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center">
                    <span className="text-sm font-medium text-gray-900 mr-2">
                      {source.estimatedValue.toFixed(2)} OCEAN
                    </span>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      source.status === 'connected' ? 'bg-green-100 text-green-800' :
                      source.status === 'error' ? 'bg-red-100 text-red-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {source.status === 'connected' ? 'Verbunden' :
                       source.status === 'error' ? 'Fehler' : 'Getrennt'}
                    </span>
                  </div>
                </div>
                <div className="mt-2">
                  <div className="relative pt-1">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-xs font-semibold inline-block text-ocean-600">
                          Datenschutzniveau: {source.privacyLevel === 'low' ? 'Niedrig' : 
                                             source.privacyLevel === 'medium' ? 'Mittel' : 
                                             source.privacyLevel === 'high' ? 'Hoch' : 'Nur Berechnung'}
                        </span>
                      </div>
                    </div>
                    <div className="overflow-hidden h-2 mb-1 text-xs flex rounded bg-ocean-100">
                      <div 
                        className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-ocean-500"
                        style={{ width: `${source.privacyLevel === 'low' ? 33 : 
                                         source.privacyLevel === 'medium' ? 66 : 
                                         source.privacyLevel === 'high' ? 100 : 50}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="px-4 py-8 text-center text-gray-500">
              <DataSourceIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p className="mb-2">Keine Datenquellen verbunden</p>
              <Link to="/data-sources" className="text-ocean-600 hover:text-ocean-800 font-medium">
                Datenquelle hinzufügen
              </Link>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;