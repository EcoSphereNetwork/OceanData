import { useState, useEffect } from 'react';
import { useAuth } from '@context/AuthContext';
import { SearchIcon, MarketplaceIcon } from '@components/icons';

// Typdefinitionen für Marktplatz-Datensätze
interface MarketplaceDataset {
  id: string;
  name: string;
  description: string;
  price: number;
  publisher: string;
  publisherAddress: string;
  category: string;
  assetId: string;
  tokenAddress: string;
  createdAt: string;
  downloads: number;
  rating: number;
  tags: string[];
}

// Komponente für die Marktplatzseite
const Marketplace = () => {
  const { user } = useAuth();
  const [datasets, setDatasets] = useState<MarketplaceDataset[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'price' | 'rating' | 'date'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Simulierte Daten für den Marktplatz
  useEffect(() => {
    const fetchMarketplaceData = async () => {
      setIsLoading(true);
      
      try {
        // In einer echten Anwendung würde hier eine API-Anfrage erfolgen
        // Für die Demo verwenden wir simulierte Daten
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const mockDatasets: MarketplaceDataset[] = [
          {
            id: '1',
            name: 'Gesundheitsdaten-Bundle',
            description: 'Umfassende Gesundheitsdaten mit Herzfrequenz, Schritten und Schlafmustern über einen Zeitraum von 6 Monaten.',
            price: 2.5,
            publisher: 'HealthDataProvider',
            publisherAddress: '0x1234...5678',
            category: 'health',
            assetId: 'did:op:1234567890abcdef',
            tokenAddress: '0xabcd...ef12',
            createdAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
            downloads: 42,
            rating: 4.7,
            tags: ['health', 'fitness', 'sleep']
          },
          {
            id: '2',
            name: 'Browser-Nutzungsdaten',
            description: 'Anonymisierte Browser-Verlaufsdaten mit Kategorisierung und Nutzungsmustern.',
            price: 1.2,
            publisher: 'WebAnalytics',
            publisherAddress: '0x2345...6789',
            category: 'web',
            assetId: 'did:op:2345678901bcdef',
            tokenAddress: '0xbcde...f123',
            createdAt: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString(),
            downloads: 78,
            rating: 4.2,
            tags: ['web', 'browser', 'analytics']
          },
          {
            id: '3',
            name: 'Smart Home Energiedaten',
            description: 'Detaillierte Energieverbrauchsdaten von Smart-Home-Geräten mit stündlicher Auflösung.',
            price: 1.8,
            publisher: 'EnergyInsights',
            publisherAddress: '0x3456...7890',
            category: 'iot',
            assetId: 'did:op:3456789012cdef',
            tokenAddress: '0xcdef...1234',
            createdAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
            downloads: 31,
            rating: 4.5,
            tags: ['iot', 'energy', 'smart-home']
          },
          {
            id: '4',
            name: 'Einkaufsverhalten-Analyse',
            description: 'Anonymisierte Daten zum Einkaufsverhalten mit Produktkategorien und Zeitmustern.',
            price: 3.2,
            publisher: 'ShoppingInsights',
            publisherAddress: '0x4567...8901',
            category: 'retail',
            assetId: 'did:op:4567890123def',
            tokenAddress: '0xdef1...2345',
            createdAt: new Date(Date.now() - 21 * 24 * 60 * 60 * 1000).toISOString(),
            downloads: 105,
            rating: 4.8,
            tags: ['retail', 'shopping', 'consumer']
          },
          {
            id: '5',
            name: 'Verkehrsdaten-Bundle',
            description: 'Anonymisierte Verkehrsdaten mit Routeninformationen und Verkehrsaufkommen.',
            price: 2.1,
            publisher: 'MobilityData',
            publisherAddress: '0x5678...9012',
            category: 'mobility',
            assetId: 'did:op:5678901234ef',
            tokenAddress: '0xef12...3456',
            createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
            downloads: 67,
            rating: 4.3,
            tags: ['mobility', 'traffic', 'transportation']
          }
        ];
        
        setDatasets(mockDatasets);
      } catch (error) {
        console.error('Fehler beim Laden der Marktplatzdaten:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchMarketplaceData();
  }, []);

  // Filtern und Sortieren der Datensätze
  const filteredAndSortedDatasets = datasets
    .filter(dataset => 
      (searchTerm === '' || 
       dataset.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
       dataset.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
       dataset.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))) &&
      (selectedCategory === null || dataset.category === selectedCategory)
    )
    .sort((a, b) => {
      if (sortBy === 'price') {
        return sortOrder === 'asc' ? a.price - b.price : b.price - a.price;
      } else if (sortBy === 'rating') {
        return sortOrder === 'asc' ? a.rating - b.rating : b.rating - a.rating;
      } else {
        return sortOrder === 'asc' 
          ? new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
          : new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      }
    });

  // Kategorien für die Filterung
  const categories = [
    { id: 'health', name: 'Gesundheit' },
    { id: 'web', name: 'Web & Browser' },
    { id: 'iot', name: 'IoT & Smart Home' },
    { id: 'retail', name: 'Einzelhandel' },
    { id: 'mobility', name: 'Mobilität' }
  ];

  // Funktion zum Umschalten der Sortierreihenfolge
  const toggleSortOrder = () => {
    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
  };

  // Funktion zum Kaufen eines Datensatzes
  const handleBuyDataset = (dataset: MarketplaceDataset) => {
    if (!user?.walletAddress) {
      alert('Bitte verbinden Sie zuerst Ihre Wallet, um Datensätze zu kaufen.');
      return;
    }
    
    alert(`Simulierter Kauf: ${dataset.name} für ${dataset.price} OCEAN`);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Ocean Protocol Marktplatz</h1>
      </div>

      {/* Suchleiste und Filter */}
      <div className="bg-white shadow rounded-lg p-4">
        <div className="flex flex-col md:flex-row md:items-center md:space-x-4">
          <div className="flex-1 mb-4 md:mb-0">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <SearchIcon className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-ocean-500 focus:border-ocean-500 sm:text-sm"
                placeholder="Datensätze durchsuchen..."
              />
            </div>
          </div>
          
          <div className="flex flex-col sm:flex-row sm:space-x-4 space-y-2 sm:space-y-0">
            <div>
              <select
                value={selectedCategory || ''}
                onChange={(e) => setSelectedCategory(e.target.value || null)}
                className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-ocean-500 focus:border-ocean-500 sm:text-sm rounded-md"
              >
                <option value="">Alle Kategorien</option>
                {categories.map(category => (
                  <option key={category.id} value={category.id}>{category.name}</option>
                ))}
              </select>
            </div>
            
            <div>
              <div className="flex items-center space-x-2">
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                  className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-ocean-500 focus:border-ocean-500 sm:text-sm rounded-md"
                >
                  <option value="date">Datum</option>
                  <option value="price">Preis</option>
                  <option value="rating">Bewertung</option>
                </select>
                
                <button
                  onClick={toggleSortOrder}
                  className="p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-ocean-500 focus:border-ocean-500"
                >
                  {sortOrder === 'asc' ? (
                    <svg className="h-5 w-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12" />
                    </svg>
                  ) : (
                    <svg className="h-5 w-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4h13M3 8h9m-9 4h9m5-4v12m0 0l-4-4m4 4l4-4" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Datensatz-Liste */}
      <div className="space-y-4">
        {isLoading && (
          <div className="flex justify-center items-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-ocean-600"></div>
            <span className="ml-3 text-lg text-gray-600">Marktplatz wird geladen...</span>
          </div>
        )}

        {!isLoading && filteredAndSortedDatasets.length === 0 && (
          <div className="bg-white shadow rounded-lg p-8 text-center">
            <MarketplaceIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-lg font-medium text-gray-900">Keine Datensätze gefunden</h3>
            <p className="mt-1 text-gray-500">
              Versuchen Sie, Ihre Suchkriterien zu ändern oder einen anderen Filter auszuwählen.
            </p>
          </div>
        )}

        {!isLoading && filteredAndSortedDatasets.map((dataset) => (
          <div key={dataset.id} className="bg-white shadow rounded-lg overflow-hidden">
            <div className="p-4 sm:p-6">
              <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start">
                <div className="flex-1">
                  <div className="flex items-center">
                    <h3 className="text-lg font-medium text-gray-900">{dataset.name}</h3>
                    <span className="ml-2 px-2 py-1 text-xs font-medium rounded-full bg-ocean-100 text-ocean-800">
                      {dataset.price.toFixed(2)} OCEAN
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-gray-500">{dataset.description}</p>
                  
                  <div className="mt-2 flex items-center">
                    <div className="flex items-center">
                      {[...Array(5)].map((_, i) => (
                        <svg
                          key={i}
                          className={`h-4 w-4 ${
                            i < Math.floor(dataset.rating) ? 'text-yellow-400' : 'text-gray-300'
                          }`}
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                        </svg>
                      ))}
                      <span className="ml-1 text-sm text-gray-500">{dataset.rating.toFixed(1)}</span>
                    </div>
                    <span className="mx-2 text-gray-300">•</span>
                    <span className="text-sm text-gray-500">{dataset.downloads} Downloads</span>
                    <span className="mx-2 text-gray-300">•</span>
                    <span className="text-sm text-gray-500">
                      Veröffentlicht: {new Date(dataset.createdAt).toLocaleDateString('de-DE')}
                    </span>
                  </div>
                  
                  <div className="mt-2 flex flex-wrap gap-1">
                    {dataset.tags.map(tag => (
                      <span 
                        key={tag} 
                        className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
                
                <div className="mt-4 sm:mt-0 sm:ml-4 flex flex-col items-end">
                  <button
                    onClick={() => handleBuyDataset(dataset)}
                    className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
                  >
                    Kaufen
                  </button>
                  <a
                    href={`https://market.oceanprotocol.com/asset/${dataset.assetId.replace('did:op:', '')}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-2 text-xs text-ocean-600 hover:text-ocean-800"
                  >
                    Details auf Ocean Market →
                  </a>
                </div>
              </div>
              
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="flex items-center text-sm text-gray-500">
                  <span>Anbieter: {dataset.publisher}</span>
                  <span className="mx-2">•</span>
                  <span className="text-xs">
                    {dataset.publisherAddress.substring(0, 6)}...
                    {dataset.publisherAddress.substring(dataset.publisherAddress.length - 4)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Marketplace;