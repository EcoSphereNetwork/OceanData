import { useState } from 'react';
import { useData, DataSource } from '@context/DataContext';
import { 
  DataSourceIcon, 
  PlusIcon, 
  RefreshIcon, 
  TrashIcon,
  SearchIcon
} from '@components/icons';

// Komponenten für das Hinzufügen einer neuen Datenquelle
const AddDataSourceModal = ({ isOpen, onClose, onAdd }: { 
  isOpen: boolean; 
  onClose: () => void;
  onAdd: (source: Omit<DataSource, 'id'>) => void;
}) => {
  const [name, setName] = useState('');
  const [type, setType] = useState('browser');
  const [privacyLevel, setPrivacyLevel] = useState<'low' | 'medium' | 'high' | 'compute_only'>('medium');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !type) return;
    
    setIsSubmitting(true);
    
    try {
      await onAdd({
        name,
        type,
        status: 'connected',
        lastSync: null,
        recordCount: 0,
        privacyLevel,
        estimatedValue: 0
      });
      
      // Zurücksetzen des Formulars
      setName('');
      setType('browser');
      setPrivacyLevel('medium');
      onClose();
    } catch (error) {
      console.error('Fehler beim Hinzufügen der Datenquelle:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Neue Datenquelle hinzufügen</h3>
        </div>
        
        <form onSubmit={handleSubmit} className="px-4 py-5 sm:p-6">
          <div className="space-y-4">
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                Name
              </label>
              <input
                type="text"
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-ocean-500 focus:border-ocean-500 sm:text-sm"
                placeholder="z.B. Chrome Browser History"
                required
              />
            </div>
            
            <div>
              <label htmlFor="type" className="block text-sm font-medium text-gray-700">
                Typ
              </label>
              <select
                id="type"
                value={type}
                onChange={(e) => setType(e.target.value)}
                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-ocean-500 focus:border-ocean-500 sm:text-sm"
                required
              >
                <option value="browser">Browser</option>
                <option value="smartwatch">Smartwatch</option>
                <option value="iot_thermostat">IoT - Thermostat</option>
                <option value="iot_light">IoT - Beleuchtung</option>
                <option value="iot_security_camera">IoT - Sicherheitskamera</option>
              </select>
            </div>
            
            <div>
              <label htmlFor="privacy" className="block text-sm font-medium text-gray-700">
                Datenschutzniveau
              </label>
              <select
                id="privacy"
                value={privacyLevel}
                onChange={(e) => setPrivacyLevel(e.target.value as any)}
                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-ocean-500 focus:border-ocean-500 sm:text-sm"
                required
              >
                <option value="low">Niedrig (mehr Daten, höherer Wert)</option>
                <option value="medium">Mittel (ausgewogenes Verhältnis)</option>
                <option value="high">Hoch (weniger Daten, geringerer Wert)</option>
                <option value="compute_only">Nur Berechnung (keine Datenfreigabe)</option>
              </select>
              <p className="mt-1 text-xs text-gray-500">
                Das Datenschutzniveau bestimmt, wie viele Ihrer Daten anonymisiert oder entfernt werden.
              </p>
            </div>
          </div>
          
          <div className="mt-6 flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
            >
              Abbrechen
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500 disabled:opacity-50"
            >
              {isSubmitting ? 'Wird hinzugefügt...' : 'Hinzufügen'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// Hauptkomponente für die Datenquellen-Seite
const DataSources = () => {
  const { dataSources, addDataSource, removeDataSource, syncDataSource, isLoading } = useData();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [syncingSourceId, setSyncingSourceId] = useState<string | null>(null);
  const [removingSourceId, setRemovingSourceId] = useState<string | null>(null);

  // Filtern der Datenquellen basierend auf dem Suchbegriff
  const filteredSources = dataSources.filter(source => 
    source.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    source.type.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleAddSource = async (source: Omit<DataSource, 'id'>) => {
    await addDataSource(source);
  };

  const handleSyncSource = async (id: string) => {
    setSyncingSourceId(id);
    try {
      await syncDataSource(id);
    } finally {
      setSyncingSourceId(null);
    }
  };

  const handleRemoveSource = async (id: string) => {
    setRemovingSourceId(id);
    try {
      await removeDataSource(id);
    } finally {
      setRemovingSourceId(null);
    }
  };

  const getSourceTypeLabel = (type: string) => {
    switch (type) {
      case 'browser': return 'Browser';
      case 'smartwatch': return 'Smartwatch';
      case 'iot_thermostat': return 'IoT - Thermostat';
      case 'iot_light': return 'IoT - Beleuchtung';
      case 'iot_security_camera': return 'IoT - Sicherheitskamera';
      default: return type;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Datenquellen</h1>
        <button
          onClick={() => setIsModalOpen(true)}
          className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
        >
          <PlusIcon className="w-5 h-5 mr-2" />
          Datenquelle hinzufügen
        </button>
      </div>

      {/* Suchleiste */}
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <SearchIcon className="h-5 w-5 text-gray-400" />
        </div>
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-ocean-500 focus:border-ocean-500 sm:text-sm"
          placeholder="Datenquellen durchsuchen..."
        />
      </div>

      {/* Datenquellen-Liste */}
      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        {isLoading && (
          <div className="flex justify-center items-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-ocean-600"></div>
            <span className="ml-2 text-gray-600">Wird geladen...</span>
          </div>
        )}

        {!isLoading && filteredSources.length === 0 && (
          <div className="py-8 text-center">
            <DataSourceIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">Keine Datenquellen</h3>
            <p className="mt-1 text-sm text-gray-500">
              Fügen Sie Ihre erste Datenquelle hinzu, um mit der Datenmonetarisierung zu beginnen.
            </p>
            <div className="mt-6">
              <button
                onClick={() => setIsModalOpen(true)}
                className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
              >
                <PlusIcon className="w-5 h-5 mr-2" />
                Datenquelle hinzufügen
              </button>
            </div>
          </div>
        )}

        {!isLoading && filteredSources.length > 0 && (
          <ul className="divide-y divide-gray-200">
            {filteredSources.map((source) => (
              <li key={source.id}>
                <div className="px-4 py-4 sm:px-6">
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
                        <h3 className="text-sm font-medium text-gray-900">{source.name}</h3>
                        <div className="flex items-center text-xs text-gray-500">
                          <span className="mr-2">{getSourceTypeLabel(source.type)}</span>
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                            source.status === 'connected' ? 'bg-green-100 text-green-800' :
                            source.status === 'error' ? 'bg-red-100 text-red-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {source.status === 'connected' ? 'Verbunden' :
                             source.status === 'error' ? 'Fehler' : 'Getrennt'}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => handleSyncSource(source.id)}
                        disabled={syncingSourceId === source.id}
                        className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
                      >
                        <RefreshIcon className={`w-5 h-5 ${syncingSourceId === source.id ? 'animate-spin' : ''}`} />
                      </button>
                      <button
                        onClick={() => handleRemoveSource(source.id)}
                        disabled={removingSourceId === source.id}
                        className="p-2 rounded-md text-gray-400 hover:text-red-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                      >
                        <TrashIcon className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                  <div className="mt-2 sm:flex sm:justify-between">
                    <div className="sm:flex">
                      <p className="flex items-center text-sm text-gray-500">
                        {source.recordCount.toLocaleString()} Datensätze
                      </p>
                      {source.lastSync && (
                        <p className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0 sm:ml-6">
                          Letzte Synchronisierung: {new Date(source.lastSync).toLocaleString('de-DE')}
                        </p>
                      )}
                    </div>
                    <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0">
                      <span className="font-medium text-ocean-600">
                        {source.estimatedValue.toFixed(2)} OCEAN
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
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Modal zum Hinzufügen einer neuen Datenquelle */}
      <AddDataSourceModal 
        isOpen={isModalOpen} 
        onClose={() => setIsModalOpen(false)} 
        onAdd={handleAddSource} 
      />
    </div>
  );
};

export default DataSources;