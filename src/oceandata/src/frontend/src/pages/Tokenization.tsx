import { useState } from 'react';
import { useData, DataSource, TokenizedDataset } from '@context/DataContext';
import { 
  TokenizationIcon, 
  PlusIcon,
  CheckIcon
} from '@components/icons';

// Komponente für den Tokenisierungsprozess
const TokenizationWizard = ({ isOpen, onClose, onTokenize }: {
  isOpen: boolean;
  onClose: () => void;
  onTokenize: (name: string, description: string, sourceIds: string[], price: number, privacyLevel: string) => Promise<TokenizedDataset | null>;
}) => {
  const { dataSources } = useData();
  const [step, setStep] = useState(1);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([]);
  const [price, setPrice] = useState(1.0);
  const [privacyLevel, setPrivacyLevel] = useState('medium');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  const handleSourceToggle = (sourceId: string) => {
    setSelectedSourceIds(prev => 
      prev.includes(sourceId)
        ? prev.filter(id => id !== sourceId)
        : [...prev, sourceId]
    );
  };

  const handleNext = () => {
    if (step === 1 && selectedSourceIds.length === 0) {
      setError('Bitte wählen Sie mindestens eine Datenquelle aus.');
      return;
    }
    
    if (step === 2 && (!name || !description)) {
      setError('Bitte geben Sie einen Namen und eine Beschreibung ein.');
      return;
    }
    
    setError('');
    setStep(prev => prev + 1);
  };

  const handleBack = () => {
    setStep(prev => prev - 1);
  };

  const handleSubmit = async () => {
    if (selectedSourceIds.length === 0 || !name || !description) {
      setError('Bitte füllen Sie alle erforderlichen Felder aus.');
      return;
    }
    
    setIsSubmitting(true);
    setError('');
    
    try {
      await onTokenize(name, description, selectedSourceIds, price, privacyLevel);
      setStep(4); // Erfolgsschritt
    } catch (err) {
      setError('Ein Fehler ist aufgetreten. Bitte versuchen Sie es später erneut.');
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const calculateTotalValue = () => {
    return dataSources
      .filter(source => selectedSourceIds.includes(source.id))
      .reduce((sum, source) => sum + source.estimatedValue, 0);
  };

  const totalValue = calculateTotalValue();

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full mx-4">
        <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Daten tokenisieren</h3>
          <p className="mt-1 text-sm text-gray-500">
            Erstellen Sie einen tokenisierten Datensatz aus Ihren Datenquellen.
          </p>
        </div>
        
        {/* Fortschrittsanzeige */}
        <div className="px-4 py-3 sm:px-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="w-full">
              <div className="flex items-center">
                <div className={`flex items-center justify-center w-8 h-8 rounded-full ${
                  step >= 1 ? 'bg-ocean-600 text-white' : 'bg-gray-200 text-gray-600'
                }`}>
                  1
                </div>
                <div className={`flex-1 h-1 mx-2 ${
                  step >= 2 ? 'bg-ocean-600' : 'bg-gray-200'
                }`}></div>
                <div className={`flex items-center justify-center w-8 h-8 rounded-full ${
                  step >= 2 ? 'bg-ocean-600 text-white' : 'bg-gray-200 text-gray-600'
                }`}>
                  2
                </div>
                <div className={`flex-1 h-1 mx-2 ${
                  step >= 3 ? 'bg-ocean-600' : 'bg-gray-200'
                }`}></div>
                <div className={`flex items-center justify-center w-8 h-8 rounded-full ${
                  step >= 3 ? 'bg-ocean-600 text-white' : 'bg-gray-200 text-gray-600'
                }`}>
                  3
                </div>
              </div>
              <div className="flex justify-between text-xs mt-1">
                <span className={step >= 1 ? 'text-ocean-600' : 'text-gray-500'}>Datenquellen</span>
                <span className={step >= 2 ? 'text-ocean-600' : 'text-gray-500'}>Details</span>
                <span className={step >= 3 ? 'text-ocean-600' : 'text-gray-500'}>Überprüfen</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="px-4 py-5 sm:p-6">
          {/* Schritt 1: Datenquellen auswählen */}
          {step === 1 && (
            <div>
              <h4 className="text-lg font-medium text-gray-900 mb-4">Datenquellen auswählen</h4>
              
              {error && (
                <div className="mb-4 p-2 bg-red-50 text-red-700 text-sm rounded">
                  {error}
                </div>
              )}
              
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {dataSources.length > 0 ? (
                  dataSources.map(source => (
                    <div 
                      key={source.id} 
                      className={`p-3 border rounded-md cursor-pointer transition-colors ${
                        selectedSourceIds.includes(source.id)
                          ? 'border-ocean-500 bg-ocean-50'
                          : 'border-gray-300 hover:border-ocean-300'
                      }`}
                      onClick={() => handleSourceToggle(source.id)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          <input
                            type="checkbox"
                            checked={selectedSourceIds.includes(source.id)}
                            onChange={() => {}}
                            className="h-4 w-4 text-ocean-600 focus:ring-ocean-500 border-gray-300 rounded"
                          />
                          <div className="ml-3">
                            <h5 className="text-sm font-medium text-gray-900">{source.name}</h5>
                            <p className="text-xs text-gray-500">
                              {source.recordCount.toLocaleString()} Datensätze • {source.estimatedValue.toFixed(2)} OCEAN
                            </p>
                          </div>
                        </div>
                        <div className="text-xs">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full font-medium ${
                            source.privacyLevel === 'low' ? 'bg-yellow-100 text-yellow-800' :
                            source.privacyLevel === 'medium' ? 'bg-green-100 text-green-800' :
                            source.privacyLevel === 'high' ? 'bg-blue-100 text-blue-800' :
                            'bg-purple-100 text-purple-800'
                          }`}>
                            {source.privacyLevel === 'low' ? 'Niedrig' : 
                             source.privacyLevel === 'medium' ? 'Mittel' : 
                             source.privacyLevel === 'high' ? 'Hoch' : 'Nur Berechnung'}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-4 text-gray-500">
                    Keine Datenquellen verfügbar. Bitte fügen Sie zuerst Datenquellen hinzu.
                  </div>
                )}
              </div>
              
              {selectedSourceIds.length > 0 && (
                <div className="mt-4 p-3 bg-gray-50 rounded-md">
                  <div className="flex justify-between items-center">
                    <div>
                      <p className="text-sm font-medium text-gray-700">
                        {selectedSourceIds.length} Datenquelle(n) ausgewählt
                      </p>
                      <p className="text-xs text-gray-500">
                        Geschätzter Gesamtwert: {totalValue.toFixed(2)} OCEAN
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* Schritt 2: Details eingeben */}
          {step === 2 && (
            <div>
              <h4 className="text-lg font-medium text-gray-900 mb-4">Datensatz-Details</h4>
              
              {error && (
                <div className="mb-4 p-2 bg-red-50 text-red-700 text-sm rounded">
                  {error}
                </div>
              )}
              
              <div className="space-y-4">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                    Name *
                  </label>
                  <input
                    type="text"
                    id="name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-ocean-500 focus:border-ocean-500 sm:text-sm"
                    placeholder="z.B. Meine Gesundheitsdaten"
                    required
                  />
                </div>
                
                <div>
                  <label htmlFor="description" className="block text-sm font-medium text-gray-700">
                    Beschreibung *
                  </label>
                  <textarea
                    id="description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    rows={4}
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-ocean-500 focus:border-ocean-500 sm:text-sm"
                    placeholder="Beschreiben Sie Ihren Datensatz..."
                    required
                  />
                </div>
                
                <div>
                  <label htmlFor="price" className="block text-sm font-medium text-gray-700">
                    Preis (OCEAN)
                  </label>
                  <div className="mt-1 relative rounded-md shadow-sm">
                    <input
                      type="number"
                      id="price"
                      value={price}
                      onChange={(e) => setPrice(parseFloat(e.target.value))}
                      min="0.1"
                      step="0.1"
                      className="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-ocean-500 focus:border-ocean-500 sm:text-sm"
                      placeholder="1.0"
                    />
                    <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                      <span className="text-gray-500 sm:text-sm">OCEAN</span>
                    </div>
                  </div>
                  <p className="mt-1 text-xs text-gray-500">
                    Empfohlener Preis basierend auf Ihren Daten: {totalValue.toFixed(2)} OCEAN
                  </p>
                </div>
                
                <div>
                  <label htmlFor="privacy" className="block text-sm font-medium text-gray-700">
                    Datenschutzniveau für Tokenisierung
                  </label>
                  <select
                    id="privacy"
                    value={privacyLevel}
                    onChange={(e) => setPrivacyLevel(e.target.value)}
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-ocean-500 focus:border-ocean-500 sm:text-sm"
                  >
                    <option value="low">Niedrig (mehr Daten, höherer Wert)</option>
                    <option value="medium">Mittel (ausgewogenes Verhältnis)</option>
                    <option value="high">Hoch (weniger Daten, geringerer Wert)</option>
                    <option value="compute_only">Nur Berechnung (keine Datenfreigabe)</option>
                  </select>
                </div>
              </div>
            </div>
          )}
          
          {/* Schritt 3: Überprüfen und Tokenisieren */}
          {step === 3 && (
            <div>
              <h4 className="text-lg font-medium text-gray-900 mb-4">Überprüfen und Tokenisieren</h4>
              
              {error && (
                <div className="mb-4 p-2 bg-red-50 text-red-700 text-sm rounded">
                  {error}
                </div>
              )}
              
              <div className="bg-gray-50 p-4 rounded-md mb-4">
                <h5 className="text-md font-medium text-gray-900 mb-2">Zusammenfassung</h5>
                <dl className="grid grid-cols-1 gap-x-4 gap-y-4 sm:grid-cols-2">
                  <div className="sm:col-span-1">
                    <dt className="text-sm font-medium text-gray-500">Name</dt>
                    <dd className="mt-1 text-sm text-gray-900">{name}</dd>
                  </div>
                  <div className="sm:col-span-1">
                    <dt className="text-sm font-medium text-gray-500">Preis</dt>
                    <dd className="mt-1 text-sm text-gray-900">{price.toFixed(2)} OCEAN</dd>
                  </div>
                  <div className="sm:col-span-2">
                    <dt className="text-sm font-medium text-gray-500">Beschreibung</dt>
                    <dd className="mt-1 text-sm text-gray-900">{description}</dd>
                  </div>
                  <div className="sm:col-span-1">
                    <dt className="text-sm font-medium text-gray-500">Datenquellen</dt>
                    <dd className="mt-1 text-sm text-gray-900">
                      {selectedSourceIds.length} Quellen ausgewählt
                    </dd>
                  </div>
                  <div className="sm:col-span-1">
                    <dt className="text-sm font-medium text-gray-500">Datenschutzniveau</dt>
                    <dd className="mt-1 text-sm text-gray-900">
                      {privacyLevel === 'low' ? 'Niedrig' : 
                       privacyLevel === 'medium' ? 'Mittel' : 
                       privacyLevel === 'high' ? 'Hoch' : 'Nur Berechnung'}
                    </dd>
                  </div>
                </dl>
              </div>
              
              <div className="bg-yellow-50 p-4 rounded-md mb-4">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-yellow-800">Wichtige Informationen</h3>
                    <div className="mt-2 text-sm text-yellow-700">
                      <p>
                        Durch die Tokenisierung werden Ihre Daten auf der Blockchain veröffentlicht und können von anderen gekauft werden. 
                        Stellen Sie sicher, dass Sie die richtigen Datenschutzeinstellungen gewählt haben.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Schritt 4: Erfolg */}
          {step === 4 && (
            <div className="text-center py-6">
              <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-green-100">
                <CheckIcon className="h-6 w-6 text-green-600" />
              </div>
              <h3 className="mt-3 text-lg font-medium text-gray-900">Tokenisierung erfolgreich!</h3>
              <p className="mt-2 text-sm text-gray-500">
                Ihr Datensatz wurde erfolgreich tokenisiert und ist jetzt auf dem Ocean Protocol Marktplatz verfügbar.
              </p>
              <div className="mt-6">
                <button
                  type="button"
                  onClick={onClose}
                  className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
                >
                  Zum Marktplatz
                </button>
              </div>
            </div>
          )}
        </div>
        
        {/* Aktionsschaltflächen */}
        {step < 4 && (
          <div className="px-4 py-3 bg-gray-50 text-right sm:px-6 border-t border-gray-200">
            <div className="flex justify-between">
              {step > 1 ? (
                <button
                  type="button"
                  onClick={handleBack}
                  className="inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
                >
                  Zurück
                </button>
              ) : (
                <button
                  type="button"
                  onClick={onClose}
                  className="inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
                >
                  Abbrechen
                </button>
              )}
              
              {step < 3 ? (
                <button
                  type="button"
                  onClick={handleNext}
                  className="ml-3 inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
                >
                  Weiter
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={isSubmitting}
                  className="ml-3 inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500 disabled:opacity-50"
                >
                  {isSubmitting ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Tokenisiere...
                    </>
                  ) : (
                    'Tokenisieren'
                  )}
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Hauptkomponente für die Tokenisierungsseite
const Tokenization = () => {
  const { tokenizedDatasets, tokenizeDataset, isLoading } = useData();
  const [isWizardOpen, setIsWizardOpen] = useState(false);

  const handleTokenize = async (
    name: string, 
    description: string, 
    sourceIds: string[], 
    price: number,
    privacyLevel: string
  ) => {
    return await tokenizeDataset(name, description, sourceIds, price, privacyLevel);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Tokenisierung</h1>
        <button
          onClick={() => setIsWizardOpen(true)}
          className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
        >
          <PlusIcon className="w-5 h-5 mr-2" />
          Daten tokenisieren
        </button>
      </div>

      {/* Tokenisierte Datensätze */}
      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        {isLoading && (
          <div className="flex justify-center items-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-ocean-600"></div>
            <span className="ml-2 text-gray-600">Wird geladen...</span>
          </div>
        )}

        {!isLoading && tokenizedDatasets.length === 0 && (
          <div className="py-8 text-center">
            <TokenizationIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">Keine tokenisierten Datensätze</h3>
            <p className="mt-1 text-sm text-gray-500">
              Tokenisieren Sie Ihre Daten, um sie auf dem Ocean Protocol Marktplatz anzubieten.
            </p>
            <div className="mt-6">
              <button
                onClick={() => setIsWizardOpen(true)}
                className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
              >
                <PlusIcon className="w-5 h-5 mr-2" />
                Daten tokenisieren
              </button>
            </div>
          </div>
        )}

        {!isLoading && tokenizedDatasets.length > 0 && (
          <ul className="divide-y divide-gray-200">
            {tokenizedDatasets.map((dataset) => (
              <li key={dataset.id}>
                <div className="px-4 py-4 sm:px-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-medium text-gray-900">{dataset.name}</h3>
                      <p className="mt-1 text-sm text-gray-500 line-clamp-2">{dataset.description}</p>
                    </div>
                    <div className="ml-2 flex-shrink-0 flex">
                      <p className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                        {dataset.price.toFixed(2)} OCEAN
                      </p>
                    </div>
                  </div>
                  <div className="mt-2 sm:flex sm:justify-between">
                    <div className="sm:flex">
                      <p className="flex items-center text-sm text-gray-500">
                        Asset-ID: {dataset.assetId.substring(0, 10)}...
                      </p>
                      <p className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0 sm:ml-6">
                        Erstellt: {new Date(dataset.createdAt).toLocaleDateString('de-DE')}
                      </p>
                    </div>
                    <div className="mt-2 flex items-center text-sm text-gray-500 sm:mt-0">
                      <a 
                        href={dataset.marketplaceUrl} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-ocean-600 hover:text-ocean-800"
                      >
                        Auf Marktplatz anzeigen →
                      </a>
                    </div>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Tokenisierungs-Wizard */}
      <TokenizationWizard 
        isOpen={isWizardOpen} 
        onClose={() => setIsWizardOpen(false)} 
        onTokenize={handleTokenize} 
      />
    </div>
  );
};

export default Tokenization;