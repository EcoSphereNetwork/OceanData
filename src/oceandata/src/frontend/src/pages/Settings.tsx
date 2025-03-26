import { useState } from 'react';
import { useAuth } from '@context/AuthContext';
import { SettingsIcon, WalletIcon } from '@components/icons';

const Settings = () => {
  const { user, connectWallet } = useAuth();
  const [activeTab, setActiveTab] = useState('profile');
  const [username, setUsername] = useState(user?.username || '');
  const [email, setEmail] = useState(user?.email || '');
  const [notificationSettings, setNotificationSettings] = useState({
    emailNotifications: true,
    marketplaceUpdates: true,
    dataSourceAlerts: true,
    securityAlerts: true
  });
  const [privacySettings, setPrivacySettings] = useState({
    shareAnalytics: true,
    allowTracking: false,
    enhancedPrivacy: true
  });
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const handleConnectWallet = () => {
    // In einer echten Anwendung würde hier die Wallet-Verbindung erfolgen
    // Für die Demo verwenden wir eine simulierte Adresse
    const mockAddress = '0x' + Math.random().toString(36).substring(2, 15);
    connectWallet(mockAddress);
  };

  const handleSaveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSaving(true);
    setSaveSuccess(false);
    
    // Simuliere API-Anfrage
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    setIsSaving(false);
    setSaveSuccess(true);
    
    // Zurücksetzen des Erfolgsstatus nach 3 Sekunden
    setTimeout(() => {
      setSaveSuccess(false);
    }, 3000);
  };

  const handleNotificationChange = (setting: keyof typeof notificationSettings) => {
    setNotificationSettings(prev => ({
      ...prev,
      [setting]: !prev[setting]
    }));
  };

  const handlePrivacyChange = (setting: keyof typeof privacySettings) => {
    setPrivacySettings(prev => ({
      ...prev,
      [setting]: !prev[setting]
    }));
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Einstellungen</h1>
      </div>

      <div className="bg-white shadow rounded-lg overflow-hidden">
        <div className="sm:flex sm:items-center border-b border-gray-200">
          <div className="px-4 py-3 sm:px-6 flex space-x-4">
            <button
              onClick={() => setActiveTab('profile')}
              className={`px-3 py-2 text-sm font-medium rounded-md ${
                activeTab === 'profile'
                  ? 'bg-ocean-100 text-ocean-700'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Profil
            </button>
            <button
              onClick={() => setActiveTab('wallet')}
              className={`px-3 py-2 text-sm font-medium rounded-md ${
                activeTab === 'wallet'
                  ? 'bg-ocean-100 text-ocean-700'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Wallet
            </button>
            <button
              onClick={() => setActiveTab('notifications')}
              className={`px-3 py-2 text-sm font-medium rounded-md ${
                activeTab === 'notifications'
                  ? 'bg-ocean-100 text-ocean-700'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Benachrichtigungen
            </button>
            <button
              onClick={() => setActiveTab('privacy')}
              className={`px-3 py-2 text-sm font-medium rounded-md ${
                activeTab === 'privacy'
                  ? 'bg-ocean-100 text-ocean-700'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Datenschutz
            </button>
          </div>
        </div>

        <div className="px-4 py-5 sm:p-6">
          {/* Profil-Tab */}
          {activeTab === 'profile' && (
            <form onSubmit={handleSaveProfile}>
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium leading-6 text-gray-900">Profilinformationen</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Aktualisieren Sie Ihre persönlichen Informationen.
                  </p>
                </div>

                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div className="relative">
                      <div className="w-16 h-16 rounded-full bg-ocean-500 flex items-center justify-center text-white text-xl">
                        {username.charAt(0).toUpperCase()}
                      </div>
                      <button
                        type="button"
                        className="absolute bottom-0 right-0 bg-white rounded-full p-1 border border-gray-300 shadow-sm"
                      >
                        <svg className="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  <div className="ml-5">
                    <h4 className="text-sm font-medium text-gray-900">{username}</h4>
                    <p className="text-sm text-gray-500">{email}</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-y-6 gap-x-4 sm:grid-cols-6">
                  <div className="sm:col-span-3">
                    <label htmlFor="username" className="block text-sm font-medium text-gray-700">
                      Benutzername
                    </label>
                    <div className="mt-1">
                      <input
                        type="text"
                        name="username"
                        id="username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        className="shadow-sm focus:ring-ocean-500 focus:border-ocean-500 block w-full sm:text-sm border-gray-300 rounded-md"
                      />
                    </div>
                  </div>

                  <div className="sm:col-span-3">
                    <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                      E-Mail-Adresse
                    </label>
                    <div className="mt-1">
                      <input
                        type="email"
                        name="email"
                        id="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        className="shadow-sm focus:ring-ocean-500 focus:border-ocean-500 block w-full sm:text-sm border-gray-300 rounded-md"
                      />
                    </div>
                  </div>

                  <div className="sm:col-span-6">
                    <label htmlFor="about" className="block text-sm font-medium text-gray-700">
                      Über mich
                    </label>
                    <div className="mt-1">
                      <textarea
                        id="about"
                        name="about"
                        rows={3}
                        className="shadow-sm focus:ring-ocean-500 focus:border-ocean-500 block w-full sm:text-sm border-gray-300 rounded-md"
                        placeholder="Erzählen Sie etwas über sich..."
                      />
                    </div>
                    <p className="mt-2 text-sm text-gray-500">
                      Kurze Beschreibung für Ihr Profil.
                    </p>
                  </div>
                </div>

                <div className="pt-5 border-t border-gray-200">
                  <div className="flex justify-end">
                    {saveSuccess && (
                      <span className="mr-3 inline-flex items-center px-3 py-2 text-sm leading-4 font-medium rounded-md text-green-700 bg-green-100">
                        Änderungen gespeichert!
                      </span>
                    )}
                    <button
                      type="button"
                      className="bg-white py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
                    >
                      Abbrechen
                    </button>
                    <button
                      type="submit"
                      disabled={isSaving}
                      className="ml-3 inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500 disabled:opacity-50"
                    >
                      {isSaving ? 'Wird gespeichert...' : 'Speichern'}
                    </button>
                  </div>
                </div>
              </div>
            </form>
          )}

          {/* Wallet-Tab */}
          {activeTab === 'wallet' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium leading-6 text-gray-900">Wallet-Verbindung</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Verbinden Sie Ihre Krypto-Wallet, um Datensätze zu kaufen und zu verkaufen.
                </p>
              </div>

              <div className="bg-gray-50 p-4 rounded-md">
                {user?.walletAddress ? (
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center">
                      <div className="p-2 rounded-full bg-green-100 text-green-600">
                        <WalletIcon className="w-6 h-6" />
                      </div>
                      <div className="ml-3">
                        <h4 className="text-sm font-medium text-gray-900">Wallet verbunden</h4>
                        <p className="text-sm text-gray-500">
                          {user.walletAddress.substring(0, 8)}...
                          {user.walletAddress.substring(user.walletAddress.length - 6)}
                        </p>
                      </div>
                    </div>
                    <div className="mt-3 sm:mt-0">
                      <button
                        type="button"
                        className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
                      >
                        Trennen
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center">
                      <div className="p-2 rounded-full bg-gray-100 text-gray-600">
                        <WalletIcon className="w-6 h-6" />
                      </div>
                      <div className="ml-3">
                        <h4 className="text-sm font-medium text-gray-900">Keine Wallet verbunden</h4>
                        <p className="text-sm text-gray-500">
                          Verbinden Sie Ihre Wallet, um am Marktplatz teilzunehmen.
                        </p>
                      </div>
                    </div>
                    <div className="mt-3 sm:mt-0">
                      <button
                        type="button"
                        onClick={handleConnectWallet}
                        className="inline-flex items-center px-3 py-2 border border-transparent shadow-sm text-sm leading-4 font-medium rounded-md text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
                      >
                        <WalletIcon className="mr-2 h-4 w-4" />
                        Wallet verbinden
                      </button>
                    </div>
                  </div>
                )}
              </div>

              <div className="border-t border-gray-200 pt-6">
                <h4 className="text-md font-medium text-gray-900 mb-4">Unterstützte Wallets</h4>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="border border-gray-200 rounded-md p-4 flex flex-col items-center">
                    <img src="https://metamask.io/images/metamask-fox.svg" alt="MetaMask" className="h-12 w-12" />
                    <span className="mt-2 text-sm font-medium text-gray-900">MetaMask</span>
                  </div>
                  <div className="border border-gray-200 rounded-md p-4 flex flex-col items-center">
                    <img src="https://walletconnect.com/walletconnect-logo.svg" alt="WalletConnect" className="h-12 w-12" />
                    <span className="mt-2 text-sm font-medium text-gray-900">WalletConnect</span>
                  </div>
                  <div className="border border-gray-200 rounded-md p-4 flex flex-col items-center">
                    <img src="https://trustwallet.com/assets/images/media/assets/trust_platform.svg" alt="Trust Wallet" className="h-12 w-12" />
                    <span className="mt-2 text-sm font-medium text-gray-900">Trust Wallet</span>
                  </div>
                  <div className="border border-gray-200 rounded-md p-4 flex flex-col items-center">
                    <img src="https://www.coinbase.com/img/favicon/favicon-32.png" alt="Coinbase Wallet" className="h-12 w-12" />
                    <span className="mt-2 text-sm font-medium text-gray-900">Coinbase Wallet</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Benachrichtigungen-Tab */}
          {activeTab === 'notifications' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium leading-6 text-gray-900">Benachrichtigungseinstellungen</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Legen Sie fest, welche Benachrichtigungen Sie erhalten möchten.
                </p>
              </div>

              <div className="space-y-4">
                <div className="relative flex items-start">
                  <div className="flex items-center h-5">
                    <input
                      id="email-notifications"
                      name="email-notifications"
                      type="checkbox"
                      checked={notificationSettings.emailNotifications}
                      onChange={() => handleNotificationChange('emailNotifications')}
                      className="focus:ring-ocean-500 h-4 w-4 text-ocean-600 border-gray-300 rounded"
                    />
                  </div>
                  <div className="ml-3 text-sm">
                    <label htmlFor="email-notifications" className="font-medium text-gray-700">
                      E-Mail-Benachrichtigungen
                    </label>
                    <p className="text-gray-500">Erhalten Sie wichtige Updates per E-Mail.</p>
                  </div>
                </div>

                <div className="relative flex items-start">
                  <div className="flex items-center h-5">
                    <input
                      id="marketplace-updates"
                      name="marketplace-updates"
                      type="checkbox"
                      checked={notificationSettings.marketplaceUpdates}
                      onChange={() => handleNotificationChange('marketplaceUpdates')}
                      className="focus:ring-ocean-500 h-4 w-4 text-ocean-600 border-gray-300 rounded"
                    />
                  </div>
                  <div className="ml-3 text-sm">
                    <label htmlFor="marketplace-updates" className="font-medium text-gray-700">
                      Marktplatz-Updates
                    </label>
                    <p className="text-gray-500">Benachrichtigungen über Käufe, Verkäufe und Preisänderungen.</p>
                  </div>
                </div>

                <div className="relative flex items-start">
                  <div className="flex items-center h-5">
                    <input
                      id="data-source-alerts"
                      name="data-source-alerts"
                      type="checkbox"
                      checked={notificationSettings.dataSourceAlerts}
                      onChange={() => handleNotificationChange('dataSourceAlerts')}
                      className="focus:ring-ocean-500 h-4 w-4 text-ocean-600 border-gray-300 rounded"
                    />
                  </div>
                  <div className="ml-3 text-sm">
                    <label htmlFor="data-source-alerts" className="font-medium text-gray-700">
                      Datenquellen-Benachrichtigungen
                    </label>
                    <p className="text-gray-500">Benachrichtigungen über Verbindungsprobleme oder Synchronisierungsfehler.</p>
                  </div>
                </div>

                <div className="relative flex items-start">
                  <div className="flex items-center h-5">
                    <input
                      id="security-alerts"
                      name="security-alerts"
                      type="checkbox"
                      checked={notificationSettings.securityAlerts}
                      onChange={() => handleNotificationChange('securityAlerts')}
                      className="focus:ring-ocean-500 h-4 w-4 text-ocean-600 border-gray-300 rounded"
                    />
                  </div>
                  <div className="ml-3 text-sm">
                    <label htmlFor="security-alerts" className="font-medium text-gray-700">
                      Sicherheitsbenachrichtigungen
                    </label>
                    <p className="text-gray-500">Wichtige Sicherheitshinweise und Warnungen.</p>
                  </div>
                </div>
              </div>

              <div className="pt-5 border-t border-gray-200">
                <div className="flex justify-end">
                  <button
                    type="button"
                    className="bg-white py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
                  >
                    Zurücksetzen
                  </button>
                  <button
                    type="button"
                    className="ml-3 inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
                  >
                    Speichern
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Datenschutz-Tab */}
          {activeTab === 'privacy' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium leading-6 text-gray-900">Datenschutzeinstellungen</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Verwalten Sie Ihre Datenschutzeinstellungen und -präferenzen.
                </p>
              </div>

              <div className="space-y-4">
                <div className="relative flex items-start">
                  <div className="flex items-center h-5">
                    <input
                      id="share-analytics"
                      name="share-analytics"
                      type="checkbox"
                      checked={privacySettings.shareAnalytics}
                      onChange={() => handlePrivacyChange('shareAnalytics')}
                      className="focus:ring-ocean-500 h-4 w-4 text-ocean-600 border-gray-300 rounded"
                    />
                  </div>
                  <div className="ml-3 text-sm">
                    <label htmlFor="share-analytics" className="font-medium text-gray-700">
                      Nutzungsstatistiken teilen
                    </label>
                    <p className="text-gray-500">
                      Helfen Sie uns, OceanData zu verbessern, indem Sie anonymisierte Nutzungsdaten teilen.
                    </p>
                  </div>
                </div>

                <div className="relative flex items-start">
                  <div className="flex items-center h-5">
                    <input
                      id="allow-tracking"
                      name="allow-tracking"
                      type="checkbox"
                      checked={privacySettings.allowTracking}
                      onChange={() => handlePrivacyChange('allowTracking')}
                      className="focus:ring-ocean-500 h-4 w-4 text-ocean-600 border-gray-300 rounded"
                    />
                  </div>
                  <div className="ml-3 text-sm">
                    <label htmlFor="allow-tracking" className="font-medium text-gray-700">
                      Tracking erlauben
                    </label>
                    <p className="text-gray-500">
                      Erlauben Sie uns, Ihre Aktivitäten auf der Plattform zu verfolgen, um personalisierte Empfehlungen zu geben.
                    </p>
                  </div>
                </div>

                <div className="relative flex items-start">
                  <div className="flex items-center h-5">
                    <input
                      id="enhanced-privacy"
                      name="enhanced-privacy"
                      type="checkbox"
                      checked={privacySettings.enhancedPrivacy}
                      onChange={() => handlePrivacyChange('enhancedPrivacy')}
                      className="focus:ring-ocean-500 h-4 w-4 text-ocean-600 border-gray-300 rounded"
                    />
                  </div>
                  <div className="ml-3 text-sm">
                    <label htmlFor="enhanced-privacy" className="font-medium text-gray-700">
                      Erweiterter Datenschutz
                    </label>
                    <p className="text-gray-500">
                      Aktivieren Sie zusätzliche Datenschutzmaßnahmen für Ihre Daten.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-yellow-50 p-4 rounded-md">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-yellow-800">Datenschutzhinweis</h3>
                    <div className="mt-2 text-sm text-yellow-700">
                      <p>
                        Ihre Datenschutzeinstellungen beeinflussen, wie Ihre Daten auf der Plattform verarbeitet werden.
                        Bitte lesen Sie unsere <a href="#" className="font-medium underline">Datenschutzrichtlinie</a> für weitere Informationen.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="pt-5 border-t border-gray-200">
                <div className="flex justify-end">
                  <button
                    type="button"
                    className="bg-white py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
                  >
                    Zurücksetzen
                  </button>
                  <button
                    type="button"
                    className="ml-3 inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
                  >
                    Speichern
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Settings;