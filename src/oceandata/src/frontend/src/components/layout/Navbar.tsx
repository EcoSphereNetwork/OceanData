import { useState } from 'react';
import { useAuth } from '@context/AuthContext';
import { BellIcon, WalletIcon } from '@components/icons';

interface NavbarProps {
  toggleSidebar: () => void;
}

const Navbar = ({ toggleSidebar }: NavbarProps) => {
  const { user, logout, connectWallet } = useAuth();
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isNotificationsOpen, setIsNotificationsOpen] = useState(false);

  const handleConnectWallet = () => {
    // In einer echten Anwendung würde hier die Wallet-Verbindung erfolgen
    // Für die Demo verwenden wir eine simulierte Adresse
    const mockAddress = '0x' + Math.random().toString(36).substring(2, 15);
    connectWallet(mockAddress);
  };

  return (
    <header className="bg-white border-b border-gray-200 z-20">
      <div className="px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <button
              onClick={toggleSidebar}
              className="p-2 rounded-md text-gray-500 hover:bg-gray-100 focus:outline-none"
            >
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              </svg>
            </button>
            <h1 className="ml-4 text-xl font-semibold text-gray-800 hidden md:block">
              OceanData Platform
            </h1>
          </div>

          <div className="flex items-center space-x-4">
            {/* Wallet Connection */}
            {user?.walletAddress ? (
              <div className="flex items-center px-3 py-1 text-sm bg-green-50 text-green-700 rounded-full">
                <WalletIcon className="w-4 h-4 mr-1" />
                <span className="hidden md:inline">
                  {user.walletAddress.substring(0, 6)}...
                  {user.walletAddress.substring(user.walletAddress.length - 4)}
                </span>
              </div>
            ) : (
              <button
                onClick={handleConnectWallet}
                className="flex items-center px-3 py-1 text-sm bg-ocean-50 text-ocean-700 rounded-full hover:bg-ocean-100"
              >
                <WalletIcon className="w-4 h-4 mr-1" />
                <span className="hidden md:inline">Wallet verbinden</span>
              </button>
            )}

            {/* Notifications */}
            <div className="relative">
              <button
                onClick={() => setIsNotificationsOpen(!isNotificationsOpen)}
                className="p-1 rounded-full text-gray-500 hover:bg-gray-100 focus:outline-none"
              >
                <BellIcon className="w-6 h-6" />
                <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full"></span>
              </button>

              {isNotificationsOpen && (
                <div className="absolute right-0 mt-2 w-80 bg-white rounded-md shadow-lg py-1 z-50 border border-gray-200">
                  <div className="px-4 py-2 border-b border-gray-200">
                    <h3 className="text-sm font-medium text-gray-700">Benachrichtigungen</h3>
                  </div>
                  <div className="max-h-60 overflow-y-auto">
                    <div className="px-4 py-2 hover:bg-gray-50">
                      <p className="text-sm text-gray-700">Neue Marktplatz-Anfrage für Ihre Daten</p>
                      <p className="text-xs text-gray-500">Vor 5 Minuten</p>
                    </div>
                    <div className="px-4 py-2 hover:bg-gray-50">
                      <p className="text-sm text-gray-700">Datenquelle erfolgreich synchronisiert</p>
                      <p className="text-xs text-gray-500">Vor 1 Stunde</p>
                    </div>
                    <div className="px-4 py-2 hover:bg-gray-50">
                      <p className="text-sm text-gray-700">Tokenisierung abgeschlossen</p>
                      <p className="text-xs text-gray-500">Vor 1 Tag</p>
                    </div>
                  </div>
                  <div className="px-4 py-2 border-t border-gray-200">
                    <a href="#" className="text-xs text-ocean-600 hover:text-ocean-800">
                      Alle Benachrichtigungen anzeigen
                    </a>
                  </div>
                </div>
              )}
            </div>

            {/* Profile Dropdown */}
            <div className="relative">
              <button
                onClick={() => setIsProfileOpen(!isProfileOpen)}
                className="flex items-center space-x-2 focus:outline-none"
              >
                <div className="w-8 h-8 rounded-full bg-ocean-500 flex items-center justify-center text-white">
                  {user?.username.charAt(0).toUpperCase()}
                </div>
                <span className="hidden md:block text-sm font-medium text-gray-700">
                  {user?.username}
                </span>
                <svg
                  className="w-4 h-4 text-gray-500"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </button>

              {isProfileOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-50 border border-gray-200">
                  <a
                    href="#"
                    className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    Profil
                  </a>
                  <a
                    href="#"
                    className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    Einstellungen
                  </a>
                  <button
                    onClick={logout}
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    Abmelden
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Navbar;