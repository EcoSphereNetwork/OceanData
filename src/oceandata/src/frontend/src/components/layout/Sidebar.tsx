import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '@context/AuthContext';
import { 
  SmolituxIcon, 
  DashboardIcon, 
  DataSourceIcon, 
  AnalyticsIcon, 
  TokenizationIcon, 
  MarketplaceIcon, 
  SettingsIcon 
} from '@components/icons';

interface SidebarProps {
  isOpen: boolean;
  toggleSidebar: () => void;
}

const Sidebar = ({ isOpen, toggleSidebar }: SidebarProps) => {
  const location = useLocation();
  const { user } = useAuth();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: <DashboardIcon className="w-6 h-6" /> },
    { path: '/data-sources', label: 'Datenquellen', icon: <DataSourceIcon className="w-6 h-6" /> },
    { path: '/analytics', label: 'Analysen', icon: <AnalyticsIcon className="w-6 h-6" /> },
    { path: '/tokenization', label: 'Tokenisierung', icon: <TokenizationIcon className="w-6 h-6" /> },
    { path: '/marketplace', label: 'Marktplatz', icon: <MarketplaceIcon className="w-6 h-6" /> },
    { path: '/settings', label: 'Einstellungen', icon: <SettingsIcon className="w-6 h-6" /> },
  ];

  return (
    <aside 
      className={`bg-white border-r border-gray-200 transition-all duration-300 ${
        isOpen ? 'w-64' : 'w-20'
      } fixed inset-y-0 left-0 z-30 md:relative`}
    >
      <div className="flex flex-col h-full">
        {/* Logo */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200">
          <Link to="/" className="flex items-center">
            <SmolituxIcon className="w-8 h-8 text-ocean-600" />
            {isOpen && (
              <span className="ml-2 text-xl font-semibold text-gray-900">OceanData</span>
            )}
          </Link>
          <button 
            onClick={toggleSidebar}
            className="p-1 rounded-md text-gray-500 hover:bg-gray-100 md:hidden"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center px-2 py-2 rounded-md transition-colors ${
                  isActive
                    ? 'bg-ocean-50 text-ocean-700'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <div className="text-gray-500">{item.icon}</div>
                {isOpen && <span className="ml-3">{item.label}</span>}
              </Link>
            );
          })}
        </nav>

        {/* User Info */}
        {isOpen && user && (
          <div className="p-4 border-t border-gray-200">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 rounded-full bg-ocean-500 flex items-center justify-center text-white">
                  {user.username.charAt(0).toUpperCase()}
                </div>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-700">{user.username}</p>
                <p className="text-xs text-gray-500 truncate">{user.email}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};

export default Sidebar;