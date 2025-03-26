import { Link } from 'react-router-dom';
import { SmolituxIcon } from '@components/icons';

const NotFound = () => {
  return (
    <div className="min-h-full flex flex-col items-center justify-center py-12">
      <div className="flex-shrink-0 flex justify-center">
        <Link to="/" className="inline-flex">
          <span className="sr-only">OceanData</span>
          <SmolituxIcon className="h-12 w-auto text-ocean-600" />
        </Link>
      </div>
      <div className="mt-6 max-w-md mx-auto text-center">
        <h1 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">404</h1>
        <p className="mt-2 text-lg text-gray-500">
          Die angeforderte Seite konnte nicht gefunden werden.
        </p>
        <div className="mt-6">
          <Link
            to="/"
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-ocean-600 hover:bg-ocean-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ocean-500"
          >
            Zurück zur Startseite
          </Link>
        </div>
      </div>
    </div>
  );
};

export default NotFound;