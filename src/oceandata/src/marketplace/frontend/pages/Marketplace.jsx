// OceanData Marketplace Frontend
// Hauptkomponente für die Datenmarktplatz-Benutzeroberfläche

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import axios from 'axios';
import './App.css';

// Mock-API-URL für das MVP
const API_BASE_URL = 'http://localhost:5000/api';

// Header-Komponente mit Navigation
const Header = () => (
  <header className="app-header">
    <div className="logo-container">
      <img src="/logo.png" alt="OceanData Logo" className="logo" />
      <h1>OceanData</h1>
    </div>
    <nav>
      <ul>
        <li><Link to="/">Home</Link></li>
        <li><Link to="/marketplace">Marketplace</Link></li>
        <li><Link to="/my-data">My Data</Link></li>
        <li><Link to="/wallet">Wallet</Link></li>
      </ul>
    </nav>
  </header>
);

// Footer-Komponente
const Footer = () => (
  <footer>
    <p>&copy; 2025 OceanData. All rights reserved.</p>
    <div className="footer-links">
      <a href="/privacy">Privacy Policy</a>
      <a href="/terms">Terms of Service</a>
      <a href="https://github.com/your-org/oceandata">GitHub</a>
      <a href="https://discord.gg/oceandata-community">Discord</a>
    </div>
  </footer>
);

// Home-Seite (Landing-Page)
const Home = () => (
  <div className="home-container">
    <section className="hero">
      <h1>Monetize Your Data Securely</h1>
      <p>OceanData is a decentralized platform that empowers you to control and monetize your data while ensuring privacy and security.</p>
      <div className="cta-buttons">
        <Link to="/marketplace" className="cta-button primary">Explore Marketplace</Link>
        <Link to="/my-data" className="cta-button secondary">Monetize Your Data</Link>
      </div>
    </section>
    
    <section className="features">
      <h2>Key Features</h2>
      <div className="feature-grid">
        <div className="feature-card">
          <div className="feature-icon">🔒</div>
          <h3>Privacy-Preserving</h3>
          <p>Your data never leaves your control. Our Compute-to-Data functionality protects your privacy.</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">💰</div>
          <h3>Fair Monetization</h3>
          <p>Receive equitable compensation for the data you choose to share.</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🌐</div>
          <h3>Decentralized</h3>
          <p>Built on blockchain technology for transparency and security.</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🤝</div>
          <h3>Community-Driven</h3>
          <p>Join a community of data providers and consumers in a fair ecosystem.</p>
        </div>
      </div>
    </section>
    
    <section className="how-it-works">
      <h2>How It Works</h2>
      <div className="steps">
        <div className="step">
          <div className="step-number">1</div>
          <h3>Connect Your Data Sources</h3>
          <p>Easily connect various data sources like browser history, health data, or IoT devices.</p>
        </div>
        <div className="step">
          <div className="step-number">2</div>
          <h3>Control Your Data</h3>
          <p>Choose what data to share and set your own terms for access.</p>
        </div>
        <div className="step">
          <div className="step-number">3</div>
          <h3>Monetize</h3>
          <p>Your data is tokenized and made available in the marketplace for interested buyers.</p>
        </div>
        <div className="step">
          <div className="step-number">4</div>
          <h3>Earn Rewards</h3>
          <p>Receive compensation when your data is used, maintaining full control throughout.</p>
        </div>
      </div>
    </section>
  </div>
);

// Komponente für den Datenmarktplatz
const Marketplace = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  
  // Simuliere das Laden von Datasets vom Backend
  useEffect(() => {
    // In einer tatsächlichen Implementierung würde dies API-Aufrufe verwenden
    const fetchDatasets = async () => {
      try {
        // Für das MVP simulieren wir die Daten
        // setLoading(true);
        // const response = await axios.get(`${API_BASE_URL}/datasets`);
        // setDatasets(response.data);
        
        // Simulierte Daten für das MVP
        setTimeout(() => {
          const mockDatasets = [
            {
              id: '1',
              name: 'Browser Navigation Patterns',
              description: 'Anonymous browser navigation data with usage patterns and preferences',
              price: 0.05,
              owner: '0x1a2b...3c4d',
              category: 'browser',
              records: 5000,
              rating: 4.2
            },
            {
              id: '2',
              name: 'Fitness Activity Dataset',
              description: 'Comprehensive fitness activity data from smart watches and trackers',
              price: 0.08,
              owner: '0x5e6f...7g8h',
              category: 'health',
              records: 8500,
              rating: 4.7
            },
            {
              id: '3',
              name: 'Smart Home Patterns',
              description: 'Usage patterns from smart home devices and energy consumption data',
              price: 0.03,
              owner: '0x9i0j...1k2l',
              category: 'iot',
              records: 3200,
              rating: 3.9
            },
            {
              id: '4',
              name: 'Social Media Engagement',
              description: 'Anonymized social media interaction and engagement metrics',
              price: 0.06,
              owner: '0x3m4n...5o6p',
              category: 'social',
              records: 12000,
              rating: 4.4
            }
          ];
          setDatasets(mockDatasets);
          setLoading(false);
        }, 1000); // Simulate network delay
      } catch (error) {
        console.error('Error fetching datasets:', error);
        setLoading(false);
      }
    };
    
    fetchDatasets();
  }, []);
  
  // Filter datasets based on category
  const filteredDatasets = filter === 'all' 
    ? datasets 
    : datasets.filter(dataset => dataset.category === filter);
  
  return (
    <div className="marketplace-container">
      <h1>Data Marketplace</h1>
      <p>Discover and purchase high-quality datasets from our community of data providers.</p>
      
      <div className="marketplace-filters">
        <button 
          className={filter === 'all' ? 'active' : ''}
          onClick={() => setFilter('all')}>
          All Categories
        </button>
        <button 
          className={filter === 'browser' ? 'active' : ''}
          onClick={() => setFilter('browser')}>
          Browser Data
        </button>
        <button 
          className={filter === 'health' ? 'active' : ''}
          onClick={() => setFilter('health')}>
          Health Data
        </button>
        <button 
          className={filter === 'iot' ? 'active' : ''}
          onClick={() => setFilter('iot')}>
          IoT Data
        </button>
        <button 
          className={filter === 'social' ? 'active' : ''}
          onClick={() => setFilter('social')}>
          Social Media
        </button>
      </div>
      
      {loading ? (
        <div className="loading">Loading datasets...</div>
      ) : (
        <div className="datasets-grid">
          {filteredDatasets.map(dataset => (
            <div key={dataset.id} className="dataset-card">
              <h3>{dataset.name}</h3>
              <p className="dataset-description">{dataset.description}</p>
              <div className="dataset-meta">
                <span className="dataset-category">{dataset.category}</span>
                <span className="dataset-records">{dataset.records.toLocaleString()} records</span>
                <span className="dataset-rating">★ {dataset.rating.toFixed(1)}</span>
              </div>
              <div className="dataset-price">
                <span className="price-value">{dataset.price} OCEAN</span>
                <span className="price-owner">by {dataset.owner}</span>
              </div>
              <div className="dataset-actions">
                <button className="action-button preview">Preview</button>
                <button className="action-button purchase">Purchase</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Komponente für eigene Daten
const MyData = () => {
  const [connectedSources, setConnectedSources] = useState([]);
  const [availableSources, setAvailableSources] = useState([]);
  const [publishedDatasets, setPublishedDatasets] = useState([]);
  
  useEffect(() => {
    // Simuliere verbundene und verfügbare Datenquellen
    setConnectedSources([
      { id: 'browser1', name: 'Chrome Browser', type: 'browser', lastSync: '2025-03-14T10:30:00Z', status: 'active' },
      { id: 'iot1', name: 'Fitbit Smartwatch', type: 'iot', lastSync: '2025-03-13T23:15:00Z', status: 'active' }
    ]);
    
    setAvailableSources([
      { id: 'browser2', name: 'Firefox Browser', type: 'browser' },
      { id: 'health1', name: 'Health Insurance Data', type: 'health' },
      { id: 'iot2', name: 'Smart Home Hub', type: 'iot' },
      { id: 'social1', name: 'Social Media Account', type: 'social' }
    ]);
    
    // Simuliere veröffentlichte Datensätze
    setPublishedDatasets([
      { 
        id: 'ds1', 
        name: 'My Browser Patterns', 
        created: '2025-03-10T14:22:00Z',
        status: 'active',
        purchases: 12,
        revenue: 0.72,
        access: 'compute-to-data'
      }
    ]);
  }, []);
  
  return (
    <div className="my-data-container">
      <h1>My Data Assets</h1>
      <p>Connect, manage and monetize your personal data sources.</p>
      
      <section className="data-sources">
        <h2>Connected Data Sources</h2>
        {connectedSources.length === 0 ? (
          <p className="empty-state">No data sources connected yet. Add one below to get started.</p>
        ) : (
          <div className="sources-grid">
            {connectedSources.map(source => (
              <div key={source.id} className="source-card">
                <div className={`source-icon ${source.type}`}></div>
                <h3>{source.name}</h3>
                <div className="source-meta">
                  <span className="source-status">Status: <span className={`status-${source.status}`}>{source.status}</span></span>
                  <span className="source-sync">Last sync: {new Date(source.lastSync).toLocaleString()}</span>
                </div>
                <div className="source-actions">
                  <button className="action-button sync">Sync Now</button>
                  <button className="action-button disconnect">Disconnect</button>
                </div>
              </div>
            ))}
          </div>
        )}
        
        <h2>Available Data Sources</h2>
        <div className="sources-grid">
          {availableSources.map(source => (
            <div key={source.id} className="source-card available">
              <div className={`source-icon ${source.type}`}></div>
              <h3>{source.name}</h3>
              <p className="source-description">Connect to add this data source to your profile.</p>
              <div className="source-actions">
                <button className="action-button connect">Connect</button>
              </div>
            </div>
          ))}
        </div>
      </section>
      
      <section className="published-datasets">
        <h2>Published Datasets</h2>
        {publishedDatasets.length === 0 ? (
          <p className="empty-state">You haven't published any datasets yet.</p>
        ) : (
          <div className="datasets-table">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Created</th>
                  <th>Status</th>
                  <th>Purchases</th>
                  <th>Revenue</th>
                  <th>Access Type</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {publishedDatasets.map(dataset => (
                  <tr key={dataset.id}>
                    <td>{dataset.name}</td>
                    <td>{new Date(dataset.created).toLocaleDateString()}</td>
                    <td><span className={`status-${dataset.status}`}>{dataset.status}</span></td>
                    <td>{dataset.purchases}</td>
                    <td>{dataset.revenue} OCEAN</td>
                    <td>{dataset.access}</td>
                    <td>
                      <button className="action-button small edit">Edit</button>
                      <button className="action-button small unpublish">Unpublish</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        
        <div className="publish-actions">
          <button className="action-button primary">Create New Dataset</button>
        </div>
      </section>
    </div>
  );
};

// Wallet-Komponente
const Wallet = () => {
  const [balance, setBalance] = useState(1.45);
  const [transactions, setTransactions] = useState([]);
  
  useEffect(() => {
    // Simuliere Transaktionsdaten
    setTransactions([
      { id: 't1', type: 'receive', amount: 0.12, from: '0x1a2b...3c4d', date: '2025-03-14T09:15:00Z', status: 'confirmed' },
      { id: 't2', type: 'receive', amount: 0.08, from: '0x5e6f...7g8h', date: '2025-03-13T16:30:00Z', status: 'confirmed' },
      { id: 't3', type: 'send', amount: 0.05, to: '0x9i0j...1k2l', date: '2025-03-12T11:45:00Z', status: 'confirmed' },
    ]);
  }, []);
  
  return (
    <div className="wallet-container">
      <h1>My Wallet</h1>
      <p>Manage your OCEAN tokens and transactions.</p>
      
      <div className="wallet-overview">
        <div className="balance-card">
          <h2>OCEAN Balance</h2>
          <div className="balance-amount">{balance.toFixed(2)} OCEAN</div>
          <div className="balance-actions">
            <button className="action-button deposit">Deposit</button>
            <button className="action-button withdraw">Withdraw</button>
          </div>
        </div>
      </div>
      
      <section className="transaction-history">
        <h2>Transaction History</h2>
        <div className="transactions-table">
          <table>
            <thead>
              <tr>
                <th>Type</th>
                <th>Amount</th>
                <th>From/To</th>
                <th>Date</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {transactions.map(tx => (
                <tr key={tx.id}>
                  <td className={`tx-type ${tx.type}`}>{tx.type === 'receive' ? 'Received' : 'Sent'}</td>
                  <td className="tx-amount">{tx.amount.toFixed(2)} OCEAN</td>
                  <td className="tx-address">{tx.type === 'receive' ? `From: ${tx.from}` : `To: ${tx.to}`}</td>
                  <td>{new Date(tx.date).toLocaleString()}</td>
                  <td><span className={`status-${tx.status}`}>{tx.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};

// Hauptanwendung
function App() {
  return (
    <Router>
      <div className="app">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/marketplace" element={<Marketplace />} />
            <Route path="/my-data" element={<MyData />} />
            <Route path="/wallet" element={<Wallet />} />
