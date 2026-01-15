import { useState } from 'react'
import { MarketScanner } from './components/scanner'
import { SignalsPanel } from './components/signals'
import { PositionsPanel } from './components/positions'
import { PerformanceDashboard } from './components/performance'
import './index.css'
import './App.css'

type Tab = 'scanner' | 'signals' | 'positions' | 'performance'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('scanner')

  return (
    <div className="app">
      <nav className="app-nav">
        <div className="nav-brand">
          <span className="brand-icon">◉</span>
          <span className="brand-text">Polymarket Bot</span>
        </div>
        <div className="nav-tabs">
          <button
            className={activeTab === 'scanner' ? 'active' : ''}
            onClick={() => setActiveTab('scanner')}
          >
            Scanner
          </button>
          <button
            className={activeTab === 'signals' ? 'active' : ''}
            onClick={() => setActiveTab('signals')}
          >
            Signals
          </button>
          <button
            className={activeTab === 'positions' ? 'active' : ''}
            onClick={() => setActiveTab('positions')}
          >
            Positions
          </button>
          <button
            className={activeTab === 'performance' ? 'active' : ''}
            onClick={() => setActiveTab('performance')}
          >
            Performance
          </button>
        </div>
      </nav>

      <main className="app-content">
        {activeTab === 'scanner' && <MarketScanner />}
        {activeTab === 'signals' && (
          <div className="panel-container">
            <SignalsPanel />
          </div>
        )}
        {activeTab === 'positions' && (
          <div className="panel-container">
            <PositionsPanel />
          </div>
        )}
        {activeTab === 'performance' && <PerformanceDashboard />}
      </main>
    </div>
  )
}

export default App
