import { ReactNode } from 'react';

interface LayoutProps {
  children: ReactNode;
  activeTab: string;
  onTabChange: (tab: string) => void;
  health: any;
}

export default function Layout({ children, activeTab, onTabChange, health }: LayoutProps) {
  const tabs = [
    { id: 'overview', name: 'Overview', icon: '📊' },
    { id: 'market', name: 'Market Data', icon: '💹' },
    { id: 'strategies', name: 'AI Strategies', icon: '🤖' },
    { id: 'portfolio', name: 'Portfolio', icon: '💼' },
    { id: 'analytics', name: 'Analytics', icon: '📈' },
    { id: 'settings', name: 'Settings', icon: '⚙️' }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">AXF</span>
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">AXF Bot 0</h1>
                  <p className="text-sm text-gray-600">AI-Powered Forex Trading</p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  health?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span className="text-sm text-gray-600">
                  {health?.status === 'healthy' ? 'System Online' : 'System Offline'}
                </span>
              </div>
              
              <div className="text-right">
                <p className="text-sm text-gray-500">Last Update</p>
                <p className="text-sm font-medium text-gray-900">
                  {new Date().toLocaleTimeString()}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.name}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </div>
    </div>
  );
}