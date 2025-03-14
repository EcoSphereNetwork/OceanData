import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ShieldAlert, Database, TrendingUp, DollarSign, Lock, Zap, PieChart, BarChart2 } from 'lucide-react';

const DataTokenizationDashboard = () => {
  const [selectedDataSource, setSelectedDataSource] = useState(null);
  const [dataSources, setDataSources] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isTokenizing, setIsTokenizing] = useState(false);
  const [tokenizationResult, setTokenizationResult] = useState(null);
  
  // Mock data for demonstration
  useEffect(() => {
    // Simulate loading connected data sources
    const mockDataSources = [
      { 
        id: 'browser-chrome', 
        name: 'Chrome Browser History', 
        type: 'browser',
        recordCount: 4582,
        dateRange: 'Jan 1, 2024 - Mar 14, 2025',
        status: 'connected',
        lastSync: '2025-03-14T08:30:00Z',
        privacyLevel: 'medium',
        estimatedValue: 3.4
      },
      { 
        id: 'smartwatch-fitbit', 
        name: 'Fitbit Health Data', 
        type: 'smartwatch',
        recordCount: 8760,
        dateRange: 'Jan 1, 2024 - Mar 14, 2025',
        status: 'connected',
        lastSync: '2025-03-14T07:15:00Z',
        privacyLevel: 'high',
        estimatedValue: 5.2
      },
      { 
        id: 'smarthome-nest', 
        name: 'Nest Thermostat Data', 
        type: 'iot_thermostat',
        recordCount: 2190,
        dateRange: 'Jan 1, 2024 - Mar 14, 2025',
        status: 'connected',
        lastSync: '2025-03-14T06:00:00Z',
        privacyLevel: 'medium',
        estimatedValue: 2.8
      }
    ];
    
    setDataSources(mockDataSources);
  }, []);
  
  // Simulate data analysis process
  const analyzeData = (dataSource) => {
    setIsAnalyzing(true);
    setAnalysisProgress(0);
    
    // Simulate progress updates
    const interval = setInterval(() => {
      setAnalysisProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsAnalyzing(false);
          
          // Generate mock analysis results
          const mockResults = generateMockAnalysisResults(dataSource);
          setAnalysisResults(mockResults);
          
          return 100;
        }
        return prev + 5;
      });
    }, 300);
  };
  
  // Generate mock analysis results based on data source
  const generateMockAnalysisResults = (dataSource) => {
    // Different mock data for each source type
    if (dataSource.type === 'browser') {
      return {
        sourceType: 'browser',
        recordCount: dataSource.recordCount,
        valueEstimate: dataSource.estimatedValue,
        anomalies: {
          count: 32,
          percentage: 0.7,
          insights: [
            { feature: 'duration', description: 'Unusually long sessions detected' }
          ]
        },
        timePatterns: {
          peakUsageTimes: ['8:00 AM', '12:00 PM', '8:00 PM'],
          weekdayUsage: [65, 70, 68, 72, 75, 45, 40],
          weekdayLabels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        },
        topFeatures: [
          { name: 'duration', importance: 0.85 },
          { name: 'time_of_day', importance: 0.72 },
          { name: 'domain_category', importance: 0.65 }
        ],
        potentialUsesCases: [
          { title: 'Consumer Behavior Analysis', description: 'Analyze browsing patterns to understand consumer preferences', valueScore: 0.8 },
          { title: 'Content Recommendation', description: 'Improve content recommendation systems', valueScore: 0.75 }
        ],
        privacyLevel: dataSource.privacyLevel,
        valuationFactors: {
          dataSize: 0.68,
          dataQuality: 0.85,
          uniqueness: 0.72,
          timeRelevance: 0.95
        }
      };
    } else if (dataSource.type === 'smartwatch') {
      return {
        sourceType: 'smartwatch',
        recordCount: dataSource.recordCount,
        valueEstimate: dataSource.estimatedValue,
        anomalies: {
          count: 48,
          percentage: 0.5,
          insights: [
            { feature: 'heart_rate', description: 'Unusual patterns detected during nighttime' }
          ]
        },
        timePatterns: {
          activityByHour: [300, 250, 200, 150, 100, 120, 400, 800, 1200, 1100, 1000, 1300, 1200, 1100, 900, 1000, 1100, 1200, 900, 700, 600, 500, 400, 350],
          hourLabels: Array.from({length: 24}, (_, i) => `${i}:00`)
        },
        topFeatures: [
          { name: 'steps', importance: 0.9 },
          { name: 'heart_rate', importance: 0.85 },
          { name: 'sleep_quality', importance: 0.8 }
        ],
        potentialUsesCases: [
          { title: 'Health Pattern Analysis', description: 'Identify correlations between activity and health', valueScore: 0.9 },
          { title: 'Fitness Program Optimization', description: 'Improve workout recommendations', valueScore: 0.85 }
        ],
        privacyLevel: dataSource.privacyLevel,
        valuationFactors: {
          dataSize: 0.75,
          dataQuality: 0.9,
          uniqueness: 0.85,
          timeRelevance: 0.95
        }
      };
    } else {
      // Generic results for other types
      return {
        sourceType: dataSource.type,
        recordCount: dataSource.recordCount,
        valueEstimate: dataSource.estimatedValue,
        anomalies: {
          count: Math.floor(dataSource.recordCount * 0.01),
          percentage: 1.0,
          insights: [
            { feature: 'main_field', description: 'Some unusual patterns detected' }
          ]
        },
        timePatterns: {
          monthlyUsage: [65, 68, 70, 72, 75, 80, 82, 85, 83, 80, 78, 75],
          monthLabels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        },
        topFeatures: [
          { name: 'feature1', importance: 0.8 },
          { name: 'feature2', importance: 0.7 },
          { name: 'feature3', importance: 0.6 }
        ],
        potentialUsesCases: [
          { title: 'General Analysis', description: 'Analyze patterns in the data', valueScore: 0.7 },
          { title: 'Predictive Modeling', description: 'Forecast future trends', valueScore: 0.65 }
        ],
        privacyLevel: dataSource.privacyLevel,
        valuationFactors: {
          dataSize: 0.6,
          dataQuality: 0.7,
          uniqueness: 0.65,
          timeRelevance: 0.8
        }
      };
    }
  };
  
  // Simulate tokenization process
  const tokenizeData = () => {
    if (!analysisResults) return;
    
    setIsTokenizing(true);
    
    // Simulate tokenization process
    setTimeout(() => {
      // Mock tokenization result
      const result = {
        success: true,
        tokenSymbol: `DT${Date.now().toString().slice(-6)}`,
        tokenAddress: `0x${Math.random().toString(16).slice(2, 42)}`,
        marketplaceUrl: `https://market.oceanprotocol.com/asset/${Math.random().toString(16).slice(2, 42)}`,
        tokenPrice: analysisResults.valueEstimate,
        createdAt: new Date().toISOString(),
        privacyLevel: analysisResults.privacyLevel
      };
      
      setTokenizationResult(result);
      setIsTokenizing(false);
    }, 3000);
  };
  
  // Get status badge color based on privacy level
  const getPrivacyBadgeColor = (level) => {
    switch(level) {
      case 'low': return 'bg-yellow-500';
      case 'medium': return 'bg-blue-500';
      case 'high': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };
  
  // Get a formatted date string
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };
  
  return (
    <div className="w-full max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Data Monetization Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {/* Step 1: Select Data Source */}
        <Card className={`${selectedDataSource ? 'border-blue-400' : 'border-dashed'}`}>
          <div className="p-6">
            <h3 className="flex items-center text-lg font-semibold mb-2">
              <Database className="mr-2 h-5 w-5" />
              Step 1: Select Data Source
            </h3>
            <p className="text-sm text-gray-500 mb-4">Choose which data you want to monetize</p>
            
            {dataSources.length > 0 ? (
              <div className="space-y-4">
                {dataSources.map(source => (
                  <div 
                    key={source.id} 
                    className={`p-4 border rounded-lg cursor-pointer transition-all ${selectedDataSource?.id === source.id ? 'border-blue-500 bg-blue-50' : 'hover:border-blue-300'}`}
                    onClick={() => setSelectedDataSource(source)}
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium">{source.name}</h3>
                        <p className="text-sm text-gray-500">{source.recordCount.toLocaleString()} records</p>
                        <p className="text-sm text-gray-500">{source.dateRange}</p>
                      </div>
                      <Badge className={getPrivacyBadgeColor(source.privacyLevel)}>
                        {source.privacyLevel}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-center py-6 text-gray-500">No data sources connected</p>
            )}
            
            <p className="text-sm text-gray-500 mt-4">Last synchronized: {selectedDataSource ? formatDate(selectedDataSource.lastSync) : 'N/A'}</p>
          </div>
        </Card>
        
        {/* Step 2: Analyze Data */}
        <Card className={`${analysisResults ? 'border-blue-400' : 'border-dashed'}`}>
          <div className="p-6">
            <h3 className="flex items-center text-lg font-semibold mb-2">
              <TrendingUp className="mr-2 h-5 w-5" />
              Step 2: Analyze Data
            </h3>
            <p className="text-sm text-gray-500 mb-4">AI analysis to determine data value</p>
            
            {selectedDataSource ? (
              isAnalyzing ? (
                <div className="space-y-4 py-6">
                  <p className="text-center">Analyzing {selectedDataSource.name}...</p>
                  <Progress value={analysisProgress} />
                  <p className="text-center text-sm text-gray-500">{analysisProgress}% complete</p>
                </div>
              ) : analysisResults ? (
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">Estimated Value:</span>
                    <span className="text-xl font-bold">{analysisResults.valueEstimate.toFixed(2)} OCEAN</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-medium">Records:</span>
                    <span>{analysisResults.recordCount.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-medium">Anomalies:</span>
                    <span>{analysisResults.anomalies.count} ({analysisResults.anomalies.percentage.toFixed(1)}%)</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="font-medium">Quality Score:</span>
                    <div className="flex items-center">
                      <Progress value={analysisResults.valuationFactors.dataQuality * 100} className="w-24 mr-2" />
                      <span>{(analysisResults.valuationFactors.dataQuality * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-6">
                  <Button onClick={() => analyzeData(selectedDataSource)}>
                    Start Analysis
                  </Button>
                  <p className="mt-3 text-sm text-gray-500">This will evaluate your data's market value</p>
                </div>
              )
            ) : (
              <p className="text-center py-6 text-gray-500">Select a data source first</p>
            )}
            
            <p className="text-sm text-gray-500 mt-4">AI analysis uses privacy-preserving techniques</p>
          </div>
        </Card>
        
        {/* Step 3: Tokenize & Publish */}
        <Card className={`${tokenizationResult ? 'border-blue-400' : 'border-dashed'}`}>
          <div className="p-6">
            <h3 className="flex items-center text-lg font-semibold mb-2">
              <DollarSign className="mr-2 h-5 w-5" />
              Step 3: Tokenize & Publish
            </h3>
            <p className="text-sm text-gray-500 mb-4">Create a data token and list on Ocean Market</p>
            
            {analysisResults ? (
              isTokenizing ? (
                <div className="space-y-4 py-6">
                  <p className="text-center">Tokenizing data...</p>
                  <div className="flex justify-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
                  </div>
                  <p className="text-center text-sm text-gray-500">Creating data token on Ocean Protocol</p>
                </div>
              ) : tokenizationResult ? (
                <div className="space-y-4">
                  <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <div className="text-center mb-2 font-medium text-green-700">Successfully Tokenized!</div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <span className="text-gray-600">Token Symbol:</span>
                      <span className="font-mono">{tokenizationResult.tokenSymbol}</span>
                      <span className="text-gray-600">Initial Price:</span>
                      <span>{tokenizationResult.tokenPrice.toFixed(2)} OCEAN</span>
                      <span className="text-gray-600">Created:</span>
                      <span>{new Date(tokenizationResult.createdAt).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <Button 
                    className="w-full" 
                    onClick={() => window.open(tokenizationResult.marketplaceUrl, '_blank')}
                  >
                    View on Ocean Market
                  </Button>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-6">
                  <Button onClick={tokenizeData}>
                    Tokenize Data
                  </Button>
                  <p className="mt-3 text-sm text-gray-500">Create a data token worth ~{analysisResults.valueEstimate.toFixed(2)} OCEAN</p>
                </div>
              )
            ) : (
              <p className="text-center py-6 text-gray-500">Complete analysis first</p>
            )}
            
            <p className="text-sm text-gray-500 mt-4">Uses Ocean Protocol for secure tokenization</p>
          </div>
        </Card>
      </div>
      
      {/* Analysis Details */}
      {analysisResults && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold mb-2">Data Analysis Results</h3>
            <p className="text-sm text-gray-500 mb-4">Detailed insights and valuation factors</p>
            
            <Tabs defaultValue="insights">
              <TabsList className="mb-4">
                <TabsTrigger value="insights">Key Insights</TabsTrigger>
                <TabsTrigger value="patterns">Time Patterns</TabsTrigger>
                <TabsTrigger value="valuation">Valuation Factors</TabsTrigger>
                <TabsTrigger value="use-cases">Potential Use Cases</TabsTrigger>
              </TabsList>
              
              <TabsContent value="insights">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card>
                    <div className="p-4">
                      <h4 className="text-base flex items-center font-medium mb-4">
                        <ShieldAlert className="mr-2 h-4 w-4" />
                        Anomaly Detection
                      </h4>
                      <p>Found {analysisResults.anomalies.count} anomalies ({analysisResults.anomalies.percentage.toFixed(1)}% of data)</p>
                      {analysisResults.anomalies.insights.map((insight, i) => (
                        <div key={i} className="mt-2 p-2 bg-yellow-50 border border-yellow-100 rounded">
                          <p className="text-sm"><strong>{insight.feature}:</strong> {insight.description}</p>
                        </div>
                      ))}
                    </div>
                  </Card>
                  
                  <Card>
                    <div className="p-4">
                      <h4 className="text-base flex items-center font-medium mb-4">
                        <BarChart2 className="mr-2 h-4 w-4" />
                        Feature Importance
                      </h4>
                      <div className="space-y-4">
                        {analysisResults.topFeatures.map((feature, i) => (
                          <div key={i} className="space-y-1">
                            <div className="flex justify-between text-sm">
                              <span>{feature.name}</span>
                              <span>{(feature.importance * 100).toFixed(0)}%</span>
                            </div>
                            <Progress value={feature.importance * 100} />
                          </div>
                        ))}
                      </div>
                    </div>
                  </Card>
                </div>
              </TabsContent>
              
              <TabsContent value="patterns">
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    {'weekdayUsage' in analysisResults.timePatterns ? (
                      <BarChart data={analysisResults.timePatterns.weekdayLabels.map((day, i) => ({
                        name: day,
                        value: analysisResults.timePatterns.weekdayUsage[i]
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="value" fill="#3b82f6" />
                      </BarChart>
                    ) : 'activityByHour' in analysisResults.timePatterns ? (
                      <LineChart data={analysisResults.timePatterns.hourLabels.map((hour, i) => ({
                        name: hour,
                        value: analysisResults.timePatterns.activityByHour[i]
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="value" stroke="#3b82f6" />
                      </LineChart>
                    ) : (
                      <BarChart data={analysisResults.timePatterns.monthLabels.map((month, i) => ({
                        name: month,
                        value: analysisResults.timePatterns.monthlyUsage[i]
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="value" fill="#3b82f6" />
                      </BarChart>
                    )}
                  </ResponsiveContainer>
                </div>
              </TabsContent>
              
              <TabsContent value="valuation">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card>
                    <div className="p-4">
                      <h4 className="text-base flex items-center font-medium mb-4">
                        <PieChart className="mr-2 h-4 w-4" />
                        Valuation Factors
                      </h4>
                      <div className="space-y-4">
                        {Object.entries(analysisResults.valuationFactors).map(([key, value]) => (
                          <div key={key} className="space-y-1">
                            <div className="flex justify-between text-sm">
                              <span className="capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
                              <span>{(value * 100).toFixed(0)}%</span>
                            </div>
                            <Progress value={value * 100} />
                          </div>
                        ))}
                      </div>
                    </div>
                  </Card>
                  
                  <Card>
                    <div className="p-4">
                      <h4 className="text-base flex items-center font-medium mb-4">
                        <DollarSign className="mr-2 h-4 w-4" />
                        Value Estimate
                      </h4>
                      <div className="flex items-center justify-center p-6">
                        <div className="text-center">
                          <div className="text-4xl font-bold text-blue-600 mb-2">
                            {analysisResults.valueEstimate.toFixed(2)}
                          </div>
                          <div className="text-xl">OCEAN Tokens</div>
                          <div className="text-sm text-gray-500 mt-2">Estimated market value</div>
                        </div>
                      </div>
                    </div>
                  </Card>
                </div>
              </TabsContent>
              
              <TabsContent value="use-cases">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {analysisResults.potentialUsesCases.map((useCase, i) => (
                    <Card key={i}>
                      <div className="p-4">
                        <h4 className="text-base flex items-center font-medium mb-2">
                          <Zap className="mr-2 h-4 w-4" />
                          {useCase.title}
                        </h4>
                        <p className="text-sm mb-4">{useCase.description}</p>
                        <div className="flex items-center">
                          <span className="text-sm mr-2">Value potential:</span>
                          <Progress value={useCase.valueScore * 100} className="flex-grow" />
                          <span className="text-sm ml-2">{(useCase.valueScore * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </Card>
                  ))}
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </Card>
      )}
    </div>
  );
};

export default DataTokenizationDashboard;
