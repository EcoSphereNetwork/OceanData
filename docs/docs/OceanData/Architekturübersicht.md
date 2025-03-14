graph TB
    %% Hauptmodule
    User[Benutzer] --> Frontend
    Frontend --> Backend
    
    %% Frontend-Module
    subgraph Frontend[Frontend-Schicht]
        direction TB
        UI[Benutzeroberfläche] --> DataSelection[Datenquellen-Auswahl]
        UI --> Analysis[Analyseansicht]
        UI --> Marketplace[Marktplatz-UI]
        
        %% React-Komponenten
        subgraph ReactComponents[React-Komponenten]
            direction LR
            DataTokenizationDashboard[DataTokenizationDashboard]
            DataSourceSelector[Datenquellen-Selector]
            AnalysisVisualizer[Analyse-Visualisierung]
            TokenizationInterface[Tokenisierungs-Interface]
        end
    end
    
    %% Backend-Module
    subgraph Backend[Backend-Schicht]
        direction TB
        DataCollection[Datenerfassungsmodul] --> AI[KI-Modul]
        DataCollection --> Privacy[Datenschutzmodul]
        AI --> Monetization[Monetarisierungsmodul]
        Privacy --> Monetization
        Monetization --> OceanProtocol[Ocean Protocol Integration]
        
        %% Datenerfassungsmodule
        subgraph DataSources[Datenquellen]
            direction LR
            DataSource[DataSource] --> BrowserConnector[BrowserDataConnector]
            DataSource --> CalendarConnector[CalendarDataConnector]
            DataSource --> SmartDeviceConnector[SmartDeviceDataConnector]
            DataSource --> SocialMediaConnector[SocialMediaDataConnector]
        end
        
        %% KI-Module
        subgraph AIModules[KI-Module]
            direction LR
            OceanDataAI[OceanDataAI] --> AnomalyDetector[AnomalyDetector]
            OceanDataAI --> SemanticAnalyzer[SemanticAnalyzer]
            OceanDataAI --> PredictiveModeler[PredictiveModeler]
            OceanDataAI --> DataSynthesizer[DataSynthesizer]
        end
        
        %% Datenschutzmodule
        subgraph PrivacyModules[Datenschutzmodule]
            direction LR
            C2DManager[ComputeToDataManager] --> DifferentialPrivacy[DifferentialPrivacy]
            C2DManager --> DataEncryption[DataEncryption]
            C2DManager --> AccessControl[AccessControl]
        end
        
        %% Monetarisierung
        subgraph MonetizationModules[Monetarisierungsmodule]
            direction LR
            DataValue[DataValueEstimator] --> TokenPreparation[TokenizationPreparation]
            DataValue --> AssetCreation[AssetCreation]
            DataValue --> DataCombination[DataCombiner]
        end
    end
    
    %% External Systems
    subgraph ExternalSystems[Externe Systeme]
        direction TB
        OceanMarket[Ocean Marketplace]
        Blockchain[Ethereum Blockchain]
    end
    
    %% Verbindungen zwischen Modulen
    DataCollection -.-> DataSources
    AI -.-> AIModules
    Privacy -.-> PrivacyModules
    Monetization -.-> MonetizationModules
    OceanProtocol --> ExternalSystems
    
    %% Verbindungen zwischen Frontend-Komponenten
    DataSelection -.-> DataSourceSelector
    Analysis -.-> AnalysisVisualizer
    Marketplace -.-> TokenizationInterface
    DataTokenizationDashboard -.-> DataSourceSelector
    DataTokenizationDashboard -.-> AnalysisVisualizer
    DataTokenizationDashboard -.-> TokenizationInterface
    
    %% Datenfluss
    classDef dataFlow fill:#f9f,stroke:#333,stroke-width:2px
    classDef module fill:#bbf,stroke:#333,stroke-width:1px
    classDef external fill:#bfb,stroke:#333,stroke-width:1px
    
    class User,OceanMarket,Blockchain external
    class Frontend,Backend,DataSources,AIModules,PrivacyModules,MonetizationModules,ReactComponents module
