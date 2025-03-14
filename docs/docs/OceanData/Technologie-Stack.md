graph TB
    %% Hauptebenen
    subgraph Präsentationsschicht
        React[React.js]
        Redux[Redux]
        Tailwind[Tailwind CSS]
        Recharts[Recharts]
    end
    
    subgraph Anwendungsschicht
        Python[Python 3.9+]
        FastAPI[FastAPI]
        AWSLambda[AWS Lambda]
    end
    
    subgraph KI-und-Analyseebene
        TensorFlow[TensorFlow]
        PyTorch[PyTorch]
        ScikitLearn[Scikit-learn]
        NLTK[NLTK]
        Transformers[HuggingFace Transformers]
        Pandas[Pandas]
        NumPy[NumPy]
    end
    
    subgraph Datenspeicherebene
        MongoDB[MongoDB]
        PostgreSQL[PostgreSQL]
        S3[AWS S3]
    end
    
    subgraph Datenschutzebene
        Cryptography[Cryptography]
        Fernet[Fernet]
        DiffPrivLib[Diffprivlib]
    end
    
    subgraph Blockchain-und-Tokenisierungsebene
        OceanJS[Ocean.js]
        Solidity[Solidity]
        Web3[Web3.js]
        Ethereum[Ethereum]
    end
    
    %% Abhängigkeiten innerhalb Schichten
    React --> Redux
    React --> Tailwind
    React --> Recharts
    
    Python --> FastAPI
    Python --> AWSLambda
    
    Python --> TensorFlow
    Python --> PyTorch
    Python --> ScikitLearn
    Python --> NLTK
    Python --> Transformers
    Python --> Pandas
    Python --> NumPy
    
    Python --> MongoDB
    Python --> PostgreSQL
    Python --> S3
    
    Python --> Cryptography
    Cryptography --> Fernet
    Python --> DiffPrivLib
    
    Python --> OceanJS
    OceanJS --> Web3
    Web3 --> Ethereum
    Solidity --> Ethereum
    
    %% Schicht-übergreifende Abhängigkeiten
    Anwendungsschicht --> KI-und-Analyseebene
    Anwendungsschicht --> Datenspeicherebene
    Anwendungsschicht --> Datenschutzebene
    Anwendungsschicht --> Blockchain-und-Tokenisierungsebene
    
    Präsentationsschicht --> Anwendungsschicht
    
    %% Styling
    classDef presentation fill:#ff9999,stroke:#333,stroke-width:1px
    classDef application fill:#99ccff,stroke:#333,stroke-width:1px
    classDef ai fill:#c2f0c2,stroke:#333,stroke-width:1px
    classDef data fill:#f9f9ad,stroke:#333,stroke-width:1px
    classDef privacy fill:#d8bfd8,stroke:#333,stroke-width:1px
    classDef blockchain fill:#ffcc99,stroke:#333,stroke-width:1px
    
    class React,Redux,Tailwind,Recharts presentation
    class Python,FastAPI,AWSLambda application
    class TensorFlow,PyTorch,ScikitLearn,NLTK,Transformers,Pandas,NumPy ai
    class MongoDB,PostgreSQL,S3 data
    class Cryptography,Fernet,DiffPrivLib privacy
    class OceanJS,Solidity,Web3,Ethereum blockchain
