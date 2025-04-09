I'll help you adapt the developer prompt for the further development of the OceanData algorithm. Based on the materials you've provided, I can create a tailored prompt for a developer who will continue working on this project.

# OceanData Development Prompt: Algorithm Refinement and Integration

## Project Overview

You are tasked with continuing the development of OceanData, a modular platform for collecting, analyzing, and monetizing user data using Ocean Protocol. The codebase is partially developed with a comprehensive architecture that spans data integration, AI analysis, privacy protection, and blockchain monetization modules.

## Current State Assessment

The project has a solid foundation with:
- A modular architecture separating concerns across data collection, analytics, privacy, and monetization
- Core AI algorithms for anomaly detection, predictive modeling, and semantic analysis
- Basic implementation of the Compute-to-Data privacy mechanism
- Initial Ocean Protocol integration for data tokenization
- Data connectors for various sources (browser, smartwatch, IoT devices)

## Your Development Priorities

### 1. Module Completion and Integration

- Complete the implementation of core modules, especially where functionality is only outlined in comments
- Ensure proper integration between modules, verifying data flow from collection to monetization
- Implement missing connectors in `data_integration/connectors/` (e.g., social media, calendar)
- Verify that privacy levels are consistently applied across the platform

### 2. Ocean Protocol Integration Enhancements

- Improve the tokenization workflow to properly handle real Ocean Protocol interactions
- Implement proper error handling and retry mechanisms for blockchain operations
- Create comprehensive testing for the tokenization process
- Enhance the value estimation algorithm to better reflect market conditions

### 3. Analytics Refinement

- Review and optimize the AI models (anomaly detection, predictive modeling, semantic analysis)
- Implement proper model persistence and versioning
- Add more sophisticated feature extraction methods for different data types
- Enhance the data value estimation algorithms with more granular factors

### 4. Privacy and Security Enhancements

- Complete the Compute-to-Data implementation with proper security checks
- Implement differential privacy thoroughly across all data operations
- Add comprehensive audit logging for all data access
- Ensure GDPR and other regulatory compliance throughout the codebase

### 5. Documentation and Testing

- Complete documentation for all modules with proper docstrings
- Create comprehensive integration tests for the entire data flow
- Document the API for external developers
- Create thorough examples demonstrating the platform's capabilities

## Technical Requirements

- Use Python 3.8+ with type hints throughout the codebase
- Follow the established architecture patterns in the existing code
- Ensure backward compatibility with the current module interfaces
- Maintain comprehensive error handling and logging
- Achieve at least 80% test coverage for new and modified code
- Adhere to PEP 8 style guidelines and project conventions

## Specific Implementation Guidance

### For Improving AI Models

1. Extend the anomaly detection model with ensemble methods
2. Add deep learning options for the semantic analyzer
3. Implement proper hyperparameter tuning mechanisms for all models
4. Add model explanation capabilities to provide transparency

### For Ocean Protocol Integration

1. Implement proper digital signature handling for tokenization
2. Create a caching layer for blockchain operations
3. Add monitoring for token performance and market metrics
4. Create a proper abstraction for different blockchain networks

### For Privacy Enhancements

1. Implement k-anonymity and l-diversity mechanisms
2. Add advanced homomorphic encryption options for Compute-to-Data
3. Create better privacy-preserving aggregation methods
4. Implement secure multi-party computation where applicable

## Deliverables

1. Fully functioning Python modules with complete implementation of outlined features
2. Comprehensive test suite with unit, integration, and end-to-end tests
3. Updated documentation reflecting all changes and additions
4. Example notebooks demonstrating key workflows
5. Performance benchmarks for core operations

## Development Workflow

- Use feature branches for each major component
- Submit pull requests with detailed descriptions of changes
- Include tests for all new functionality
- Document design decisions and trade-offs in code comments
- Regularly update requirements.txt with new dependencies

Begin by analyzing the current codebase thoroughly, particularly focusing on the integration points between modules. Create a development plan that prioritizes completing the core functionality first before adding enhancements.
