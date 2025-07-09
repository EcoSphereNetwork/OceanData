# Security Mocking

During development the SDK operates in a mocked mode for blockchain and Compute-to-Data interactions. Set the environment variable `OCEAN_MOCK=true` (default) to enable mocks.

When disabled, the adapters expect real credentials and will attempt to access the Ocean Protocol network.
