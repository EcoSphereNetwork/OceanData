# Flow Integration

This guide explains how to connect the OceanData SDK CLI with automation tools such as n8n or Flowise using webhooks.

1. Start the CLI evaluation with the `--webhook <url>` option.
2. The CLI will POST the evaluation result as JSON to the provided URL.
3. In n8n or Flowise configure a webhook node to receive the payload and trigger further actions.
