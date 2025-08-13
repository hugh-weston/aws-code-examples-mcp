# AWS Docs Code Examples MCP Server

Model Context Protocol (MCP) server for AWS Docs Code Examples

This MCP server provides tools to access AWS code examples, search for content, and get recommendations.

## Features

- **Read Documentation**: Fetch and convert AWS documentation pages to markdown format
- **Search Documentation**: Search AWS documentation using the official search API (global only)
- **Recommendations**: Get content recommendations for AWS documentation pages (global only)
- **Get Available Services List**: Get a list of available AWS services in China regions (China only)

## Prerequisites

### Installation Requirements

1. Install `uv` from [Astral](https://docs.astral.sh/uv/getting-started/installation/) or the [GitHub README](https://github.com/astral-sh/uv#installation)
2. Install Python 3.10 or newer using `uv python install 3.10` (or a more recent version)

## Installation

Configure the MCP server in your MCP client configuration (e.g., for Amazon Q Developer CLI, edit `~/.aws/amazonq/mcp.json`):

```json
{
  "mcpServers": {
    "aws-code-examples-mcp": {
      "command": "uvx",
      "args": ["aws-code-examples-mcp@latest"],
      "env": {
        "FASTMCP_LOG_LEVEL": "ERROR",
        "AWS_DOCUMENTATION_PARTITION": "aws"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

> **Note**: Set `AWS_DOCUMENTATION_PARTITION` to `aws-cn` to query AWS China documentation instead of global AWS documentation.

## Basic Usage

Example:

- "look up documentation on S3 bucket naming rule. cite your sources"
- "recommend content for page https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html"

## Tools

### read_documentation

Fetches an AWS documentation page and converts it to markdown format.

```python
read_documentation(url: str) -> str
```

### search_documentation (global only)

Searches AWS documentation using the official AWS Documentation Search API.

```python
search_documentation(search_phrase: str, limit: int) -> list[dict]
```

### recommend (global only)

Gets content recommendations for an AWS documentation page.

```python
recommend(url: str) -> list[dict]
```

### get_available_services (China only)

Gets a list of available AWS services in China regions.

```python
get_available_services() -> str
```

## Development

Install dependencies:
```bash
uv sync --dev
```

Run tests:
```bash
uv run pytest
```

Run the server:
```bash
uv run aws-code-examples-mcp
```

## License

Apache-2.0
