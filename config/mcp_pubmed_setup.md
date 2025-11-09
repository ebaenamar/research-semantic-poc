# PubMed MCP Server Setup

## Installation

### Option 1: cyanheads/pubmed-mcp-server (Recommended)

```bash
# Clone the server
cd ~/mcp-servers
git clone https://github.com/cyanheads/pubmed-mcp-server.git
cd pubmed-mcp-server

# Install dependencies
npm install

# Build
npm run build
```

### Option 2: JackKuo666/PubMed-MCP-Server (Simpler)

```bash
cd ~/mcp-servers
git clone https://github.com/JackKuo666/PubMed-MCP-Server.git
cd PubMed-MCP-Server

npm install
npm run build
```

## Configuration in Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pubmed": {
      "command": "node",
      "args": ["/Users/YOUR_USERNAME/mcp-servers/pubmed-mcp-server/dist/index.js"],
      "env": {
        "NCBI_API_KEY": "your_ncbi_api_key_here",
        "LOG_LEVEL": "info"
      }
    }
  }
}
```

## Get NCBI API Key

1. Go to: https://www.ncbi.nlm.nih.gov/account/
2. Create account or log in
3. Settings → API Key Management
4. Create new API key
5. Copy and paste in config above

## Test Installation

In Claude Desktop or Claude Code:

```
Can you search PubMed for papers about "pediatric oncology immunotherapy" from 2023?
```

Should return:
- List of PMIDs
- Titles
- Authors
- Abstracts

## Available Tools

### cyanheads server:
- `pubmed_search_articles` - Advanced search
- `pubmed_fetch_contents` - Full metadata
- `pubmed_article_connections` - Citation networks
- `pubmed_research_agent` - Research planning
- `pubmed_generate_chart` - Visualizations

### JackKuo666 server:
- `search_pubmed` - Paper search
- `get_paper_details` - Metadata
- `analyze_paper` - Deep analysis
- `download_pdf` - PDF retrieval (when available)

## Rate Limits

- **With API key**: 10 requests/second
- **Without API key**: 3 requests/second

## Troubleshooting

### Server not starting
```bash
# Check if port is in use
lsof -i :3000

# Try running manually
node dist/index.js
```

### API errors
- Verify API key is valid
- Check NCBI status: https://www.ncbi.nlm.nih.gov/
- Respect rate limits

### No results
- Verify query syntax
- Check date ranges
- Try simpler queries first
