# Webhooks Service

This service handles incoming webhooks from partners using URL-based authentication tokens. Each partner gets a unique webhook URL containing their authentication token, making it simple to send webhook events securely.

## Database Schema

The service uses a single table to manage partner information and webhook configurations:

### WebhookPartner Table

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key and unique identifier for the partner |
| name | String | Display name of the partner (indexed) |
| description | String (optional) | Description of the partner's webhook integration |
| webhook_token | String | URL-safe token used in webhook URL (unique, indexed) |
| status | Enum | Partner status: ACTIVE, INACTIVE, or SUSPENDED |
| validation_config | JSONB | Partner-specific validation rules (e.g., required fields) |
| created_at | Timestamp | When the partner was added |
| modified_at | Timestamp | When the partner was last updated |
| deleted_at | Timestamp (optional) | Soft delete timestamp |

## Webhook System Overview

The service uses a secure token-based system for receiving webhooks. Each partner gets a unique webhook URL that includes their authentication token, eliminating the need for additional headers or authentication steps.

### How It Works

1. Each partner is registered in the `webhook_partners` table with a unique ID
2. A secure webhook token is generated and stored in the table
3. Partners receive a unique webhook URL containing this token
4. Our system validates the token and processes the webhook based on the partner's configuration

### Webhook URL Format

The webhook URL follows this pattern:
```
https://api.example.com/webhook/v1/{token}
```

For example:
```
https://api.example.com/webhook/v1/nNsCbbEO316t13WsH3fw4QIb
```

The token is a URL-safe string that uniquely identifies the partner in our system.

### Partner Integration Guide

Partners only need to:

1. Use their provided webhook URL
2. Send POST requests with JSON payloads
3. Use HTTPS for all requests

Example request:
```http
POST https://api.example.com/webhook/v1/nNsCbbEO316t13WsH3fw4QIb
Content-Type: application/json

{
    "event_type": "user.created",
    "timestamp": "2024-03-21T10:00:00Z",
    "data": {
        "user_id": "123",
        "email": "user@example.com"
    }
}
```

### Partner Onboarding Steps

1. **Generate Partner Token**
   Write a function to generate a secure, URL-safe token for webhook URLs.

2. **Create Partner Record**
   Create a one off script to add the partner to the database.

3. **Share Integration Guide**
   ```markdown
   # Webhook Integration Guide
   
   Your unique webhook URL:
   POST https://api.example.com/webhook/v1/{webhook_token}
   
   Content-Type: application/json
   
   Example Request:
   curl -X POST https://api.example.com/webhook/v1/{webhook_token} \
        -H "Content-Type: application/json" \
        -d '{"event_type": "user.created", "data": {...}}'
   ```

## Development

To add support for new webhook types:

1. Add appropriate validation rules in the partner's `validation_config`
2. Create a handler function for the partner's webhook events
3. Update the webhook handler to process events based on the partner's configuration 

## Deployment

The webhook service is deployed using GitHub Actions to Google Cloud Run. The deployment process is automated and includes the following key features:

### Deployment Configuration

- **Environments**: Supports both staging and production environments
- **Service Name**: Deployed as `webhooks-service`
- **Container Registry**: Images are stored in Google Container Registry (GCR)
- **Memory/CPU**: Configured with 24Gi memory and 6 CPU cores
- **Database**: Uses Cloud SQL for PostgreSQL database connectivity

### Deployment Process

1. **Build Stage**:
   - Uses Docker with BuildX for efficient layer caching
   - Builds from `Dockerfile.webhooks` with environment-specific configurations
   - Creates secure `.env` file with environment variables

2. **Authentication**:
   - Uses Google Cloud authentication for secure deployments
   - Manages secrets through GitHub Actions secrets

3. **Service Configuration**:
   - Runs on port 8080 using uvicorn server
   - Includes automatic health checks
   - Supports both manual triggers and automated deployments

### Deployment Commands

To manually trigger a deployment:
1. Go to GitHub Actions
2. Select "Build and Deploy" workflow
3. Choose "Run workflow"
4. Configure the following parameters:
   - Environment: `staging` or `production`
   - Image: `webhooks-service`
   - Region: `us-east4` or `us-central1`

The service automatically handles database migrations and configuration updates during deployment. 