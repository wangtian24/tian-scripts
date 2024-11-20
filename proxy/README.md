# Amplitude Analytics Proxy

This is an API Gateway proxy on GCP to opaquely forward analytics events from Amplitude SDK from user brwosers to Amplitude's HTTP API V2. It exists to prevent adblockers such as uBlock Origin or Safari from blocking requests to Amplitude's API.

## API Endpoint

- **Endpoint**: `/2/httpapi`
- **Method**: POST (for sending analytics events), OPTIONS (for preflight requests)

## Deployment

First, make changes to the yaml file, then create a new api config (configuration is immutable). 
```
gcloud api-gateway api-configs create NAME_OF_NEW_CONFIG \
  --api=amplitude-proxy --openapi-spec=openapi2-functions.yaml \
  --project=yupp-llms
```

Then deploy the api gateway to your region.
```
gcgcloud api-gateway gateways update amplitude-proxy-gateway \
  --api=amplitude-proxy --api-config=NAME_OF_NEW_CONFIG \
  --location=us-east4 --project=yupp-llms
```

## See more
* [Amplitude documentation on domain proxies](https://amplitude.com/docs/analytics/domain-proxy)
* [How to set up API proxy in GCP](https://cloud.google.com/api-gateway/docs/secure-traffic-gcloud#securing_access_by_using_an_api_key)
