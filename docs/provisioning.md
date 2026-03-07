# Model Provisioning

Willow models are delivered as optimized `.int8` binary signatures. The SDK supports three primary workflows for handling these files.

### 1. Ephemeral RAM Loading (Recommended)
Load the model directly into RAM over HTTPS. The model is never written to disk, ensuring maximum security and DRM compliance.

```python
from willow import WillowClient

# Initialize with full credentials from your Partner Dashboard
client = WillowClient(
    api_url="https://api.your-gateway.com",
    api_key="sk_live_...",
    customer_id="cust_12345"
)

# Fetch directly to memory
model = client.get_model("model-id-here") 
```

### 2. Local Disk Caching
Download the model for offline usage in field environments or air-gapped devices.

```python
# Initialize Client
client = WillowClient(
    api_url="https://api.your-gateway.com",
    api_key="sk_live_...",
    customer_id="cust_12345"
)

# Save to disk
client.download_model("model-id-here", "./cache/action.int8")
```

### 3. Air-Gapped Manual Loading
Load models that were manually downloaded from the Willow Web Interface. This method requires no API credentials.

```python
from willow import load_local_model

model = load_local_model("/path/to/downloaded/model.int8")
```