# Model Provisioning

Willow models are delivered as optimized `.int8` binary signatures. The SDK supports three primary workflows for handling these files.

### 1. Ephemeral RAM Loading (Recommended)
Load the model directly into RAM over HTTPS. The model is never written to disk, ensuring maximum security and DRM compliance.

```python
from willow import WillowClient

client = WillowClient(api_key="YOUR_API_KEY")
model = client.get_model("model-id-here") # Stays in RAM
```

### 2. Local Disk Caching
Download the model for offline usage in field environments.

```python
client.download_model("model-id-here", "./cache/action.int8")
```

### 3. Air-Gapped Manual Loading
Load models that were manually downloaded from the Willow Web Interface.

```python
from willow import load_local_model

model = load_local_model("/path/to/downloaded/model.int8")
```