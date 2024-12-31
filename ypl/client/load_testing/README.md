# Load Testing

First, run either a local instance of `yupp-mind` or the remote instance you'd like to test.
Next, install the optional client dependencies, e.g., `poetry install --with client`.
**WARNING:** Please make sure `--users` is not set to a very large number!!!

## QuickTake

This runs a load test on the `generate_quicktake` endpoint, where `--chat-id` is required and should be a chat ID that exists in the database.

```bash
locust --headless --users 1 -H http://localhost:8000 -f ypl/client/load_testing/generate_quicktake.py --chat-id aaa2e958-09f7-46ab-9923-0b1f3b305808
```

By default, the above creates a single user that makes a request every 10-20 seconds.
