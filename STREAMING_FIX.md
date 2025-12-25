# Streaming Fix for EXO Chat Completions

## Problem

When using `stream: true` in `/v1/chat/completions` requests, the proxy was returning an error:

```json
{
  "detail": "Failed to connect to EXO API at http://192.168.1.188:52415: All connection attempts failed"
}
```

**Root Cause:** The async context manager (`async with client.stream()`) was closing the HTTP connection before the `StreamingResponse` generator could consume the data.

## Solution

### Backend Fix (`app/api/v1/routes/exo_proxy.py`)

**Before (Broken):**
```python
async with client.stream(...) as response:
    # ... error checking ...
    
    if is_streaming:
        async def stream_generator():
            async for chunk in response.aiter_bytes():
                yield chunk
        
        return StreamingResponse(stream_generator(), ...)
    # The async with exits here, closing the connection
    # But the generator hasn't started consuming yet!
```

**After (Fixed):**
```python
if is_streaming:
    # Move the entire stream lifecycle INSIDE the generator
    async def stream_generator():
        async with client.stream(...) as response:
            # Error checking and streaming all in one place
            async for chunk in response.aiter_bytes():
                yield chunk
    
    return StreamingResponse(stream_generator(), ...)
```

**Key Changes:**

1. **Moved `async with` inside the generator** - The connection stays open as long as the generator is consuming data
2. **Increased timeout for streaming** - Changed from 60s to 120s for longer responses
3. **Error handling in SSE format** - Streaming errors are now returned as SSE `data:` events
4. **Added streaming headers**:
   - `Cache-Control: no-cache`
   - `Connection: keep-alive`
   - `X-Accel-Buffering: no` (prevents nginx from buffering)

5. **Separated streaming and non-streaming paths** - Non-streaming can safely use `async with` at the top level since it collects all chunks before returning

### Frontend Fix (`app/templates/admin/api_tester.html`)

Added error detection in the SSE stream:

```javascript
const chunkData = JSON.parse(jsonStr);

// Check for errors in the stream
if (chunkData.error) {
    // Display error and stop streaming
    throw new Error(chunkData.error.message);
}

// Normal content processing
if (chunkData.choices && ...) {
    fullContent += chunkData.choices[0].delta.content;
}
```

## How It Works Now

### Streaming Mode (`stream: true`)

```
1. Client sends request with stream: true
2. Proxy creates a generator function
3. Generator opens connection to EXO (inside async with)
4. Generator yields chunks as they arrive from EXO
5. FastAPI's StreamingResponse sends chunks to client
6. Connection closes when generator finishes (async with exits)
```

### Non-Streaming Mode (`stream: false`)

```
1. Client sends request with stream: false  
2. Proxy opens connection to EXO (async with)
3. Proxy collects ALL chunks from EXO stream
4. Proxy builds a single JSON response
5. Proxy closes connection (async with exits)
6. Proxy returns complete JSON to client
```

## Technical Details

### Why the Original Code Failed

The issue is with **async context manager lifecycle vs. generator execution**:

```python
async with client.stream() as response:  # 1. Connection opens
    async def generator():                # 2. Define generator
        async for chunk in response:      # 4. Try to read (FAILS - connection closed!)
            yield chunk
    return StreamingResponse(generator()) # 3. Return (exits async with, closes connection)
```

**The generator doesn't start executing until FastAPI begins sending the response**, which happens AFTER the function returns. By that time, the `async with` has already exited and closed the connection.

### Why the Fix Works

```python
async def generator():                        # 1. Define generator
    async with client.stream() as response:   # 3. Connection opens (when generator starts)
        async for chunk in response:          # 4. Read chunks (connection is open)
            yield chunk                       # 5. Yield to client
                                             # 6. Connection closes when generator ends
return StreamingResponse(generator())         # 2. Return (generator not started yet)
```

The `async with` is now **inside** the generator, so it doesn't execute until FastAPI starts consuming the generator. This keeps the connection open for the entire streaming duration.

## Testing

### Test Non-Streaming Mode:
```bash
curl -X POST http://localhost:8000/admin/exo-api-test/chat/completions?exo_base_url=http%3A%2F%2F192.168.1.188%3A52415 \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

Expected: Single JSON response with complete content

### Test Streaming Mode:
```bash
curl -N -X POST http://localhost:8000/admin/exo-api-test/chat/completions?exo_base_url=http%3A%2F%2F192.168.1.188%3A52415 \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Expected: Server-Sent Events stream with `data: {...}` chunks

## Related Issues

This is a common pitfall when using httpx's streaming with FastAPI:

- The generator must own the async context manager
- Don't close the response before the generator finishes
- Use appropriate timeouts for long-running streams

## Files Modified

1. **`app/api/v1/routes/exo_proxy.py`**
   - Restructured streaming logic
   - Added error handling in SSE format
   - Increased streaming timeout
   - Added streaming headers

2. **`app/templates/admin/api_tester.html`**
   - Added error detection in SSE stream
   - Better error display for streaming failures

## Status

✅ **Fixed:** Both streaming and non-streaming modes now work correctly
- Non-streaming: Returns complete JSON response
- Streaming: Returns real-time SSE chunks
- Error handling: Works in both modes

