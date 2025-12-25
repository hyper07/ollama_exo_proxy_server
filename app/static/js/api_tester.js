function getBaseUrl() {
    return document.getElementById('base-url').value.replace(/\/$/, '');
}

function getApiKey() {
    return document.getElementById('api-key').value.trim();
}

function getCsrfToken() {
    const csrfInput = document.querySelector('input[name="csrf_token"]');
    return csrfInput ? csrfInput.value : '';
}

function createFormData() {
    const formData = new FormData();
    formData.append('csrf_token', getCsrfToken());
    formData.append('exo_base_url', getBaseUrl());
    const apiKey = getApiKey();
    if (apiKey) {
        formData.append('exo_api_key', apiKey);
    }
    return formData;
}

function showResponse(elementId, data, error = false) {
    const element = document.getElementById(elementId);
    element.classList.remove('hidden');
    
    if (error) {
        element.style.borderLeft = '4px solid #ef4444';
        element.textContent = '❌ Error:\n\n' + (typeof data === 'string' ? data : JSON.stringify(data, null, 2));
    } else {
        element.style.borderLeft = '4px solid #10b981';
        element.textContent = '✅ Success:\n\n' + JSON.stringify(data, null, 2);
    }
}

function showLoading(button) {
    button.disabled = true;
    button.innerHTML = '<span class="loading"></span> Loading...';
}

function hideLoading(button, originalText) {
    button.disabled = false;
    button.textContent = originalText;
}

async function testNodeId() {
    const button = event.target;
    showLoading(button);
    
    try {
        const formData = createFormData();
        const response = await fetch('/admin/api-tester/node-id', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) {
            showResponse('response-node-id', data.detail || data, true);
        } else {
            showResponse('response-node-id', data);
        }
    } catch (error) {
        showResponse('response-node-id', error.message, true);
    } finally {
        hideLoading(button, 'Test');
    }
}

async function testModels() {
    const button = event.target;
    showLoading(button);
    
    try {
        const formData = createFormData();
        const response = await fetch('/admin/api-tester/models', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) {
            showResponse('response-models', data.detail || data, true);
        } else {
            showResponse('response-models', data);
        }
    } catch (error) {
        showResponse('response-models', error.message, true);
    } finally {
        hideLoading(button, 'Test');
    }
}

async function testState() {
    const button = event.target;
    showLoading(button);
    
    try {
        const formData = createFormData();
        const response = await fetch('/admin/api-tester/state', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) {
            showResponse('response-state', data.detail || data, true);
        } else {
            showResponse('response-state', data);
        }
    } catch (error) {
        showResponse('response-state', error.message, true);
    } finally {
        hideLoading(button, 'Test');
    }
}

async function testEvents() {
    const button = event.target;
    showLoading(button);
    
    try {
        const formData = createFormData();
        const response = await fetch('/admin/api-tester/events', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) {
            showResponse('response-events', data.detail || data, true);
        } else {
            showResponse('response-events', data);
        }
    } catch (error) {
        showResponse('response-events', error.message, true);
    } finally {
        hideLoading(button, 'Test');
    }
}

async function testPlacement() {
    const button = event.target;
    showLoading(button);
    
    try {
        const modelId = document.getElementById('placement-model-id').value;
        const sharding = document.getElementById('placement-sharding').value;
        const instanceMeta = document.getElementById('placement-instance-meta').value;
        const minNodes = document.getElementById('placement-min-nodes').value;
        
        if (!modelId) {
            throw new Error('Model ID is required');
        }
        
        const formData = createFormData();
        formData.append('model_id', modelId);
        formData.append('sharding', sharding);
        formData.append('instance_meta', instanceMeta);
        formData.append('min_nodes', minNodes);
        
        const response = await fetch('/admin/api-tester/placement', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) {
            showResponse('response-placement', data.detail || data, true);
        } else {
            showResponse('response-placement', data);
        }
    } catch (error) {
        showResponse('response-placement', error.message, true);
    } finally {
        hideLoading(button, 'Test Placement');
    }
}

async function testPlacementPreviews() {
    const button = event.target;
    showLoading(button);
    
    try {
        const formData = createFormData();
        const response = await fetch('/admin/api-tester/previews', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) {
            showResponse('response-placement-previews', data.detail || data, true);
        } else {
            showResponse('response-placement-previews', data);
        }
    } catch (error) {
        showResponse('response-placement-previews', error.message, true);
    } finally {
        hideLoading(button, 'Test');
    }
}

async function testCreateInstance() {
    const button = event.target;
    showLoading(button);
    
    try {
        const instanceJson = document.getElementById('create-instance-json').value;
        
        if (!instanceJson) {
            throw new Error('Instance JSON is required');
        }
        
        const formData = createFormData();
        formData.append('instance_json', instanceJson);
        
        const response = await fetch('/admin/api-tester/create-instance', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (!response.ok) {
            showResponse('response-create-instance', data.detail || data, true);
        } else {
            showResponse('response-create-instance', data);
        }
    } catch (error) {
        showResponse('response-create-instance', error.message, true);
    } finally {
        hideLoading(button, 'Create Instance');
    }
}

async function testPlaceInstance() {
    const button = event.target;
    showLoading(button);
    
    try {
        const modelId = document.getElementById('place-model-id').value;
        const sharding = document.getElementById('place-sharding').value;
        const instanceMeta = document.getElementById('place-instance-meta').value;
        const minNodes = parseInt(document.getElementById('place-min-nodes').value);
        
        if (!modelId) {
            throw new Error('Model ID is required');
        }
        
        const formData = createFormData();
        formData.append('model_id', modelId);
        formData.append('sharding', sharding);
        formData.append('instance_meta', instanceMeta);
        formData.append('min_nodes', minNodes.toString());
        
        const response = await fetch('/admin/api-tester/place-instance', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (!response.ok) {
            showResponse('response-place-instance', data.detail || data, true);
        } else {
            showResponse('response-place-instance', data);
        }
    } catch (error) {
        showResponse('response-place-instance', error.message, true);
    } finally {
        hideLoading(button, 'Place Instance');
    }
}

async function testGetInstance() {
    const button = event.target;
    showLoading(button);
    
    try {
        const instanceId = document.getElementById('get-instance-id').value;
        
        if (!instanceId) {
            throw new Error('Instance ID is required');
        }
        
        const formData = createFormData();
        formData.append('instance_id', instanceId);
        
        const response = await fetch('/admin/api-tester/get-instance', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) {
            showResponse('response-get-instance', data.detail || data, true);
        } else {
            showResponse('response-get-instance', data);
        }
    } catch (error) {
        showResponse('response-get-instance', error.message, true);
    } finally {
        hideLoading(button, 'Get Instance');
    }
}

async function testDeleteInstance() {
    const button = event.target;
    showLoading(button);
    
    try {
        const instanceId = document.getElementById('delete-instance-id').value;
        
        if (!instanceId) {
            throw new Error('Instance ID is required');
        }
        
        if (!confirm(`Are you sure you want to delete instance: ${instanceId}?`)) {
            hideLoading(button, 'Delete Instance');
            return;
        }
        
        const formData = createFormData();
        formData.append('instance_id', instanceId);
        
        const response = await fetch('/admin/api-tester/delete-instance', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (!response.ok) {
            showResponse('response-delete-instance', data.detail || data, true);
        } else {
            showResponse('response-delete-instance', data);
        }
    } catch (error) {
        showResponse('response-delete-instance', error.message, true);
    } finally {
        hideLoading(button, 'Delete Instance');
    }
}

async function checkModel() {
    const button = event.target;
    showLoading(button);
    
    try {
        const model = document.getElementById('chat-model').value;
        
        if (!model) {
            throw new Error('Model is required');
        }
        
        // Get state
        const stateFormData = createFormData();
        const stateResponse = await fetch('/admin/api-tester/state', {
            method: 'POST',
            body: stateFormData
        });
        const stateData = await stateResponse.json();
        
        // Get models
        const modelsFormData = createFormData();
        const modelsResponse = await fetch('/admin/api-tester/models', {
            method: 'POST',
            body: modelsFormData
        });
        const modelsData = await modelsResponse.json();
        
        // Build model mapping (short ID -> full ID)
        const modelMapping = [];
        for (const m of modelsData.data || []) {
            const shortId = m.id || '';
            const fullId = m.hugging_face_id || '';
            if (shortId && fullId) {
                modelMapping.push({
                    short_id: shortId,
                    full_id: fullId,
                    name: m.name || ''
                });
            }
        }
        
        // Check if model exists (as short ID or full ID)
        const availableShortIds = modelMapping.map(m => m.short_id);
        const availableFullIds = modelMapping.map(m => m.full_id);
        const modelExists = availableShortIds.includes(model) || availableFullIds.includes(model);
        
        // Find the corresponding full model ID if user provided short ID
        let resolvedFullId = model;
        for (const mapping of modelMapping) {
            if (model === mapping.short_id) {
                resolvedFullId = mapping.full_id;
                break;
            }
        }
        
        // Check if instance exists for this model
        const instances = stateData.instances || {};
        let instanceFound = false;
        let instanceInfo = null;
        
        for (const [instanceId, instanceData] of Object.entries(instances)) {
            const instanceType = Object.keys(instanceData)[0];
            const instance = instanceData[instanceType];
            
            if (instance.shardAssignments) {
                const shardModel = instance.shardAssignments.modelId || '';
                // Match against full ID or if either contains the other
                if (shardModel === resolvedFullId || 
                    model.includes(shardModel) || 
                    shardModel.includes(model) ||
                    resolvedFullId.includes(shardModel)) {
                    instanceFound = true;
                    instanceInfo = {
                        instance_id: instance.instanceId || instanceId,
                        type: instanceType,
                        model_id: shardModel
                    };
                    break;
                }
            }
        }
        
        // Build helpful message
        let message = '';
        if (modelExists && instanceFound) {
            message = `✅ Model is ready for chat completions! Use this model ID: ${instanceInfo.model_id}`;
        } else if (modelExists) {
            const matchingModel = modelMapping.find(m => model === m.short_id || model === m.full_id);
            if (matchingModel) {
                message = `⚠️ Model exists but no instance is placed. Use POST /place_instance with short ID "${matchingModel.short_id}", then use full ID "${matchingModel.full_id}" for chat.`;
            } else {
                message = 'Model exists but no instance is placed - use POST /place_instance';
            }
        } else {
            message = `❌ Model not found. Available models: ${availableShortIds.join(', ')}`;
        }
        
        const ready = modelExists && instanceFound;
        
        // Format a nice response
        let resultText = '🔍 Model Check Results:\n\n';
        resultText += `Input Model: ${model}\n`;
        if (resolvedFullId && resolvedFullId !== model) {
            resultText += `Resolved Full ID: ${resolvedFullId}\n`;
        }
        resultText += `Model Exists: ${modelExists ? '✅ Yes' : '❌ No'}\n`;
        resultText += `Instance Found: ${instanceFound ? '✅ Yes' : '❌ No'}\n`;
        resultText += `Ready for Chat: ${ready ? '✅ Yes' : '❌ No'}\n\n`;
        resultText += `${message}\n\n`;
        
        if (modelMapping.length > 0) {
            resultText += `📋 Available Models (Short ID → Full ID):\n`;
            modelMapping.forEach(m => {
                resultText += `  ${m.short_id} → ${m.full_id}\n`;
            });
            resultText += '\n';
        }
        
        if (instanceInfo) {
            resultText += `📦 Instance Info:\n`;
            resultText += `  Instance ID: ${instanceInfo.instance_id}\n`;
            resultText += `  Type: ${instanceInfo.type}\n`;
            resultText += `  Full Model ID: ${instanceInfo.model_id}\n\n`;
            resultText += `💡 Copy this model ID to the Model field above: ${instanceInfo.model_id}\n`;
        }
        
        const responseBox = document.getElementById('response-check-model');
        responseBox.classList.remove('hidden');
        responseBox.style.borderLeft = ready ? '4px solid #10b981' : '4px solid #f59e0b';
        responseBox.textContent = resultText;
    } catch (error) {
        showResponse('response-check-model', error.message, true);
    } finally {
        hideLoading(button, 'Check Model Readiness');
    }
}

function showCurlExample() {
    const model = document.getElementById('chat-model').value || 'mlx-community/Llama-3.2-1B-Instruct-4bit';
    const messagesJson = document.getElementById('chat-messages').value || '[{"role": "user", "content": "Hello!"}]';
    const temperature = parseFloat(document.getElementById('chat-temperature').value) || 0.7;
    const maxTokens = parseInt(document.getElementById('chat-max-tokens').value) || 100;
    const stream = document.getElementById('chat-stream').checked;
    const baseUrl = getBaseUrl();
    const apiKey = getApiKey();
    
    let authHeader = '';
    if (apiKey) {
        authHeader = `  -H "Authorization: Bearer ${apiKey}" \\\n`;
    }
    
    let curlCommand = '';
    
    if (stream) {
        curlCommand = `# Streaming Mode (Server-Sent Events)
# Direct call to EXO API (no proxy)
curl -N -X POST "${baseUrl}/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
${authHeader}  -d '{
    "model": "${model}",
    "messages": ${messagesJson},
    "temperature": ${temperature},
    "max_tokens": ${maxTokens},
    "stream": true
  }'

# Output format (SSE):
# data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}
# data: {"id":"...","choices":[{"delta":{"content":"!"}}]}
# data: [DONE]`;
    } else {
        curlCommand = `# Non-Streaming Mode (Single JSON Response)
# Direct call to EXO API (no proxy)
curl -X POST "${baseUrl}/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
${authHeader}  -d '{
    "model": "${model}",
    "messages": ${messagesJson},
    "temperature": ${temperature},
    "max_tokens": ${maxTokens},
    "stream": false
  }'

# Output format (JSON):
# {
#   "id": "...",
#   "choices": [{
#     "message": {"role": "assistant", "content": "Hello!"},
#     "finish_reason": "stop"
#   }],
#   "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
# }`;
    }
    
    const responseBox = document.getElementById('response-chat');
    responseBox.classList.remove('hidden');
    responseBox.style.borderLeft = '4px solid #3b82f6';
    responseBox.textContent = '📋 Direct Curl Command (to EXO):\n\n' + curlCommand;
}

// Update button icon when streaming checkbox changes
document.addEventListener('DOMContentLoaded', function() {
    const streamCheckbox = document.getElementById('chat-stream');
    const btnIcon = document.getElementById('btn-icon');
    
    if (streamCheckbox && btnIcon) {
        streamCheckbox.addEventListener('change', function() {
            if (this.checked) {
                btnIcon.textContent = '📡';
            } else {
                btnIcon.textContent = '📤';
            }
        });
    }
});

async function testChatCompletion() {
    const button = event.target;
    showLoading(button);
    
    try {
        const model = document.getElementById('chat-model').value;
        const messagesJson = document.getElementById('chat-messages').value;
        const temperature = parseFloat(document.getElementById('chat-temperature').value);
        const maxTokens = parseInt(document.getElementById('chat-max-tokens').value);
        const stream = document.getElementById('chat-stream').checked;
        
        if (!model) {
            throw new Error('Model is required');
        }
        
        if (!messagesJson) {
            throw new Error('Messages are required');
        }
        
        const messages = JSON.parse(messagesJson);
        
        const formData = createFormData();
        formData.append('model', model);
        formData.append('messages', messagesJson);
        formData.append('temperature', temperature.toString());
        formData.append('max_tokens', maxTokens.toString());
        formData.append('stream', stream.toString());
        
        const response = await fetch('/admin/api-tester/chat', {
            method: 'POST',
            body: formData
        });
        
        if (stream) {
            // Check if we got an error response before streaming starts
            if (!response.ok) {
                const text = await response.text();
                try {
                    const errorData = JSON.parse(text);
                    const errorMsg = errorData.error || errorData.detail || `HTTP ${response.status}: ${text}`;
                    throw new Error(errorMsg);
                } catch (e) {
                    if (e.message) throw e;
                    throw new Error(`HTTP ${response.status}: ${text}`);
                }
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullContent = '';
            let buffer = '';
            let firstChunk = true;
            let streamError = null;
            let currentTTFT = null;
            let currentTPS = null;
            const startTime = performance.now();
            
            const responseBox = document.getElementById('response-chat');
            responseBox.classList.remove('hidden');
            responseBox.style.borderLeft = '4px solid #3b82f6';
            responseBox.textContent = '📡 Streaming...\n\n';
            
            let chunkCount = 0;
            
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                        console.log('Stream ended. Total chunks:', chunkCount, 'Full content length:', fullContent.length);
                        // Process any remaining buffer when stream ends
                        if (buffer.trim()) {
                            const trimmedLine = buffer.trim();
                            if (trimmedLine) {
                                try {
                                    let data = JSON.parse(trimmedLine);
                                    console.log('Processing final buffer chunk:', data);
                                    // Process the last chunk
                                    if (data.message && data.message.content) {
                                        fullContent += data.message.content;
                                    }
                                } catch (e) {
                                    console.debug('Could not parse final buffer:', e);
                                }
                            }
                        }
                        break;
                    }
                    
                    chunkCount++;
                    const decoded = decoder.decode(value, { stream: true });
                    buffer += decoded;
                    
                    // Log first few chunks for debugging
                    if (chunkCount <= 3) {
                        console.log(`Chunk ${chunkCount} (${decoded.length} bytes):`, decoded.substring(0, 200));
                    }
                    
                    // Handle both ndjson format (newline-delimited) and SSE format (data: prefix)
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || ''; // Keep partial line for next chunk
                    
                    for (const line of lines) {
                        const trimmedLine = line.trim();
                        if (!trimmedLine) continue;
                        
                        try {
                            let data;
                            
                            // Check if it's SSE format (starts with "data: ")
                            if (trimmedLine.startsWith('data: ')) {
                                const jsonStr = trimmedLine.substring(6).trim();
                                if (jsonStr === '[DONE]') {
                                    continue;
                                }
                                if (!jsonStr) continue;
                                data = JSON.parse(jsonStr);
                            } else {
                                // Assume ndjson format (direct JSON) - this is what the backend returns
                                data = JSON.parse(trimmedLine);
                            }
                            
                            // Check for errors with better error message formatting
                            if (data.error) {
                                let errorMessage = '';
                                if (typeof data.error === 'string') {
                                    errorMessage = data.error;
                                } else if (data.error.message) {
                                    errorMessage = data.error.message;
                                } else {
                                    errorMessage = JSON.stringify(data.error);
                                }
                                
                                // Add error type and details if available
                                let fullErrorMessage = `❌ Error: ${errorMessage}`;
                                if (data.type) {
                                    fullErrorMessage += `\n\nType: ${data.type}`;
                                }
                                if (data.details) {
                                    fullErrorMessage += `\n\nDetails: ${data.details}`;
                                }
                                if (data.status_code) {
                                    fullErrorMessage += `\n\nHTTP Status: ${data.status_code}`;
                                }
                                
                                // Add troubleshooting tips based on error type
                                if (data.type === 'connection_error') {
                                    fullErrorMessage += '\n\n💡 Troubleshooting:\n';
                                    fullErrorMessage += '  • Check if EXO instance is running\n';
                                    fullErrorMessage += '  • Verify the EXO Base URL is correct\n';
                                    fullErrorMessage += '  • Check network connectivity\n';
                                    fullErrorMessage += '  • Ensure firewall allows connections';
                                } else if (data.type === 'timeout_error') {
                                    fullErrorMessage += '\n\n💡 Troubleshooting:\n';
                                    fullErrorMessage += '  • The request took too long (>600s)\n';
                                    fullErrorMessage += '  • Try reducing max_tokens\n';
                                    fullErrorMessage += '  • Check EXO instance performance\n';
                                    fullErrorMessage += '  • Verify model is loaded and ready';
                                } else if (data.type === 'api_error') {
                                    fullErrorMessage += '\n\n💡 Troubleshooting:\n';
                                    fullErrorMessage += '  • Check EXO API logs for details\n';
                                    fullErrorMessage += '  • Verify model name is correct\n';
                                    fullErrorMessage += '  • Ensure model instance is placed\n';
                                    fullErrorMessage += '  • Check EXO instance status';
                                }
                                
                                streamError = new Error(errorMessage);
                                responseBox.style.borderLeft = '4px solid #ef4444';
                                responseBox.textContent = fullErrorMessage;
                                throw streamError;
                            }
                            
                            // Handle OpenAI SSE format (choices[0].delta.content) - direct from EXO
                            if (data.choices && data.choices[0] && data.choices[0].delta && data.choices[0].delta.content) {
                                const content = data.choices[0].delta.content;
                                if (content) {
                                    if (firstChunk) {
                                        firstChunk = false;
                                        const ttft = Math.round(performance.now() - startTime);
                                        currentTTFT = ttft;
                                        fullContent = '';
                                    }
                                    fullContent += content;
                                    
                                    // Update display in real-time
                                    let displayText = '📡 Streaming (SSE Format):\n\n';
                                    displayText += fullContent;
                                    if (currentTTFT) {
                                        displayText += `\n\n---\n\n📊 TTFT: ${currentTTFT}ms`;
                                    }
                                    responseBox.textContent = displayText;
                                    responseBox.scrollTop = responseBox.scrollHeight;
                                }
                                
                                // Check for finish_reason (end of stream)
                                if (data.choices[0].finish_reason) {
                                    responseBox.style.borderLeft = '4px solid #10b981';
                                    let finalText = '✅ Stream Complete (SSE Format):\n\n';
                                    finalText += '📝 Generated Content:\n';
                                    finalText += fullContent || '(no content)';
                                    finalText += '\n\n---\n\n';
                                    finalText += '📊 Statistics:\n';
                                    if (currentTTFT) {
                                        finalText += `  Time to First Token (TTFT): ${currentTTFT}ms\n`;
                                    }
                                    if (data.usage) {
                                        finalText += `  Prompt Tokens: ${data.usage.prompt_tokens || 0}\n`;
                                        finalText += `  Completion Tokens: ${data.usage.completion_tokens || 0}\n`;
                                        finalText += `  Total Tokens: ${data.usage.total_tokens || 0}\n`;
                                    }
                                    finalText += `  Finish Reason: ${data.choices[0].finish_reason}\n`;
                                    finalText += '\n---\n\n';
                                    finalText += '💡 Note: With stream=true, you receive chunks in real-time via Server-Sent Events (SSE) format.';
                                    
                                    responseBox.textContent = finalText;
                                }
                            }
                            // Handle playground ndjson format (message.content) - from backend conversion
                            else if (data.message && data.message.content) {
                                if (firstChunk) {
                                    firstChunk = false;
                                    const ttft = Math.round(performance.now() - startTime);
                                    currentTTFT = ttft;
                                    fullContent = '';
                                    console.log('First chunk received, TTFT:', ttft, 'ms');
                                }
                                fullContent += data.message.content;
                                
                                // Update display in real-time
                                let displayText = '📡 Streaming:\n\n';
                                displayText += fullContent;
                                if (currentTTFT) {
                                    displayText += `\n\n---\n\n📊 TTFT: ${currentTTFT}ms`;
                                }
                                responseBox.textContent = displayText;
                                responseBox.scrollTop = responseBox.scrollHeight;
                                
                                // Log progress every 10 chunks
                                if (chunkCount % 10 === 0) {
                                    console.log(`Streaming progress: ${fullContent.length} characters received`);
                                }
                            }
                            
                            // Handle final chunk with stats (playground format)
                            if (data.done === true && data.hasOwnProperty('eval_count') && data.hasOwnProperty('eval_duration')) {
                                const tokensPerSecond = data.eval_duration > 0 ? (data.eval_count / (data.eval_duration / 1e9)) : 0;
                                currentTPS = tokensPerSecond;
                                
                                responseBox.style.borderLeft = '4px solid #10b981';
                                let finalText = '✅ Stream Complete:\n\n';
                                finalText += '📝 Generated Content:\n';
                                finalText += fullContent || '(no content)';
                                finalText += '\n\n---\n\n';
                                finalText += '📊 Statistics:\n';
                                if (currentTTFT) {
                                    finalText += `  Time to First Token (TTFT): ${currentTTFT}ms\n`;
                                }
                                finalText += `  Tokens Generated: ${data.eval_count || 'N/A'}\n`;
                                finalText += `  Duration: ${(data.eval_duration / 1e9).toFixed(2)}s\n`;
                                if (currentTPS) {
                                    finalText += `  Speed: ${currentTPS.toFixed(2)} tokens/second\n`;
                                }
                                finalText += '\n---\n\n';
                                finalText += '💡 Note: With stream=true, you receive chunks in real-time using newline-delimited JSON format.';
                                
                                responseBox.textContent = finalText;
                            }
                        } catch (err) {
                            if (err.message && err.message !== 'Streaming error' && !streamError) {
                                // Only log parse errors if we haven't already hit a real error
                                // This is normal for partial chunks - they'll be buffered
                                console.debug('JSON parsing error in stream (likely partial chunk), buffering...', trimmedLine.substring(0, 100));
                            } else if (streamError) {
                                // Re-throw if we already have a stream error
                                throw err;
                            }
                        }
                    }
                }
            } catch (error) {
                if (error.name !== 'AbortError' && !streamError) {
                    streamError = error;
                    responseBox.style.borderLeft = '4px solid #ef4444';
                    let errorText = '❌ Streaming Error:\n\n';
                    errorText += error.message || 'Unknown error occurred';
                    errorText += '\n\n💡 Troubleshooting:\n';
                    errorText += '  • Check browser console for details\n';
                    errorText += '  • Verify EXO instance is accessible\n';
                    errorText += '  • Try the request again';
                    responseBox.textContent = errorText;
                }
            } finally {
                // If stream ended without final stats, show what we have
                if (!streamError && fullContent && !responseBox.textContent.includes('Stream Complete')) {
                    responseBox.style.borderLeft = '4px solid #10b981';
                    let finalText = '✅ Stream Complete:\n\n';
                    finalText += '📝 Generated Content:\n';
                    finalText += fullContent || '(no content)';
                    finalText += '\n\n---\n\n';
                    finalText += '📊 Statistics:\n';
                    if (currentTTFT) {
                        finalText += `  Time to First Token (TTFT): ${currentTTFT}ms\n`;
                    }
                    responseBox.textContent = finalText;
                }
                
                if (buffer.trim() && !streamError) {
                    // Handle any remaining buffer
                    try {
                        let jsonStr = buffer.trim();
                        if (jsonStr.startsWith('data: ')) {
                            jsonStr = jsonStr.substring(6).trim();
                        }
                        if (jsonStr && jsonStr !== '[DONE]') {
                            const data = JSON.parse(jsonStr);
                            if (data.error) {
                                let errorMessage = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
                                let fullErrorMessage = `❌ Error: ${errorMessage}`;
                                if (data.details) {
                                    fullErrorMessage += `\n\nDetails: ${data.details}`;
                                }
                                responseBox.style.borderLeft = '4px solid #ef4444';
                                responseBox.textContent = fullErrorMessage;
                            } else if (data.message && data.message.content) {
                                // Process any remaining content
                                fullContent += data.message.content;
                                if (!responseBox.textContent.includes('Stream Complete')) {
                                    let displayText = '📡 Streaming:\n\n';
                                    displayText += fullContent;
                                    if (currentTTFT) {
                                        displayText += `\n\n---\n\n📊 TTFT: ${currentTTFT}ms`;
                                    }
                                    responseBox.textContent = displayText;
                                }
                            }
                        }
                    } catch (e) {
                        // Ignore parse errors for remaining buffer
                        console.debug('Could not parse remaining buffer:', e);
                    }
                }
            }
        } else {
            // Non-streaming mode - expect a single JSON response
            const contentType = response.headers.get('content-type') || '';
            
            if (!response.ok) {
                const text = await response.text();
                try {
                    const errorData = JSON.parse(text);
                    const errorMsg = errorData.error || errorData.detail || `HTTP ${response.status}: ${text}`;
                    showResponse('response-chat', errorMsg, true);
                    return;
                } catch (e) {
                    throw new Error(`HTTP ${response.status}: ${text || 'Empty response'}`);
                }
            }
            
            // Check content type
            if (!contentType.includes('application/json')) {
                const text = await response.text();
                throw new Error(`Unexpected response type: ${contentType}. Response: ${text}`);
            }
            
            const text = await response.text();
            if (!text) {
                throw new Error('Empty response from server');
            }
            
            let data;
            try {
                data = JSON.parse(text);
            } catch (e) {
                throw new Error(`Invalid JSON response: ${e.message}\n\nResponse: ${text.substring(0, 500)}`);
            }
            
            // Format a nice non-streaming response
            const responseBox = document.getElementById('response-chat');
            responseBox.classList.remove('hidden');
            responseBox.style.borderLeft = '4px solid #10b981';
            
            let displayText = '✅ Non-Streaming Response :\n\n';
            
            // Show full JSON
            displayText += 'JSON Response:\n';
            displayText += JSON.stringify(data, null, 2);
            
            responseBox.textContent = displayText;
        }
    } catch (error) {
        showResponse('response-chat', error.message, true);
    } finally {
        hideLoading(button, 'Send Chat Request');
    }
}

