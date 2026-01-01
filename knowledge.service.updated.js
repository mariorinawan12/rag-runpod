/**
 * Knowledge Service - UPDATED for Dual-Query Architecture
 */

const axios = require('axios');
const agentService = require('./agent.service');
const chatService = require('./chat.service');
const { Pool } = require('pg');
const dbConfig = require('../config/database.config');

const pool = new Pool(dbConfig);
const PYTHON_ASK_URL = process.env.PYTHON_ENGINE_URL || 'http://localhost:5000/ask';
const PYTHON_STREAM_URL = process.env.PYTHON_ENGINE_URL
    ? process.env.PYTHON_ENGINE_URL.replace('/ask', '/ask/stream')
    : 'http://localhost:5000/ask/stream';

// NEW: Query Process endpoint
const PYTHON_QUERY_PROCESS_URL = process.env.PYTHON_ENGINE_URL
    ? process.env.PYTHON_ENGINE_URL.replace('/ask', '/query/process')
    : 'http://localhost:5000/query/process';


// ============================================================================
// GET DOC IDS FOR AGENT
// ============================================================================
async function getDocIdsForAgent(agentId) {
    const result = await pool.query(
        `SELECT document_id FROM documents WHERE agent_id = $1 AND status IN ('ready', 'completed')`,
        [agentId]
    );
    return result.rows.map(r => r.document_id);
}


// ============================================================================
// BUILD CONVERSATION HISTORY - UPDATED: Returns array instead of string
// ============================================================================
async function buildConversationHistory(sessionId, maxPairs = 2, maxContentLength = null) {
    console.log(`   📨 [getSessionMessages] sessionId=${sessionId}, limit=${maxPairs * 2}`);

    const result = await chatService.getSessionMessages(sessionId, maxPairs * 2);
    const messages = result?.messages || [];

    console.log(`   📨 [getSessionMessages] Got ${messages.length} messages`);

    if (messages.length === 0) {
        return [];
    }

    // Return as array of {role, content}
    const history = [];
    for (const msg of messages) {
        let content = msg.message_text || '';

        // Truncate content if maxContentLength specified
        if (maxContentLength && content.length > maxContentLength) {
            content = content.slice(0, maxContentLength) + '... [truncated]';
        }

        history.push({
            role: msg.sender === 'user' ? 'user' : 'assistant',
            content: content
        });
    }

    return history;
}


// ============================================================================
// NEW: Process Query (Classify + Rewrite)
// ============================================================================
async function processQuery(currentQuery, conversationHistory) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`🔍 [QUERY PROCESS] START`);
    console.log(`${'='.repeat(60)}`);

    const requestBody = {
        current_query: currentQuery,
        conversation_history: conversationHistory
    };

    console.log(`📤 [QUERY PROCESS] Request Body:`);
    console.log(JSON.stringify(requestBody, null, 2));

    try {
        const response = await axios.post(PYTHON_QUERY_PROCESS_URL, requestBody, {
            timeout: 30000
        });

        const result = response.data;
        console.log(`\n📥 [QUERY PROCESS] Response:`);
        console.log(JSON.stringify(result, null, 2));
        console.log(`${'='.repeat(60)}\n`);

        return result;

    } catch (error) {
        console.error(`❌ [QUERY PROCESS] Error:`, error.message);
        // Fallback: return original query as-is
        return {
            is_ambiguous: false,
            is_transform: false,
            answer_source: 'document',
            retrieval_query: currentQuery,
            original_query: currentQuery
        };
    }
}


// ============================================================================
// NEW: Decide which query to use for generation
// ============================================================================
function decideGenerationQuery(queryResult) {
    const { is_ambiguous, is_transform, retrieval_query, original_query } = queryResult;

    if (is_transform) {
        // Transform case: preserve original intent (rangkumin, bikin tabel, etc)
        return original_query;
    } else if (is_ambiguous) {
        // Ambiguous non-transform: use rewritten (original is incomplete)
        return retrieval_query;
    } else {
        // Normal case: use original
        return original_query;
    }
}


// ============================================================================
// EXISTING: GET RESPONSE (NON-STREAMING) - UPDATED
// ============================================================================
async function getResponse(userQuery, agentId, sessionId) {
    const startTime = Date.now();

    const agentName = await agentService.getAgentName(agentId);
    if (!agentName) {
        throw new Error(`Agent dengan ID ${agentId} tidak ditemukan.`);
    }

    const docIds = await getDocIdsForAgent(agentId);
    const pipeline = docIds.length > 0 ? "rag" : "chat";

    // NEW: Get conversation history as array
    const conversationHistory = await buildConversationHistory(sessionId, 2);

    // NEW: Process query (classify + rewrite)
    const queryResult = await processQuery(userQuery, conversationHistory);
    const generationQuery = decideGenerationQuery(queryResult);

    const requestBody = {
        pipeline: pipeline,
        original_query: generationQuery,
        retrieval_query: queryResult.retrieval_query,
        doc_ids: pipeline === "rag" ? docIds : null,
        conversation_history: conversationHistory
    };

    try {
        const pyRes = await axios.post(PYTHON_ASK_URL, requestBody, {
            timeout: 120000
        });

        const result = pyRes.data;

        // Enrich sources with document names
        let sourcesWithNames = [];
        if (pipeline === "rag" && result.sources && result.sources.length > 0) {
            for (const source of result.sources) {
                const docId = source.document_id;
                let docName = null;
                try {
                    const names = await agentService.getDocumentNames([docId]);
                    docName = names[0] || null;
                } catch (e) { }

                sourcesWithNames.push({
                    citations: source.citations || {}
                });
            }
        }

        return {
            response: result.answer,
            token_usage: result.token_usage,
            sources: sourcesWithNames,
            agent_name: agentName,
            pipeline: pipeline,
            metadata: {
                retrieval_confidence: result.confidence,
                chunks_retrieved: result.stats?.chunks_included || 0,
                total_time_ms: Date.now() - startTime,
                // NEW: Include query processing info
                is_ambiguous: queryResult.is_ambiguous,
                is_transform: queryResult.is_transform
            }
        };

    } catch (error) {
        console.error(`❌ [PYTHON ERROR]`, error.response?.data || error.message);
        throw new Error(`Engine error: ${error.message}`);
    }
}


// ============================================================================
// GET RESPONSE STREAM - UPDATED for Dual Query
// ============================================================================
async function getResponseStream(userQuery, documentIds, sessionId, onChunk) {
    const startTime = Date.now();

    const pipeline = documentIds.length > 0 ? "rag" : "chat";
    console.log(`🌊 [STREAM] Pipeline: ${pipeline}, Docs: ${documentIds.length}`);

    // Step 1: Get conversation history as array (FULL, no truncation)
    const conversationHistory = await buildConversationHistory(sessionId, 2);

    // Step 2: Process query (classify + rewrite)
    const queryResult = await processQuery(userQuery, conversationHistory);
    const generationQuery = decideGenerationQuery(queryResult);

    console.log(`🌊 [STREAM] Query processed:`, {
        is_ambiguous: queryResult.is_ambiguous,
        is_transform: queryResult.is_transform,
        answer_source: queryResult.answer_source,
        generation_query: generationQuery.slice(0, 50) + '...',
        retrieval_query: queryResult.retrieval_query.slice(0, 50) + '...'
    });

    // Step 3: Build request with dual queries
    const requestBody = {
        pipeline: pipeline,
        original_query: generationQuery,
        retrieval_query: queryResult.retrieval_query,
        doc_ids: pipeline === "rag" ? documentIds : null,
        conversation_history: conversationHistory,
        query_metadata: {
            is_ambiguous: queryResult.is_ambiguous,
            is_transform: queryResult.is_transform,
            answer_source: queryResult.answer_source || 'document'
        }
    };

    console.log(`\n${'='.repeat(60)}`);
    console.log(`🌊 [ASK STREAM] REQUEST BODY`);
    console.log(`${'='.repeat(60)}`);
    console.log(JSON.stringify(requestBody, null, 2));
    console.log(`${'='.repeat(60)}\n`);

    console.log(`🌊 [STREAM] Calling Python /ask/stream...`);

    // Step 4: Stream request to Python
    const response = await axios({
        method: 'post',
        url: PYTHON_STREAM_URL,
        data: requestBody,
        responseType: 'stream',
        timeout: 120000
    });

    // Step 5: Process stream
    return new Promise((resolve, reject) => {
        let buffer = '';
        let processing = false;
        const queue = [];

        const processQueue = async () => {
            if (processing) return;
            processing = true;
            while (queue.length > 0) {
                const line = queue.shift();
                if (!line.trim()) continue;
                try {
                    const parsed = JSON.parse(line);
                    if (parsed.type === 'sources' && parsed.data) {
                        const sourcesWithNames = [];
                        const docIds = [...new Set(parsed.data.map(s => s.doc_id).filter(Boolean))];
                        let docNameMap = {};
                        try {
                            const names = await agentService.getDocumentNames(docIds);
                            docIds.forEach((id, idx) => {
                                const nameData = names[idx];
                                if (typeof nameData === 'object' && nameData !== null) {
                                    docNameMap[id] = nameData.document_name || null;
                                } else {
                                    docNameMap[id] = nameData || null;
                                }
                            });
                        } catch (e) {
                            console.error('Failed to get doc names:', e);
                        }
                        for (const source of parsed.data) {
                            sourcesWithNames.push({
                                citation_number: source.citation_number,
                                chunk_id: source.chunk_id,
                                document_id: source.doc_id,
                                document_name: docNameMap[source.doc_id] || null,
                                bboxes: source.bboxes,
                                score: source.score,
                                section: source.section,
                                text_preview: source.text_preview
                            });
                        }
                        parsed.sourcesWithNames = sourcesWithNames;
                        console.log('📍 [STREAM] Sources:', sourcesWithNames.length, 'citations');
                    }
                    await onChunk(parsed);
                } catch (e) {
                    console.error('🌊 [STREAM] Parse error:', e.message, 'Line:', line);
                }
            }
            processing = false;
        };

        response.data.on('data', (chunk) => {
            buffer += chunk.toString();
            const lines = buffer.split('\n');
            buffer = lines.pop();
            queue.push(...lines);
            processQueue();
        });

        response.data.on('end', async () => {
            while (queue.length > 0 || processing) {
                await processQueue();
                if (processing) {
                    await new Promise(r => setTimeout(r, 10));
                }
            }
            console.log(`🌊 [STREAM] Completed in ${Date.now() - startTime}ms`);
            resolve();
        });

        response.data.on('error', (err) => {
            console.error('🌊 [STREAM] Error:', err.message);
            reject(err);
        });
    });
}


// ============================================================================
// EXPORTS
// ============================================================================
module.exports = {
    getResponse,
    getResponseStream,
    getDocIdsForAgent,
    buildConversationHistory,  // Renamed from buildConversationContext
    processQuery               // NEW
};
