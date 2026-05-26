// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remote runtime tests for the NeMo Guardrails plugin component.
#![allow(clippy::await_holding_lock)]

use super::*;

#[tokio::test]
async fn remote_initialization_installs_non_streaming_execution_intercept() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-execution-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-remote",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "guarded"},
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "state": {"state": {"conversation": "server-state"}},
            "output_data": {"decision": "allow"}
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default",
            "headers": {"x-guardrails-auth": "token"},
            "timeout_millis": 5_000
        },
        "request_defaults": {
            "context": {"tenant": "acme"},
            "thread_id": "thread-1234567890",
            "state": {"state": {"conversation": "client-state"}},
            "rails": {"input": true, "retrieval": ["kb"]},
            "llm_params": {"temperature": 0.1},
            "llm_output": true,
            "output_vars": ["answer"],
            "log": {"activated_rails": true}
        }
    })))
    .await
    .unwrap();

    let original_called = Arc::new(AtomicBool::new(false));
    let called = Arc::clone(&original_called);
    let func: LlmExecutionNextFn = Arc::new(move |_req| {
        called.store(true, Ordering::SeqCst);
        Box::pin(async move { Ok(json!({"response": "original"})) })
    });

    let response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap();

    assert!(!original_called.load(Ordering::SeqCst));
    assert_eq!(response["id"], json!("chatcmpl-remote"));
    assert_eq!(response["object"], json!("chat.completion"));
    assert_eq!(response["model"], json!("gpt-4o-mini"));
    assert_eq!(
        response["choices"][0]["message"]["content"],
        json!("guarded")
    );
    assert_eq!(
        response["guardrails"]["output_data"]["decision"],
        json!("allow")
    );
    assert_eq!(
        response["guardrails"]["state"]["state"]["conversation"],
        json!("server-state")
    );

    let captured = recv_captured_request(&request_rx);
    assert_eq!(captured.path, "/v1/chat/completions");
    assert!(captured.content_type.starts_with("application/json"));

    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(request_json["messages"][0]["content"], json!("hello"));
    assert_eq!(request_json["stream"], json!(false));
    assert_eq!(
        request_json["guardrails"]["config_id"],
        json!("safety-default")
    );
    assert_eq!(
        request_json["guardrails"]["context"]["tenant"],
        json!("acme")
    );
    assert_eq!(
        request_json["guardrails"]["thread_id"],
        json!("thread-1234567890")
    );
    assert_eq!(
        request_json["guardrails"]["state"]["state"]["conversation"],
        json!("client-state")
    );
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["retrieval"],
        json!(["kb"])
    );
    assert_eq!(
        request_json["guardrails"]["options"]["llm_output"],
        json!(true)
    );

    let captured_events = events.lock().unwrap().clone();
    let mark_names: Vec<_> = captured_events
        .iter()
        .filter(|event| event.kind() == "mark")
        .map(|event| event.name().to_string())
        .collect();
    assert!(mark_names.contains(&"nemo_guardrails.remote.start".to_string()));
    assert!(mark_names.contains(&"nemo_guardrails.remote.end".to_string()));

    let start_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.start")
        .unwrap();
    assert_eq!(
        start_mark.data().unwrap()["config_id"],
        json!("safety-default")
    );
    assert_eq!(start_mark.data().unwrap()["stream"], json!(false));

    let end_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.end")
        .unwrap();
    assert_eq!(end_mark.data().unwrap()["http_status"], json!(200));
    assert_eq!(end_mark.data().unwrap()["stream"], json!(false));

    deregister_subscriber("nemo-guardrails-remote-execution-events").unwrap();
}

#[tokio::test]
async fn remote_request_uses_config_ids_when_config_id_is_not_set() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-remote",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "guarded"},
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_ids": ["safety-a", "safety-b"]
        }
    })))
    .await
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(move |_req| Box::pin(async move { Ok(json!({"response": "original"})) }));

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap();

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(
        request_json["guardrails"]["config_ids"],
        json!(["safety-a", "safety-b"])
    );
    assert!(request_json["guardrails"].get("config_id").is_none());
}

#[tokio::test]
async fn remote_llm_request_disables_input_rails_when_surface_is_off() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-remote",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "guarded"},
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "input": false,
        "output": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        },
        "request_defaults": {
            "rails": {
                "input": ["self check input"],
                "output": ["self check output"],
                "retrieval": ["kb"]
            }
        }
    })))
    .await
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(move |_req| Box::pin(async move { Ok(json!({"response": "original"})) }));

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap();

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["input"],
        json!(false)
    );
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["output"],
        json!(["self check output"])
    );
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["retrieval"],
        json!(["kb"])
    );
}

#[tokio::test]
async fn remote_llm_request_disables_output_rails_when_surface_is_off() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-remote",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "guarded"},
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "input": true,
        "output": false,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        },
        "request_defaults": {
            "rails": {
                "input": ["self check input"],
                "output": ["self check output"],
                "dialog": true
            }
        }
    })))
    .await
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(move |_req| Box::pin(async move { Ok(json!({"response": "original"})) }));

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap();

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["input"],
        json!(["self check input"])
    );
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["output"],
        json!(false)
    );
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["dialog"],
        json!(true)
    );
}

#[tokio::test]
async fn remote_initialization_installs_stream_execution_intercept() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-stream-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-remote\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"guard\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-remote\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ed\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
        sse_body.len(),
        sse_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let original_called = Arc::new(AtomicBool::new(false));
    let called = Arc::clone(&original_called);
    let func: LlmStreamExecutionNextFn = Arc::new(move |_req| {
        called.store(true, Ordering::SeqCst);
        Box::pin(async move {
            let stream = tokio_stream::iter(vec![Ok(json!({"chunk": "original"}))]);
            Ok(Box::pin(stream) as LlmJsonStream)
        })
    });

    let streaming_codec = OpenAIChatStreamingCodec::new();
    let collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(true))
            .func(func)
            .collector(collector)
            .finalizer(finalizer)
            .attributes(LlmAttributes::STREAMING)
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let mut chunks = Vec::new();
    while let Some(chunk) = tokio::time::timeout(TEST_TIMEOUT, stream.next())
        .await
        .expect("timed out waiting for remote stream chunk")
    {
        chunks.push(chunk.unwrap());
    }

    assert!(!original_called.load(Ordering::SeqCst));
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0]["choices"][0]["delta"]["content"], json!("guard"));
    assert_eq!(chunks[1]["choices"][0]["delta"]["content"], json!("ed"));

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(request_json["stream"], json!(true));
    assert_eq!(
        request_json["guardrails"]["config_id"],
        json!("safety-default")
    );

    let captured_events = events.lock().unwrap().clone();
    let start_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.start")
        .unwrap();
    assert_eq!(start_mark.data().unwrap()["stream"], json!(true));

    let end_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.end")
        .unwrap();
    assert_eq!(end_mark.data().unwrap()["http_status"], json!(200));
    assert_eq!(end_mark.data().unwrap()["stream"], json!(true));

    deregister_subscriber("nemo-guardrails-remote-stream-events").unwrap();
}

#[tokio::test]
async fn remote_non_streaming_http_errors_are_reported_and_marked() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-error-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = r#"{"error":"backend unavailable"}"#;
    let http_response = format!(
        "HTTP/1.1 502 Bad Gateway\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let original_called = Arc::new(AtomicBool::new(false));
    let called = Arc::clone(&original_called);
    let func: LlmExecutionNextFn = Arc::new(move |_req| {
        called.store(true, Ordering::SeqCst);
        Box::pin(async move { Ok(json!({"response": "original"})) })
    });

    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap_err();

    assert!(!original_called.load(Ordering::SeqCst));
    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("status 502"));
            assert!(message.contains("backend unavailable"));
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured_events = events.lock().unwrap().clone();
    assert!(
        captured_events
            .iter()
            .any(|event| event.name() == "nemo_guardrails.remote.start")
    );
    let error_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.error")
        .unwrap();
    assert_eq!(error_mark.data().unwrap()["http_status"], json!(502));
    assert_eq!(error_mark.data().unwrap()["stream"], json!(false));
    assert!(
        error_mark.data().unwrap()["error"]
            .as_str()
            .unwrap()
            .contains("error body omitted from marks")
    );

    deregister_subscriber("nemo-guardrails-remote-error-events").unwrap();
}

#[tokio::test]
async fn remote_streaming_http_errors_are_reported_and_marked() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-stream-error-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = r#"{"error":"stream backend unavailable"}"#;
    let http_response = format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let original_called = Arc::new(AtomicBool::new(false));
    let called = Arc::clone(&original_called);
    let func: LlmStreamExecutionNextFn = Arc::new(move |_req| {
        called.store(true, Ordering::SeqCst);
        Box::pin(async move {
            let stream = tokio_stream::iter(vec![Ok(json!({"chunk": "original"}))]);
            Ok(Box::pin(stream) as LlmJsonStream)
        })
    });

    let streaming_codec = OpenAIChatStreamingCodec::new();
    let collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let error = match llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(true))
            .func(func)
            .collector(collector)
            .finalizer(finalizer)
            .attributes(LlmAttributes::STREAMING)
            .response_codec(response_codec)
            .build(),
    )
    .await
    {
        Ok(_) => panic!("expected remote streaming request to fail"),
        Err(error) => error,
    };

    assert!(!original_called.load(Ordering::SeqCst));
    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("status 503"));
            assert!(message.contains("stream backend unavailable"));
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured_events = events.lock().unwrap().clone();
    assert!(
        captured_events
            .iter()
            .any(|event| event.name() == "nemo_guardrails.remote.start")
    );
    let error_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.error")
        .unwrap();
    assert_eq!(error_mark.data().unwrap()["http_status"], json!(503));
    assert_eq!(error_mark.data().unwrap()["stream"], json!(true));
    assert!(
        error_mark.data().unwrap()["error"]
            .as_str()
            .unwrap()
            .contains("error body omitted from marks")
    );

    deregister_subscriber("nemo-guardrails-remote-stream-error-events").unwrap();
}

#[tokio::test]
async fn remote_non_streaming_invalid_json_is_reported_and_marked() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-invalid-json-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = "{not-json}";
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(move |_req| Box::pin(async move { Ok(json!({"response": "original"})) }));

    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("failed to parse remote response JSON"));
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured_events = events.lock().unwrap().clone();
    let error_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.error")
        .unwrap();
    assert_eq!(error_mark.data().unwrap()["http_status"], json!(200));
    assert_eq!(error_mark.data().unwrap()["stream"], json!(false));

    deregister_subscriber("nemo-guardrails-remote-invalid-json-events").unwrap();
}

#[tokio::test]
async fn remote_streaming_malformed_chunk_is_reported_and_marked() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-malformed-stream-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let sse_body = "data: {not-json}\n\n";
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
        sse_body.len(),
        sse_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let func: LlmStreamExecutionNextFn = Arc::new(move |_req| {
        Box::pin(async move {
            let stream = tokio_stream::iter(vec![Ok(json!({"chunk": "original"}))]);
            Ok(Box::pin(stream) as LlmJsonStream)
        })
    });

    let streaming_codec = OpenAIChatStreamingCodec::new();
    let collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(true))
            .func(func)
            .collector(collector)
            .finalizer(finalizer)
            .attributes(LlmAttributes::STREAMING)
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let error = tokio::time::timeout(TEST_TIMEOUT, stream.next())
        .await
        .expect("timed out waiting for remote stream error")
        .unwrap()
        .unwrap_err();
    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(!message.is_empty());
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured_events = events.lock().unwrap().clone();
    let error_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.error")
        .unwrap();
    assert_eq!(error_mark.data().unwrap()["http_status"], json!(200));
    assert_eq!(error_mark.data().unwrap()["stream"], json!(true));

    deregister_subscriber("nemo-guardrails-remote-malformed-stream-events").unwrap();
}

#[tokio::test]
async fn remote_preflight_tool_choice_failure_is_reported_and_marked() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-preflight-error-events");

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": unused_local_endpoint(),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(move |_req| Box::pin(async move { Ok(json!({"response": "original"})) }));
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup data",
                    "parameters": {"type": "object"}
                }
            }]
        }),
    };

    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(request)
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("does not support OpenAI tool definitions or tool_choice"));
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured_events = events.lock().unwrap().clone();
    assert!(
        captured_events
            .iter()
            .any(|event| event.name() == "nemo_guardrails.remote.start")
    );
    let error_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.error")
        .unwrap();
    assert_eq!(error_mark.data().unwrap()["stream"], json!(false));
    assert!(
        error_mark.data().unwrap()["error"]
            .as_str()
            .unwrap()
            .contains("does not support OpenAI tool definitions or tool_choice")
    );

    deregister_subscriber("nemo-guardrails-remote-preflight-error-events").unwrap();
}

#[tokio::test]
async fn remote_transport_failure_is_reported_and_marked() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-transport-error-events");

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": unused_local_endpoint(),
            "config_id": "safety-default",
            "timeout_millis": 50
        }
    })))
    .await
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(move |_req| Box::pin(async move { Ok(json!({"response": "original"})) }));

    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("remote request failed"));
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured_events = events.lock().unwrap().clone();
    let error_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.error")
        .unwrap();
    assert_eq!(error_mark.data().unwrap()["stream"], json!(false));
    assert!(error_mark.data().unwrap().get("http_status").is_none());

    deregister_subscriber("nemo-guardrails-remote-transport-error-events").unwrap();
}

#[tokio::test]
async fn remote_success_without_guardrails_payload_is_allowed() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-remote",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "guarded"},
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(move |_req| Box::pin(async move { Ok(json!({"response": "original"})) }));

    let response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(response["id"], json!("chatcmpl-remote"));
    assert!(response.get("guardrails").is_none());
}

#[tokio::test]
async fn remote_tool_input_block_rejects_before_tool_execution() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-tool-input-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-input-blocked",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "blocked"},
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": [{
                    "name": "tool_input_block",
                    "stop": true
                }]
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let original_called = Arc::new(AtomicBool::new(false));
    let called = Arc::clone(&original_called);
    let error = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                called.store(true, Ordering::SeqCst);
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap_err();

    assert!(!original_called.load(Ordering::SeqCst));
    match error {
        crate::error::FlowError::GuardrailRejected(message) => {
            assert!(message.contains("tool_input"));
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["tool_input"],
        json!(true)
    );
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["tool_output"],
        json!(false)
    );

    let captured_events = events.lock().unwrap().clone();
    let start_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.start")
        .unwrap();
    assert_eq!(start_mark.data().unwrap()["surface"], json!("tool_input"));
    assert_eq!(
        start_mark.data().unwrap()["tool_name"],
        json!("weather_lookup")
    );
    let end_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.end")
        .unwrap();
    assert_eq!(end_mark.data().unwrap()["surface"], json!("tool_input"));

    deregister_subscriber("nemo-guardrails-remote-tool-input-events").unwrap();
}

#[tokio::test]
async fn remote_tool_input_can_rewrite_tool_arguments() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-input-modified",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Boston\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let seen_args = Arc::new(Mutex::new(None::<Json>));
    let seen_args_for_call = Arc::clone(&seen_args);
    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |args| {
                *seen_args_for_call.lock().unwrap() = Some(args.clone());
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result, json!({"forecast": "sunny"}));
    assert_eq!(*seen_args.lock().unwrap(), Some(json!({"city": "Boston"})));

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(request_json["messages"][0]["role"], json!("user"));
}

#[tokio::test]
async fn remote_tool_output_can_rewrite_tool_result() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-output-modified",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Phoenix\"},\"result\":{\"forecast\":\"cloudy\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_output": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result, json!({"forecast": "cloudy"}));

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["tool_input"],
        json!(false)
    );
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["tool_output"],
        json!(true)
    );
}

#[tokio::test]
async fn remote_tool_input_preserves_named_rail_selectors() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-input-modified",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Boston\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        },
        "request_defaults": {
            "rails": {
                "tool_input": ["validate_tool_input"]
            }
        }
    })))
    .await
    .unwrap();

    let _ = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap();

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["tool_input"],
        json!(["validate_tool_input"])
    );
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["tool_output"],
        json!(false)
    );
}

#[tokio::test]
async fn remote_tool_output_preserves_named_rail_selectors() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-output-modified",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Phoenix\"},\"result\":{\"forecast\":\"cloudy\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_output": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        },
        "request_defaults": {
            "rails": {
                "tool_output": ["validate_tool_output"]
            }
        }
    })))
    .await
    .unwrap();

    let _ = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap();

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["tool_input"],
        json!(false)
    );
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["tool_output"],
        json!(["validate_tool_output"])
    );
}

#[tokio::test]
async fn remote_tool_input_invalid_modified_arguments_are_reported() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-input-invalid",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{not-json}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let error = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("modified tool arguments content that is not valid JSON"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn remote_tool_output_missing_result_field_is_reported() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-output-missing-result",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Phoenix\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_output": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let error = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("without a 'result' field"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn remote_tool_output_does_not_run_when_tool_callback_errors() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_output": true,
        "remote": {
            "endpoint": unused_local_endpoint(),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let error = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move {
                    Err(crate::error::FlowError::Internal(
                        "tool callback failed".to_string(),
                    ))
                })
            }))
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert_eq!(message, "tool callback failed");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn remote_tool_input_rewrite_with_mismatched_tool_name_is_rejected() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-input-mismatch",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"different_lookup\",\"arguments\":{\"city\":\"Boston\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let error = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("unexpected tool 'different_lookup'"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn remote_tool_input_and_output_run_in_order() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let input_response_body = json!({
        "id": "chatcmpl-tool-input-modified",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Boston\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let output_response_body = json!({
        "id": "chatcmpl-tool-output-modified",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Boston\"},\"result\":{\"forecast\":\"cloudy\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let input_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        input_response_body.len(),
        input_response_body
    )
    .into_bytes();
    let output_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        output_response_body.len(),
        output_response_body
    )
    .into_bytes();
    spawn_http_responder_sequence(listener, vec![input_response, output_response], request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let seen_args = Arc::new(Mutex::new(None::<Json>));
    let seen_args_for_call = Arc::clone(&seen_args);
    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |args| {
                *seen_args_for_call.lock().unwrap() = Some(args.clone());
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(*seen_args.lock().unwrap(), Some(json!({"city": "Boston"})));
    assert_eq!(result, json!({"forecast": "cloudy"}));

    let first_request = recv_captured_request(&request_rx);
    let first_request_json: Json = serde_json::from_slice(&first_request.body).unwrap();
    assert_eq!(first_request_json["messages"][0]["role"], json!("user"));
    assert_eq!(
        first_request_json["guardrails"]["options"]["rails"]["tool_input"],
        json!(true)
    );
    assert_eq!(
        first_request_json["guardrails"]["options"]["rails"]["tool_output"],
        json!(false)
    );

    let second_request = recv_captured_request(&request_rx);
    let second_request_json: Json = serde_json::from_slice(&second_request.body).unwrap();
    assert_eq!(second_request_json["messages"][0]["role"], json!("user"));
    assert_eq!(
        second_request_json["messages"][1]["role"],
        json!("assistant")
    );
    assert_eq!(
        second_request_json["guardrails"]["options"]["rails"]["tool_input"],
        json!(false)
    );
    assert_eq!(
        second_request_json["guardrails"]["options"]["rails"]["tool_output"],
        json!(true)
    );
}

#[tokio::test]
async fn remote_tool_checks_forward_context_state_and_thread_id() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-input-context",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Phoenix\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        },
        "request_defaults": {
            "context": {"tenant": "smoke"},
            "thread_id": "1234567890abcdef",
            "state": {"events": []}
        }
    })))
    .await
    .unwrap();

    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result, json!({"forecast": "sunny"}));

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(
        request_json["guardrails"]["context"],
        json!({"tenant": "smoke"})
    );
    assert_eq!(
        request_json["guardrails"]["thread_id"],
        json!("1234567890abcdef")
    );
    assert_eq!(request_json["guardrails"]["state"], json!({"events": []}));
}

#[tokio::test]
async fn remote_tool_only_configuration_does_not_intercept_llm_calls() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "remote": {
            "endpoint": unused_local_endpoint(),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let expected = json!({"response": "original"});
    let func: LlmExecutionNextFn = Arc::new(move |_req| {
        let expected = expected.clone();
        Box::pin(async move { Ok(expected) })
    });

    let response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(response, json!({"response": "original"}));
}
