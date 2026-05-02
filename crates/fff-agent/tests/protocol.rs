use fff_agent::{AgentRequest, AgentResponse, GrepMode};

#[test]
fn grep_request_serializes_literal_plain_mode_by_default() {
    let request = AgentRequest::Grep {
        query: "attachApplicationScoring(".to_string(),
        limit: 20,
        mode: GrepMode::Plain,
        json: false,
    };

    let encoded = serde_json::to_string(&request).unwrap();
    let decoded: AgentRequest = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, request);
    assert!(encoded.contains("\"mode\":\"plain\""));
}

#[test]
fn response_round_trips_text_and_shutdown() {
    let text = AgentResponse::Text {
        text: "src/main.rs:10: fn main()".to_string(),
    };
    let shutdown = AgentResponse::Shutdown;

    assert_eq!(
        serde_json::from_str::<AgentResponse>(&serde_json::to_string(&text).unwrap()).unwrap(),
        text
    );
    assert_eq!(
        serde_json::from_str::<AgentResponse>(&serde_json::to_string(&shutdown).unwrap()).unwrap(),
        shutdown
    );
}
