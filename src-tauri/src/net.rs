use std::time::Duration;

/// GETs `url` and reports whether it responded with a 2xx status.
pub async fn check_health(url: &str) -> bool {
    reqwest::Client::new()
        .get(url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .map(|res| res.status().is_success())
        .unwrap_or(false)
}
