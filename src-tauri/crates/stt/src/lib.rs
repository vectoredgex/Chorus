//! Speech-to-text integration for the Rhema application.
//!
//! Provides real-time transcription via multiple providers:
//! - **Deepgram** (cloud) — WebSocket streaming with keyword boosting
//! - **Whisper** (local) — offline inference via whisper.cpp
//!
//! # Key types
//!
//! - [`SttProvider`] — trait for swappable STT backends
//! - [`DeepgramClient`] — Deepgram WebSocket/REST provider
//! - [`TranscriptEvent`] — streaming transcript events (partial, final, etc.)
//! - [`SttConfig`] — API configuration
//! - [`SttError`] — error type for STT operations
//!
//! # Feature flags
//!
//! - `rest-fallback` — enables REST API fallback client
//! - `whisper` — enables local Whisper STT provider
//! - `parakeet` — enables local NVIDIA Parakeet STT provider (English, CPU)

pub mod cleanup;
pub mod deepgram;
pub mod error;
pub mod keyterms;
pub mod provider;
pub mod rest;
pub mod types;

#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(feature = "parakeet")]
pub mod parakeet;

pub use deepgram::DeepgramClient;
pub use error::SttError;
pub use keyterms::bible_keyterms;
pub use provider::SttProvider;
pub use types::{SttConfig, TranscriptEvent, Word};

#[cfg(feature = "whisper")]
pub use whisper::WhisperProvider;

#[cfg(feature = "parakeet")]
pub use parakeet::ParakeetProvider;
