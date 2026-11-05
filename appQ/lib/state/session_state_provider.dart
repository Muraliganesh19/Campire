import 'package:flutter/foundation.dart';
import 'package:campire/models/recording_session.dart';

/// The high-level session state, driven by the recording lifecycle.
///
/// Used by the UI to decide what to show (idle screen, recording indicator,
/// error dialog, etc.).
enum SessionStatus {
  /// No active session — default state.
  idle,

  /// Controller has sent start_at, waiting for acks from remotes.
  awaitingSync,

  /// All devices are recording.
  recording,

  /// An error occurred during the session.
  error,
}

/// Provider managing the active recording session state.
///
/// Drives what the UI shows — idle screen, syncing spinner, recording
/// indicator, or error message.
class SessionStateProvider extends ChangeNotifier {
  SessionStatus _status = SessionStatus.idle;
  RecordingSession? _currentSession;
  String? _errorMessage;

  SessionStatus get status => _status;
  RecordingSession? get currentSession => _currentSession;
  String? get errorMessage => _errorMessage;

  /// Transitions to [SessionStatus.awaitingSync] with the given session.
  void startSession(RecordingSession session) {
    _currentSession = session;
    _status = SessionStatus.awaitingSync;
    _errorMessage = null;
    notifyListeners();
  }

  /// Transitions to [SessionStatus.recording].
  void setRecording() {
    _status = SessionStatus.recording;
    _errorMessage = null;
    notifyListeners();
  }

  /// Transitions to [SessionStatus.error] with a message.
  void setError(String message) {
    _status = SessionStatus.error;
    _errorMessage = message;
    notifyListeners();
  }

  /// Resets to [SessionStatus.idle] and clears the session.
  void resetSession() {
    _status = SessionStatus.idle;
    _currentSession = null;
    _errorMessage = null;
    notifyListeners();
  }
}
