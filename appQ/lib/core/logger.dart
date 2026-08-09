import 'package:flutter/foundation.dart';
import 'package:intl/intl.dart';

/// Simple timestamped logger to help trace network sync events with millisecond precision.
class SyncLogger {
  static final DateFormat _dateFormat = DateFormat('yyyy-MM-dd HH:mm:ss.SSS');

  /// Logs an informational message with a timestamp.
  static void info(String message) {
    _log('INFO', message);
  }

  /// Logs an error message with a timestamp.
  static void error(String message, [Object? error, StackTrace? stackTrace]) {
    _log('ERROR', '$message${error != null ? ' | Error: $error' : ''}');
    if (stackTrace != null) {
      // TODO: Log stack trace to log file if available
      debugPrint(stackTrace.toString());
    }
  }

  /// Logs a synchronization-specific timing event.
  static void sync(String message) {
    _log('SYNC', message);
  }

  static void _log(String level, String message) {
    final timestamp = _dateFormat.format(DateTime.now());
    // TODO: Write to a local file for post-session post-mortems if required
    debugPrint('[$timestamp] [$level] $message');
  }
}
