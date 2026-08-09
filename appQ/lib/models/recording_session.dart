import 'package:flutter/foundation.dart' show listEquals;

/// Immutable snapshot of a synchronised recording session.
///
/// [scheduledStartMicros] is microseconds since epoch — a raw [int] so it can
/// be directly compared against clock-synced timestamps without DateTime
/// conversion overhead or timezone ambiguity.
///
/// [participatingDeviceIds] is the list of device IDs enrolled in this session
/// (both controller and remotes). We store IDs rather than full [DeviceInfo]
/// objects to keep the model lightweight and avoid circular dependencies.
class RecordingSession {
  final String sessionId;
  final int scheduledStartMicros;
  final List<String> participatingDeviceIds;

  const RecordingSession({
    required this.sessionId,
    required this.scheduledStartMicros,
    required this.participatingDeviceIds,
  });

  /// Returns a copy with the given fields replaced.
  RecordingSession copyWith({
    String? sessionId,
    int? scheduledStartMicros,
    List<String>? participatingDeviceIds,
  }) {
    return RecordingSession(
      sessionId: sessionId ?? this.sessionId,
      scheduledStartMicros: scheduledStartMicros ?? this.scheduledStartMicros,
      participatingDeviceIds:
          participatingDeviceIds ?? this.participatingDeviceIds,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'sessionId': sessionId,
      'scheduledStartMicros': scheduledStartMicros,
      'participatingDeviceIds': participatingDeviceIds,
    };
  }

  factory RecordingSession.fromJson(Map<String, dynamic> json) {
    return RecordingSession(
      sessionId: json['sessionId'] as String,
      scheduledStartMicros: json['scheduledStartMicros'] as int,
      participatingDeviceIds:
          (json['participatingDeviceIds'] as List<dynamic>)
              .cast<String>()
              .toList(),
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is RecordingSession &&
          runtimeType == other.runtimeType &&
          sessionId == other.sessionId &&
          scheduledStartMicros == other.scheduledStartMicros &&
          listEquals(participatingDeviceIds, other.participatingDeviceIds);

  @override
  int get hashCode =>
      Object.hash(sessionId, scheduledStartMicros, Object.hashAll(participatingDeviceIds));

  @override
  String toString() =>
      'RecordingSession(id: $sessionId, startMicros: $scheduledStartMicros, '
      'devices: $participatingDeviceIds)';
}
