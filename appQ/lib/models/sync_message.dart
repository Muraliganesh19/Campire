/// The type of message exchanged between controller and remotes over the
/// command channel.
///
/// Wire format uses snake_case strings (e.g. `"handshake_request"`) so that
/// the JSON is readable and language-neutral.
enum SyncMessageType {
  discover('discover'),
  discoverResponse('discover_response'),
  clockSyncRequest('clock_sync_request'),
  clockSyncResponse('clock_sync_response'),
  handshakeRequest('handshake_request'),
  handshakeResponse('handshake_response'),
  startAt('start_at'),
  startNow('start_now'),
  statusUpdate('status_update'),
  ack('ack'),
  stop('stop');

  const SyncMessageType(this.wireValue);

  /// The string written into JSON's `"type"` field.
  final String wireValue;

  /// Parses a [SyncMessageType] from its [wireValue].
  /// Throws [ArgumentError] if [value] is unrecognised — we want to fail
  /// loudly on unknown messages rather than silently swallow them.
  static SyncMessageType fromWire(String value) =>
      SyncMessageType.values.firstWhere(
        (e) => e.wireValue == value,
        orElse: () => throw ArgumentError('Unknown SyncMessageType: $value'),
      );
}

/// A single message in the sync protocol.
///
/// Immutable, serialisable, and deliberately free of business logic.
/// [payload] carries type-specific key/value data (clock offsets, session IDs,
/// etc.) — kept as a flat [Map] to avoid coupling the model layer to every
/// future protocol extension.
///
/// [timestampMicros] is microseconds since epoch. We use a raw [int] rather
/// than [DateTime] because the value will be compared against clock-synced
/// timestamps and arithmetic on ints is simpler / more predictable.
class SyncMessage {
  final SyncMessageType type;
  final String senderDeviceId;
  final int timestampMicros;
  final Map<String, dynamic> payload;

  const SyncMessage({
    required this.type,
    required this.senderDeviceId,
    required this.timestampMicros,
    this.payload = const {},
  });

  Map<String, dynamic> toJson() {
    return {
      'type': type.wireValue,
      'senderDeviceId': senderDeviceId,
      'timestampMicros': timestampMicros,
      'payload': payload,
    };
  }

  factory SyncMessage.fromJson(Map<String, dynamic> json) {
    return SyncMessage(
      type: SyncMessageType.fromWire(json['type'] as String),
      senderDeviceId: json['senderDeviceId'] as String,
      timestampMicros: json['timestampMicros'] as int,
      payload: (json['payload'] as Map<String, dynamic>?) ?? const {},
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is SyncMessage &&
          runtimeType == other.runtimeType &&
          type == other.type &&
          senderDeviceId == other.senderDeviceId &&
          timestampMicros == other.timestampMicros &&
          _mapsEqual(payload, other.payload);

  @override
  int get hashCode {
    // Sort keys for deterministic ordering; hash each key-value pair.
    final sortedKeys = payload.keys.toList()..sort();
    final payloadHash = Object.hashAll(
      sortedKeys.expand((k) => [k, payload[k]]),
    );
    return Object.hash(type, senderDeviceId, timestampMicros, payloadHash);
  }

  @override
  String toString() =>
      'SyncMessage(type: ${type.wireValue}, sender: $senderDeviceId, '
      'ts: $timestampMicros, payload: $payload)';

  /// Shallow equality check for two maps — sufficient here because payload
  /// values are JSON primitives.
  static bool _mapsEqual(Map<String, dynamic> a, Map<String, dynamic> b) {
    if (a.length != b.length) return false;
    for (final key in a.keys) {
      if (!b.containsKey(key) || a[key] != b[key]) return false;
    }
    return true;
  }
}
