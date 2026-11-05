/// The role a device plays in a recording session.
enum DeviceRole {
  controller,
  remote;

  /// Parses a [DeviceRole] from its [name] string.
  /// Returns [DeviceRole.remote] if [value] is unrecognised.
  static DeviceRole fromString(String value) =>
      DeviceRole.values.firstWhere(
        (e) => e.name == value,
        orElse: () => DeviceRole.remote,
      );
}

/// Connection lifecycle of a remote device as seen by the controller.
enum ConnectionStatus {
  disconnected,
  connecting,
  connected,
  synced;

  /// Parses a [ConnectionStatus] from its [name] string.
  /// Returns [ConnectionStatus.disconnected] if [value] is unrecognised.
  static ConnectionStatus fromString(String value) =>
      ConnectionStatus.values.firstWhere(
        (e) => e.name == value,
        orElse: () => ConnectionStatus.disconnected,
      );
}

/// Immutable snapshot of a device on the local network.
///
/// Used by both controller (to track remotes) and remote (to describe itself
/// during discovery/handshake). Contains no business logic — serialisation
/// only.
class DeviceInfo {
  final String id;
  final String displayName;
  final String ipAddress;
  final DeviceRole role;
  final ConnectionStatus connectionStatus;
  final bool isRecording;

  const DeviceInfo({
    required this.id,
    required this.displayName,
    required this.ipAddress,
    required this.role,
    required this.connectionStatus,
    this.isRecording = false,
  });

  /// Returns a copy with the given fields replaced.
  DeviceInfo copyWith({
    String? id,
    String? displayName,
    String? ipAddress,
    DeviceRole? role,
    ConnectionStatus? connectionStatus,
    bool? isRecording,
  }) {
    return DeviceInfo(
      id: id ?? this.id,
      displayName: displayName ?? this.displayName,
      ipAddress: ipAddress ?? this.ipAddress,
      role: role ?? this.role,
      connectionStatus: connectionStatus ?? this.connectionStatus,
      isRecording: isRecording ?? this.isRecording,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'displayName': displayName,
      'ipAddress': ipAddress,
      'role': role.name,
      'connectionStatus': connectionStatus.name,
      'isRecording': isRecording,
    };
  }

  factory DeviceInfo.fromJson(Map<String, dynamic> json) {
    return DeviceInfo(
      id: json['id'] as String,
      displayName: json['displayName'] as String,
      ipAddress: json['ipAddress'] as String,
      role: DeviceRole.fromString(json['role'] as String),
      connectionStatus:
          ConnectionStatus.fromString(json['connectionStatus'] as String),
      isRecording: json['isRecording'] as bool? ?? false,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is DeviceInfo &&
          runtimeType == other.runtimeType &&
          id == other.id &&
          displayName == other.displayName &&
          ipAddress == other.ipAddress &&
          role == other.role &&
          connectionStatus == other.connectionStatus &&
          isRecording == other.isRecording;

  @override
  int get hashCode => Object.hash(id, displayName, ipAddress, role, connectionStatus, isRecording);

  @override
  String toString() =>
      'DeviceInfo(id: $id, displayName: $displayName, ip: $ipAddress, '
      'role: ${role.name}, status: ${connectionStatus.name}, isRecording: $isRecording)';
}
