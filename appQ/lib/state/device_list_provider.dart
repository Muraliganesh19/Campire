import 'package:flutter/foundation.dart';
import 'package:campire/models/device_info.dart';

/// Provider managing the controller's list of discovered and connected remotes.
///
/// Updated by:
/// * [DiscoveryService] — new devices found
/// * [ClockSyncService] — synced status
/// * [CommandChannel] — connection status changes
class DeviceListProvider extends ChangeNotifier {
  final List<DeviceInfo> _devices = [];

  /// An unmodifiable snapshot of the current device list.
  List<DeviceInfo> get devices => List.unmodifiable(_devices);

  /// Number of devices with a given [ConnectionStatus].
  int countByStatus(ConnectionStatus status) =>
      _devices.where((d) => d.connectionStatus == status).length;

  /// Whether at least one device is synced and ready to record.
  bool get hasSyncedDevices =>
      _devices.any((d) => d.connectionStatus == ConnectionStatus.synced);

  /// Adds a new device or updates an existing one (matched by [DeviceInfo.id]).
  ///
  /// If a device with the same ID already exists, it is replaced. This allows
  /// callers to update connection status, IP address, etc. by passing a new
  /// [DeviceInfo] with the same ID.
  void updateDevice(DeviceInfo device) {
    final index = _devices.indexWhere((d) => d.id == device.id);
    if (index >= 0) {
      _devices[index] = device;
    } else {
      _devices.add(device);
    }
    notifyListeners();
  }

  /// Updates only the [ConnectionStatus] for the device with [deviceId].
  ///
  /// No-op if the device is not found.
  void updateStatus(String deviceId, ConnectionStatus status) {
    final index = _devices.indexWhere((d) => d.id == deviceId);
    if (index < 0) return;
    _devices[index] = _devices[index].copyWith(connectionStatus: status);
    notifyListeners();
  }

  /// Removes the device with the given [deviceId].
  void removeDevice(String deviceId) {
    _devices.removeWhere((d) => d.id == deviceId);
    notifyListeners();
  }

  /// Returns the [DeviceInfo] for [deviceId], or `null` if not found.
  DeviceInfo? getDevice(String deviceId) {
    try {
      return _devices.firstWhere((d) => d.id == deviceId);
    } catch (_) {
      return null;
    }
  }

  /// Clears all devices (e.g. on disconnect / reset / role switch).
  void clear() {
    _devices.clear();
    notifyListeners();
  }
}
