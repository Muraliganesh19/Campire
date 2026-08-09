import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:campire/camera/recording_manager.dart';
import 'package:campire/core/logger.dart';
import 'package:campire/models/device_info.dart';
import 'package:campire/models/recording_session.dart';
import 'package:campire/models/sync_message.dart';
import 'package:campire/network/clock_sync_service.dart';
import 'package:campire/network/command_channel.dart';
import 'package:campire/network/discovery_service.dart';
import 'package:campire/state/app_role_provider.dart';
import 'package:campire/state/device_list_provider.dart';
import 'package:campire/state/session_state_provider.dart';
import 'package:campire/ui/controller/device_list_tile.dart';
import 'package:campire/utils/wakelock_helper.dart';

/// Main screen for the Controller role.
///
/// Manages device discovery, clock sync, and triggers synchronised recording
/// across all connected remotes.
class ControllerHomeScreen extends StatefulWidget {
  const ControllerHomeScreen({super.key});

  @override
  State<ControllerHomeScreen> createState() => _ControllerHomeScreenState();
}

class _ControllerHomeScreenState extends State<ControllerHomeScreen> {
  // ── Services ──────────────────────────────────────────────────────────
  late final String _localDeviceId;
  DiscoveryService? _discoveryService;
  ClockSyncService? _clockSyncService;
  RecordingManager? _recordingManager;

  /// One [CommandChannel] per connected remote, keyed by device ID.
  final Map<String, CommandChannel> _channels = {};

  StreamSubscription<DeviceInfo>? _discoverySub;
  final List<StreamSubscription<SyncMessage>> _channelSubs = [];

  bool _isScanning = false;

  @override
  void initState() {
    super.initState();
    // Simple device ID — good enough for v1 local-network use.
    _localDeviceId = 'controller_${DateTime.now().millisecondsSinceEpoch}';

    _clockSyncService = ClockSyncService(localDeviceId: _localDeviceId);
    _recordingManager = RecordingManager(
      localDeviceId: _localDeviceId,
    );

    WakelockHelper.enable();
  }

  @override
  void dispose() {
    _discoverySub?.cancel();
    _discoveryService?.dispose();
    _clockSyncService?.dispose();
    for (final sub in _channelSubs) {
      sub.cancel();
    }
    for (final ch in _channels.values) {
      ch.dispose();
    }
    WakelockHelper.disable();
    super.dispose();
  }

  // ── Discovery ─────────────────────────────────────────────────────────

  Future<void> _startScan() async {
    if (_isScanning) return;
    setState(() => _isScanning = true);

    _discoveryService?.dispose();
    _discoveryService = DiscoveryService(
      localDeviceId: _localDeviceId,
      localDisplayName: 'Controller',
      localRole: DeviceRole.controller,
    );

    _discoverySub?.cancel();
    _discoverySub = _discoveryService!.discoveredRemotes.listen(
      _onDeviceDiscovered,
    );

    await _discoveryService!.start();

    // Let the scan run for the discovery timeout, then stop broadcasting.
    Future<void>.delayed(const Duration(seconds: 10), () {
      if (mounted) {
        _discoveryService?.stop();
        setState(() => _isScanning = false);
      }
    });
  }

  Future<void> _onDeviceDiscovered(DeviceInfo device) async {
    if (!mounted) return;
    final deviceList = context.read<DeviceListProvider>();

    // Skip if we already know this device and it's connected or better.
    final existing = deviceList.getDevice(device.id);
    if (existing != null &&
        existing.connectionStatus != ConnectionStatus.disconnected) {
      return;
    }

    // Add/update as connecting.
    deviceList.updateDevice(
      device.copyWith(connectionStatus: ConnectionStatus.connecting),
    );

    // Open a TCP command channel to the remote.
    try {
      final channel = await CommandChannel.connect(
        ipAddress: device.ipAddress,
      );

      _channels[device.id] = channel;

      // Listen for messages from this remote.
      final sub = channel.messages.listen((msg) {
        _handleRemoteMessage(device.id, msg);
      });
      _channelSubs.add(sub);

      channel.onDisconnected = () {
        SyncLogger.info('Controller: lost connection to ${device.id}');
        if (mounted) {
          deviceList.updateStatus(device.id, ConnectionStatus.disconnected);
        }
        _channels.remove(device.id);
      };

      deviceList.updateDevice(
        device.copyWith(connectionStatus: ConnectionStatus.connected),
      );
      SyncLogger.info('Controller: ${device.id} connecting -> connected');

      // Immediately run clock sync.
      await _runClockSync(device);
    } catch (e) {
      SyncLogger.error('Controller: failed to connect to ${device.id}', e);
      if (mounted) {
        deviceList.updateStatus(device.id, ConnectionStatus.disconnected);
      }
    }
  }

  Future<void> _runClockSync(DeviceInfo device) async {
    if (!mounted) return;
    final deviceList = context.read<DeviceListProvider>();

    try {
      await _clockSyncService!.computeOffset(device);
      if (mounted) {
        deviceList.updateStatus(device.id, ConnectionStatus.synced);
        SyncLogger.info('Controller: ${device.id} connected -> synced');
      }
    } catch (e) {
      SyncLogger.error('Controller: clock sync failed for ${device.id}', e);
      // Device stays "connected" but not synced — controller can still
      // retry sync manually if needed in a future version.
    }
  }

  void _handleRemoteMessage(String deviceId, SyncMessage message) {
    if (message.type == SyncMessageType.ack) {
      _recordingManager?.handleAck(message);
      if (mounted) {
        final deviceList = context.read<DeviceListProvider>();
        final device = deviceList.getDevice(deviceId);
        if (device != null) {
          deviceList.updateDevice(device.copyWith(isRecording: true));
        }
      }
    } else if (message.type == SyncMessageType.statusUpdate) {
      final isRecording = message.payload['isRecording'] as bool? ?? false;
      if (mounted) {
        final deviceList = context.read<DeviceListProvider>();
        final device = deviceList.getDevice(deviceId);
        if (device != null) {
          deviceList.updateDevice(device.copyWith(isRecording: isRecording));
        }
      }
    }
  }

  // ── Recording ─────────────────────────────────────────────────────────

  Future<void> _startRecording() async {
    if (!mounted) return;
    final session = context.read<SessionStateProvider>();
    final deviceList = context.read<DeviceListProvider>();

    // Collect synced devices.
    final syncedDevices = deviceList.devices
        .where((d) => d.connectionStatus == ConnectionStatus.synced)
        .toList();

    if (syncedDevices.isEmpty) return;

    // Create session.
    final sessionId = 'session_${DateTime.now().millisecondsSinceEpoch}';
    session.startSession(RecordingSession(
      sessionId: sessionId,
      scheduledStartMicros: 0, // will be set by triggerSyncRecording
      participatingDeviceIds: [
        _localDeviceId,
        ...syncedDevices.map((d) => d.id),
      ],
    ));

    try {
      await _recordingManager!.triggerSyncRecording(
        syncedDevices: syncedDevices,
        channels: _channels,
      );
      if (mounted) session.setRecording();
    } catch (e) {
      SyncLogger.error('Controller: failed to start sync recording', e);
      if (mounted) session.setError('Failed to start recording: $e');
    }
  }

  Future<void> _stopRecording() async {
    if (!mounted) return;
    final session = context.read<SessionStateProvider>();

    final activeChannels = _channels.values
        .where((ch) => ch.isConnected)
        .toList();

    try {
      await _recordingManager!.triggerStop(remoteChannels: activeChannels);
    } catch (e) {
      SyncLogger.error('Controller: failed to stop recording', e);
    }
    if (mounted) session.resetSession();
  }

  // ── Build ─────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final devicesProvider = context.watch<DeviceListProvider>();
    final sessionProvider = context.watch<SessionStateProvider>();
    final isRecording = sessionProvider.status == SessionStatus.recording;
    final hasSynced = devicesProvider.hasSyncedDevices;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Controller Dashboard'),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () {
              context.read<AppRoleProvider>().setRole(AppRole.none);
            },
          ),
        ],
      ),
      body: Column(
        children: [
          // Role Banner
          Container(
            color: Colors.deepPurple.shade50,
            padding: const EdgeInsets.all(12),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Role: CONTROLLER',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.deepPurple,
                  ),
                ),
                Text('Devices: ${devicesProvider.devices.length}/6'),
              ],
            ),
          ),

          // Device List
          Expanded(
            child: devicesProvider.devices.isEmpty
                ? const Center(
                    child: Text(
                      'No remote devices discovered yet.\n'
                      'Tap Scan to find devices on the network.',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.grey),
                    ),
                  )
                : ListView.builder(
                    itemCount: devicesProvider.devices.length,
                    itemBuilder: (context, idx) {
                      final device = devicesProvider.devices[idx];
                      return DeviceListTile(
                        device: device,
                        onTrigger: () {
                          final channel = _channels[device.id];
                          if (channel != null) {
                            if (device.isRecording) {
                              _recordingManager?.triggerSingleStop(device, channel);
                            } else {
                              _recordingManager?.triggerSingleRecording(device, channel);
                            }
                          }
                        },
                      );
                    },
                  ),
          ),

          // Action Buttons
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: Row(
                children: [
                  // Scan button
                  Expanded(
                    child: ElevatedButton(
                      onPressed: _isScanning ? null : _startScan,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.grey.shade200,
                        foregroundColor: Colors.black,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                      child: Text(_isScanning ? 'Scanning...' : 'Scan'),
                    ),
                  ),
                  const SizedBox(width: 16),

                  // Record / Stop button
                  Expanded(
                    flex: 2,
                    child: ElevatedButton(
                      onPressed: isRecording
                          ? _stopRecording
                          : (hasSynced ? _startRecording : null),
                      style: ElevatedButton.styleFrom(
                        backgroundColor:
                            isRecording ? Colors.orange : Colors.red,
                        foregroundColor: Colors.white,
                        disabledBackgroundColor: Colors.grey.shade300,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                      child: Text(
                        isRecording ? 'STOP RECORDING' : 'START SYNC RECORD',
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
