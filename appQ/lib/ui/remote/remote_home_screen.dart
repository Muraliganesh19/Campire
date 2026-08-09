import 'dart:async';

import 'package:camera/camera.dart';
import 'package:campire/core/constants.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:campire/camera/camera_controller_service.dart';
import 'package:campire/camera/recording_manager.dart';
import 'package:campire/core/logger.dart';
import 'package:campire/models/device_info.dart';
import 'package:campire/models/sync_message.dart';
import 'package:campire/network/clock_sync_service.dart';
import 'package:campire/network/command_channel.dart';
import 'package:campire/network/discovery_service.dart';
import 'package:campire/state/app_role_provider.dart';
import 'package:campire/state/session_state_provider.dart';
import 'package:campire/ui/remote/remote_status_indicator.dart';
import 'package:campire/models/recording_session.dart';
import 'package:campire/utils/wakelock_helper.dart';

/// Screen representing the Remote camera device UI.
///
/// Shows a live camera preview and a status overlay. This device is fully
/// controller-driven — no manual camera controls.
class RemoteHomeScreen extends StatefulWidget {
  const RemoteHomeScreen({super.key});

  @override
  State<RemoteHomeScreen> createState() => _RemoteHomeScreenState();
}

class _RemoteHomeScreenState extends State<RemoteHomeScreen> {
  // ── Services ──────────────────────────────────────────────────────────
  late final String _localDeviceId;
  CameraControllerService? _cameraService;
  ClockSyncService? _clockSyncService;
  DiscoveryService? _discoveryService;
  RecordingManager? _recordingManager;
  CommandServer? _commandServer;
  CommandChannel? _controllerChannel;

  StreamSubscription<CommandChannel>? _serverSub;
  StreamSubscription<SyncMessage>? _messageSub;

  bool _cameraReady = false;
  bool _isExposureLocked = false;
  bool _isReinitializing = false;
  ResolutionPreset _selectedResolution = AppConstants.targetResolutionPreset;
  int _selectedFps = AppConstants.targetFps;
  String? _connectionLabel;

  @override
  void initState() {
    super.initState();
    _localDeviceId = 'remote_${DateTime.now().millisecondsSinceEpoch}';

    _cameraService = CameraControllerService(localDeviceId: _localDeviceId);
    _clockSyncService = ClockSyncService(localDeviceId: _localDeviceId);
    _recordingManager = RecordingManager(
      cameraService: _cameraService!,
      localDeviceId: _localDeviceId,
    );

    WakelockHelper.enable();
    _initServices();
  }

  Future<void> _initServices() async {
    // Pre-warm camera.
    final ok = await _cameraService!.initialize();
    if (mounted) {
      setState(() => _cameraReady = ok);
    }

    // Start listening for discovery broadcasts.
    _discoveryService = DiscoveryService(
      localDeviceId: _localDeviceId,
      localDisplayName: 'Remote Camera',
      localRole: DeviceRole.remote,
    );
    await _discoveryService!.start();

    // Start clock sync responder.
    await _clockSyncService!.startResponder();

    // Start command server — accept TCP connection from controller.
    _commandServer = CommandServer();
    await _commandServer!.start();

    _serverSub = _commandServer!.connections.listen(_onControllerConnected);

    if (mounted) {
      setState(() => _connectionLabel = 'Waiting for controller...');
    }
  }

  void _onControllerConnected(CommandChannel channel) {
    SyncLogger.info('Remote: controller connected');

    _controllerChannel = channel;
    _messageSub?.cancel();
    _messageSub = channel.messages.listen(_onControllerMessage);

    channel.onDisconnected = () {
      SyncLogger.info('Remote: controller disconnected');
      if (mounted) {
        setState(() => _connectionLabel = 'Controller disconnected');
        context.read<SessionStateProvider>().resetSession();
      }
    };

    if (mounted) {
      setState(() => _connectionLabel = 'Connected to controller');
    }
  }

  void _onControllerMessage(SyncMessage message) {
    if (!mounted) return;
    final session = context.read<SessionStateProvider>();

    switch (message.type) {
      case SyncMessageType.startAt:
      case SyncMessageType.startNow:
        if (session.status == SessionStatus.recording) {
          SyncLogger.info('Remote: already recording, ignoring ${message.type.wireValue}');
          return;
        }

        SyncLogger.sync('Remote: received ${message.type.wireValue}');
        session.startSession(
          _buildSession(message),
        );

        if (message.type == SyncMessageType.startAt) {
          _recordingManager!.handleStartAt(
            message: message,
            clockSyncService: _clockSyncService!,
            controllerChannel: _controllerChannel!,
          );
          // The recording manager schedules the timer. When it fires,
          // recording starts. We optimistically set to recording after
          // the scheduled delay.
          final targetMicros = message.payload['targetStartMicros'] as int;
          final offset =
              _clockSyncService!.offsets[message.senderDeviceId] ?? 0;
          final localTargetMicros = targetMicros + offset;
          final nowMicros = DateTime.now().microsecondsSinceEpoch;
          final delayMicros = localTargetMicros - nowMicros;

          Future<void>.delayed(
            Duration(microseconds: delayMicros > 0 ? delayMicros : 0),
            () {
              if (mounted) {
                session.setRecording();
                _sendStatusUpdate(true);
              }
            },
          );
        } else {
          // startNow
          _startLocalRecording();
        }
        break;

      case SyncMessageType.stop:
        SyncLogger.sync('Remote: received stop');
        _recordingManager!.handleStop();
        session.resetSession();
        _sendStatusUpdate(false);
        break;

      default:
        // Ignore messages not relevant to this role.
        break;
    }
  }

  RecordingSession _buildSession(SyncMessage msg) {
    return RecordingSession(
      sessionId: 'remote_session_${DateTime.now().millisecondsSinceEpoch}',
      scheduledStartMicros: msg.payload['targetStartMicros'] as int? ?? 0,
      participatingDeviceIds: [msg.senderDeviceId],
    );
  }

  Future<void> _startLocalRecording() async {
    if (!mounted) return;
    final session = context.read<SessionStateProvider>();
    if (session.status == SessionStatus.recording) return;

    if (session.status != SessionStatus.awaitingSync) {
      session.startSession(RecordingSession(
        sessionId: 'local_${DateTime.now().millisecondsSinceEpoch}',
        scheduledStartMicros: DateTime.now().microsecondsSinceEpoch,
        participatingDeviceIds: [_localDeviceId],
      ));
    }
    
    await _cameraService?.startRecording();
    if (mounted) session.setRecording();
    _sendStatusUpdate(true);
  }

  Future<void> _stopLocalRecording() async {
    if (!mounted) return;
    final session = context.read<SessionStateProvider>();
    
    _recordingManager?.cancelScheduledStart();
    await _cameraService?.stopRecording();
    if (mounted) session.resetSession();
    _sendStatusUpdate(false);
  }

  void _sendStatusUpdate(bool isRecording) {
    if (_controllerChannel != null && _controllerChannel!.isConnected) {
      final msg = SyncMessage(
        type: SyncMessageType.statusUpdate,
        senderDeviceId: _localDeviceId,
        timestampMicros: DateTime.now().microsecondsSinceEpoch,
        payload: {'isRecording': isRecording},
      );
      _controllerChannel!.send(msg);
    }
  }

  // ── Exposure toggle ──────────────────────────────────────────────────

  Future<void> _onSettingsChanged(ResolutionPreset? res, int? fps) async {
    if (res == null && fps == null) return;
    
    final newRes = res ?? _selectedResolution;
    final newFps = fps ?? _selectedFps;

    setState(() {
      _selectedResolution = newRes;
      _selectedFps = newFps;
      _isReinitializing = true;
    });

    await _cameraService?.reinitialize(newRes, newFps);

    if (mounted) {
      setState(() => _isReinitializing = false);
    }
  }

  Future<void> _toggleExposureLock() async {
    final service = _cameraService;
    if (service == null) return;

    if (_isExposureLocked) {
      await service.unlockExposure();
    } else {
      await service.lockExposure();
    }

    if (mounted) {
      setState(() => _isExposureLocked = service.isExposureLocked);
    }
  }

  @override
  void dispose() {
    _messageSub?.cancel();
    _serverSub?.cancel();
    _controllerChannel?.dispose();
    _commandServer?.dispose();
    _discoveryService?.dispose();
    _clockSyncService?.dispose();
    _cameraService?.dispose();
    _recordingManager?.cancelScheduledStart();
    WakelockHelper.disable();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final sessionProvider = context.watch<SessionStateProvider>();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Remote Camera'),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () {
              context.read<AppRoleProvider>().setRole(AppRole.none);
            },
          ),
        ],
      ),
      body: Stack(
        children: [
          // Camera Preview (or placeholder)
          if (_cameraReady && _cameraService?.controller != null)
            SizedBox.expand(
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreview(_cameraService!.controller!),
                  if (_isReinitializing)
                    Container(
                      color: Colors.black54,
                      child: const Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            CircularProgressIndicator(color: Colors.white),
                            SizedBox(height: 16),
                            Text('Switching...', style: TextStyle(color: Colors.white)),
                          ],
                        ),
                      ),
                    ),
                ],
              ),
            )
          else
            Container(
              color: Colors.black,
              child: const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.camera_alt, size: 64, color: Colors.white54),
                    SizedBox(height: 16),
                    Text(
                      'Initialising camera...',
                      style: TextStyle(color: Colors.white70, fontSize: 16),
                    ),
                  ],
                ),
              ),
            ),

          // Connection label
          if (_connectionLabel != null)
            Positioned(
              top: 16,
              left: 16,
              right: 16,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                decoration: BoxDecoration(
                  // ignore: deprecated_member_use
                  color: Colors.black.withOpacity(0.6),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _connectionLabel!,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white, fontSize: 14),
                ),
              ),
            ),

          // Settings and Exposure Lock Control Cluster
          if (_cameraReady)
            Positioned(
              top: 60,
              right: 16,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  // Actual negotiated values
                  if (_cameraService?.controller?.value.previewSize != null)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          // ignore: deprecated_member_use
                          color: Colors.black.withOpacity(0.5),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(
                          'Actual: ${_cameraService!.controller!.value.previewSize!.height.toInt()}p',
                          style: const TextStyle(color: Colors.white70, fontSize: 10),
                        ),
                      ),
                    ),
                  
                  // Controls container
                  Container(
                    decoration: BoxDecoration(
                      // ignore: deprecated_member_use
                      color: Colors.black.withOpacity(0.6),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    padding: const EdgeInsets.all(8),
                    child: IgnorePointer(
                      ignoring: sessionProvider.status == SessionStatus.recording || _isReinitializing,
                      child: Opacity(
                        opacity: (sessionProvider.status == SessionStatus.recording || _isReinitializing) ? 0.5 : 1.0,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            // Resolution Dropdown
                            DropdownButton<ResolutionPreset>(
                              value: _selectedResolution,
                              dropdownColor: Colors.black87,
                              style: const TextStyle(color: Colors.white, fontSize: 12),
                              isDense: true,
                              underline: const SizedBox(),
                              items: const [
                                DropdownMenuItem(value: ResolutionPreset.low, child: Text('Low')),
                                DropdownMenuItem(value: ResolutionPreset.medium, child: Text('Medium')),
                                DropdownMenuItem(value: ResolutionPreset.high, child: Text('High')),
                                DropdownMenuItem(value: ResolutionPreset.veryHigh, child: Text('Very High')),
                              ],
                              onChanged: (val) => _onSettingsChanged(val, null),
                            ),
                            const SizedBox(height: 8),
                            // FPS Dropdown
                            DropdownButton<int>(
                              value: _selectedFps,
                              dropdownColor: Colors.black87,
                              style: const TextStyle(color: Colors.white, fontSize: 12),
                              isDense: true,
                              underline: const SizedBox(),
                              items: const [
                                DropdownMenuItem(value: 30, child: Text('30 fps')),
                                DropdownMenuItem(value: 60, child: Text('60 fps')),
                                DropdownMenuItem(value: 120, child: Text('120 fps')),
                              ],
                              onChanged: (val) => _onSettingsChanged(null, val),
                            ),
                            const SizedBox(height: 12),
                            // AE Lock Toggle
                            InkWell(
                              borderRadius: BorderRadius.circular(4),
                              onTap: _toggleExposureLock,
                              child: Padding(
                                padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 2),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(
                                      _isExposureLocked ? Icons.lock : Icons.lock_open,
                                      color: _isExposureLocked ? Colors.amber : Colors.white70,
                                      size: 16,
                                    ),
                                    const SizedBox(width: 4),
                                    Text(
                                      _isExposureLocked ? 'AE Locked' : 'AE Auto',
                                      style: TextStyle(
                                        color: _isExposureLocked ? Colors.amber : Colors.white70,
                                        fontSize: 12,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),

          // Status Overlay
          Positioned(
            bottom: 24,
            left: 24,
            right: 100, // Make room for FAB
            child: RemoteStatusIndicator(
              status: sessionProvider.status,
            ),
          ),

          // Local Shutter Button
          Positioned(
            bottom: 24,
            right: 24,
            child: FloatingActionButton(
              onPressed: sessionProvider.status == SessionStatus.recording
                  ? _stopLocalRecording
                  : _startLocalRecording,
              backgroundColor: sessionProvider.status == SessionStatus.recording ? Colors.orange : Colors.red,
              child: Icon(
                sessionProvider.status == SessionStatus.recording ? Icons.stop : Icons.fiber_manual_record,
                color: Colors.white,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
