import 'dart:io';
import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'package:campire/camera/camera_controller_service.dart';
import 'package:campire/camera/recording_manager.dart';
import 'package:campire/models/device_info.dart';
import 'package:campire/models/sync_message.dart';
import 'package:campire/network/clock_sync_service.dart';
import 'package:campire/network/discovery_service.dart';
import 'package:campire/network/node_client_service.dart';

import 'package:campire/ui/remote/widgets/floating_header_hud.dart';
import 'package:campire/ui/remote/widgets/floating_connection_bar.dart';
import 'package:campire/ui/remote/widgets/recording_status_indicator.dart';
import 'package:campire/ui/remote/widgets/camera_grid_overlay.dart';
import 'package:campire/ui/remote/widgets/remote_shutter_ring.dart';

class RemoteHomeScreen extends StatefulWidget {
  final String deviceName;

  const RemoteHomeScreen({super.key, required this.deviceName});

  @override
  State<RemoteHomeScreen> createState() => _RemoteHomeScreenState();
}

class _RemoteHomeScreenState extends State<RemoteHomeScreen> {
  late final String _localDeviceId;
  CameraControllerService? _cameraService;
  ClockSyncService? _clockSyncService;
  DiscoveryService? _discoveryService;
  RecordingManager? _recordingManager;
  NodeClientService? _nodeClientService;

  bool _cameraReady = false;
  
  bool _isConnected = false;
  String _hostIp = '';
  final TextEditingController _ipController = TextEditingController();

  // Recording State
  bool _isRecording = false;
  DateTime? _recordingStartTime;
  Duration _recordingDuration = Duration.zero;
  Timer? _uiTimer;

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

    _initCamera();
    _startDiscovery();
    
    // Poll for recording state to update UI overlays
    _uiTimer = Timer.periodic(const Duration(milliseconds: 200), _checkRecordingState);
  }

  void _checkRecordingState(Timer t) {
    if (!mounted) return;
    final recordingNow = _cameraService?.isRecording ?? false;
    
    if (recordingNow && !_isRecording) {
      setState(() {
        _isRecording = true;
        _recordingStartTime = DateTime.now();
        _recordingDuration = Duration.zero;
      });
    } else if (!recordingNow && _isRecording) {
      setState(() {
        _isRecording = false;
        _recordingStartTime = null;
      });
    } else if (recordingNow && _recordingStartTime != null) {
      setState(() {
        _recordingDuration = DateTime.now().difference(_recordingStartTime!);
      });
    }
  }

  Future<void> _initCamera() async {
    final ok = await _cameraService!.initialize();
    if (mounted) {
      setState(() => _cameraReady = ok);
    }
  }

  void _startDiscovery() async {
    _discoveryService = DiscoveryService(
      localDeviceId: _localDeviceId,
      localDisplayName: 'Remote Camera',
      localRole: DeviceRole.remote,
    );
    await _discoveryService!.start();
  }

  void _connectToHost(String ip) {
    if (ip.isEmpty) return;
    
    _nodeClientService?.disconnect();
    _nodeClientService = NodeClientService(
      localDeviceId: _localDeviceId,
      deviceName: widget.deviceName,
      hostIp: ip,
      onMessageReceived: _handleMessage,
      onConnected: () {
        if (mounted) {
          setState(() {
            _isConnected = true;
            _hostIp = ip;
          });
        }
      },
      onDisconnected: () {
        if (mounted) {
          setState(() {
            _isConnected = false;
            _hostIp = '';
          });
        }
      },
    );
    _nodeClientService!.connect();
  }

  void _handleMessage(SyncMessage message) async {
    if (!mounted) return;

    if (message.type == SyncMessageType.clockSyncRequest) {
       final response = _clockSyncService?.handleRequest(message);
       if (response != null) _nodeClientService?.send(response);
    } else if (message.type == SyncMessageType.startAt) {
       _recordingManager?.handleStartAt(
          message: message,
          clockSyncService: _clockSyncService!,
          sendAck: (ack) => _nodeClientService?.send(ack),
       );
    } else if (message.type == SyncMessageType.stop) {
       final xfile = await _recordingManager?.handleStop();
       if (xfile != null) {
          await _nodeClientService?.uploadVideo(File(xfile.path));
       }
    }
  }

  @override
  void dispose() {
    _uiTimer?.cancel();
    _cameraService?.dispose();
    _clockSyncService?.dispose();
    _discoveryService?.dispose();
    _nodeClientService?.disconnect();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        bottom: false,
        child: Column(
          children: [
            // TOP: Viewport Stack
            Expanded(
              child: Stack(
                fit: StackFit.expand,
                children: [
                  // Base: Camera Preview
                  if (_cameraReady && _cameraService?.controller != null)
                    CameraPreview(_cameraService!.controller!)
                  else
                    const Center(child: CircularProgressIndicator(color: Colors.white)),
                  
                  // Overlay: 3x3 Grid Lines & Reticle
                  if (_cameraReady)
                    const Positioned.fill(child: CameraGridOverlay()),

                  // Overlay: Top Navigation Bar
                  const Positioned(
                    top: 0,
                    left: 0,
                    right: 0,
                    child: FloatingHeaderHUD(),
                  ),

                  // Overlay: Floating Timer Pill
                  Positioned(
                    bottom: 24,
                    left: 0,
                    right: 0,
                    child: Center(
                      child: RecordingStatusIndicator(
                        isRecording: _isRecording,
                        duration: _recordingDuration,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            
            // BOTTOM: Dark Control Panel
            Container(
              color: Colors.black,
              padding: const EdgeInsets.only(left: 24, right: 24, top: 24, bottom: 48),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Row 1: Networking Bar
                  NetworkingBar(
                    isConnected: _isConnected,
                    ipController: _ipController,
                    connectedIp: _hostIp,
                    onConnect: () => _connectToHost(_ipController.text),
                  ),
                  const SizedBox(height: 48),
                  
                  // Row 2: Shutter Control
                  RemoteShutterRing(isRecording: _isRecording),
                  const SizedBox(height: 16),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
